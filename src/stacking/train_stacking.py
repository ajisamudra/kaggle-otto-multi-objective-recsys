import click
import polars as pl
import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import datetime
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from src.training.train import downsample
from src.utils.constants import (
    CFG,
    write_json,
    check_directory,
    get_processed_local_validation_dir,
    get_data_output_local_submission_dir,  # scoring output dir
    get_data_output_local_ensemble_submission_dir,
    get_data_output_submission_dir,
    get_data_output_ensemble_submission_dir,
    get_artifacts_tuning_dir,
)
from src.metrics.model_evaluation import (
    summarise_feature_importance,
    plot_and_save_feature_importance,
    plot_and_save_score_distribution,
)
from src.model.model import StackingModels
from src.utils.constants import get_artifacts_training_dir, check_directory
from src.stacking.score_stacking import scoring_stacking
from src.stacking.make_submission_stacking import make_submission
from src.scoring.eval_submission import eval_submission

from src.metrics.submission_evaluation import measure_recall
from src.utils.memory import freemem, round_float_3decimals
from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"

CFG_MODEL = {
    "clicks_models": [
        "2022-12-25_clicks_cat_ranker_45439_89900",  # cv 0.56873
        # "2022-12-26_clicks_lgbm_ranker_48370_89600",  # cv 0.567242
        "2023-01-02_clicks_cat_ranker_60593_91362",  # cv 0.57012
        "2023-01-02_clicks_lgbm_ranker_61844_91502",  # cv 0.56920
        "2023-01-10_clicks_cat_ranker_61483_91316",  # cv 0.56993
        # "2023-01-17_clicks_cat_regressor_69418_93202",  # cv 0.5696
        "2023-01-17_clicks_cat_classifier_69783_92833",  # cv 0.5690
        # "2023-01-17_clicks_cat_ranker_71442_93554",  # cv 0.56974
    ],
    "carts_models": [
        "2022-12-25_carts_cat_ranker_65389_94260",  # cv 0.56873
        # "2022-12-26_carts_lgbm_ranker_66196_93867",  # cv 0.567242
        "2023-01-02_carts_cat_ranker_75708_94697",  # cv 0.57012
        "2023-01-02_carts_lgbm_ranker_75879_94504",  # cv 0.56920
        "2023-01-10_carts_cat_ranker_76221_94688",  # cv 0.56993
        # "2023-01-17_carts_cat_regressor_70657_93800",  # cv 0.5696
        "2023-01-17_carts_cat_classifier_72864_93779",  # cv 0.5690
        # "2023-01-17_carts_cat_ranker_74566_93914",  # cv 0.56974
    ],
    "orders_models": [
        "2022-12-25_orders_cat_ranker_80132_96912",  # cv 0.56873
        # "2022-12-26_orders_lgbm_ranker_78372_95814",  # cv 0.567242
        "2023-01-02_orders_cat_ranker_86779_97309",  # cv 0.57012
        "2023-01-02_orders_lgbm_ranker_85371_96813",  # cv 0.56920
        "2023-01-10_orders_cat_ranker_87315_97301",  # cv 0.56993
        # "2023-01-17_orders_cat_regressor_86921_97516",  # cv 0.5696
        "2023-01-17_orders_cat_classifier_89226_97489",  # cv 0.5690
        # "2023-01-17_orders_cat_ranker_88789_97503",  # cv 0.56974
    ],
}


def train_stacking(algo: str):
    input_path = ""
    output_path = ""
    week_model = "w2"
    unique_model_names = []
    for EVENT in ["clicks", "carts", "orders"]:
        logging.info(f"start reading stacking dataset in event: {EVENT.upper()}")
        ARTIFACTS = CFG_MODEL[f"{EVENT}_models"]
        event_model_name = ""  # for accessing submission later
        i = 0
        event_model_name = "stacking"
        output_path = get_data_output_local_ensemble_submission_dir(
            event=EVENT, model=event_model_name, week_model=week_model
        )
        tmp_path = f"{output_path}/test_{i}_{EVENT}_stacking_dataset.parquet"
        df = pl.read_parquet(tmp_path)

        # preprocess: fill_null with median for each cols
        for fea in ARTIFACTS:
            df = df.with_column(
                pl.col(fea).fill_null(pl.median(fea)),
            )

        # # preprocess: create pow2/4 as new features
        for fea in ARTIFACTS:
            df = df.with_columns(
                [
                    np.power(pl.col(fea), 4),
                ]
            )

        # train simple linear model
        df = df.sort(by=["session", "candidate_aid"], reverse=[True, False])
        df = df.to_pandas()
        # df = downsample(df)
        X = df[ARTIFACTS]
        group = df["session"]
        y = df[TARGET]

        skgfold = StratifiedGroupKFold(n_splits=2)
        train_idx, val_idx = [], []

        # init ensemble models
        stacking_model = StackingModels()
        train_pr_aucs = []
        train_roc_aucs = []
        val_pr_aucs = []
        val_roc_aucs = []
        j = 0
        logging.info(f"start train cross validation")
        feature_importances = []
        val_score_dists = []
        for tidx, vidx in tqdm(skgfold.split(X, y, groups=group)):
            train_idx, val_idx = tidx, vidx
            X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
            y_train, y_val = y[train_idx], y[val_idx]
            # group_train, group_val = group[train_idx], group[val_idx]

            logging.info(
                f"FOLD {j} y_train {np.mean(y_train)} | y_val {np.mean(y_val)}"
            )
            logging.info(f"FOLD {j} X_train {X_train.shape} | X_val {X_val.shape}")
            # init model for this fold
            if algo == "logreg":
                model = LogisticRegression(
                    penalty="l2", C=0.5, class_weight="balanced", random_state=1234
                )
            # elif algo == "mlp":
            #     model = MLPClassifier()
            else:
                model = LogisticRegression()

            model.fit(X=X_train, y=y_train)
            # score trainset
            y_proba = model.predict_proba(X=X_train)[:, 1]
            roc_auc = roc_auc_score(y_true=y_train, y_score=y_proba)
            pr_auc = average_precision_score(y_true=y_train, y_score=y_proba)
            train_roc_aucs.append(roc_auc)
            train_pr_aucs.append(pr_auc)
            # logging.info(
            #     f"TRAIN {EVENT.upper()}:{j} ROC AUC {roc_auc} | PR AUC {pr_auc}"
            # )
            # score valset
            y_proba = model.predict_proba(X=X_val)[:, 1]
            roc_auc = roc_auc_score(y_true=y_val, y_score=y_proba)
            pr_auc = average_precision_score(y_true=y_val, y_score=y_proba)
            val_roc_aucs.append(roc_auc)
            val_pr_aucs.append(pr_auc)
            # logging.info(f"VAL {EVENT.upper()}:{j} ROC AUC {roc_auc} | PR AUC {pr_auc}")

            # save feature importance
            # get df feature importance
            # feature importance as DataFrame
            df_fea_imp = pd.DataFrame(
                {"feature": X_train.columns.to_list(), "importance": model.coef_[0]}
            ).sort_values(by="importance", ascending=False, ignore_index=True)
            feature_importances.append(df_fea_imp)

            # save val score distribution
            val_score = {"y": y_val, "y_hat": y_proba}
            val_score = pd.DataFrame(val_score)
            val_score_dists.append(val_score)

            # append to stacking
            stacking_model.append(model=model)
            j += 1

        # print eval
        train_pr_aucs = np.mean(train_pr_aucs)
        train_roc_aucs = np.mean(train_roc_aucs)
        val_pr_aucs = np.mean(val_pr_aucs)
        val_roc_aucs = np.mean(val_roc_aucs)
        logging.info(
            f"TRAIN {EVENT.upper()} ROC AUC {train_roc_aucs} | PR AUC {train_pr_aucs}"
        )
        logging.info(
            f"VAL {EVENT.upper()} ROC AUC {val_roc_aucs} | PR AUC {val_pr_aucs}"
        )

        # save artifact
        # save artifacts to
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # get current date
        # naming unique model
        mean_val_roc_auc = int(val_roc_aucs * 100000)
        mean_val_pr_auc = int(val_pr_aucs * 100000)
        unique_model_name = (
            f"{current_date}_{EVENT}_{algo}_{mean_val_pr_auc}_{mean_val_roc_auc}"
        )
        artifact_path = get_artifacts_training_dir(event=f"{EVENT}_stacking", week="w2")
        # create artifact dir
        filepath = artifact_path / unique_model_name
        check_directory(filepath)
        logging.info(f"saving artifacts to: {filepath}")
        summary_feature_importance = summarise_feature_importance(feature_importances)
        plot_and_save_feature_importance(
            summary_feature_importance, filepath, metric="median_importance"
        )
        # plot score distribution per target variable
        plot_and_save_score_distribution(
            dfs=val_score_dists, filepath=filepath, dataset="val"
        )

        # pickle the model
        model_name = f"{filepath}/model.pkl"
        joblib.dump(stacking_model, model_name)

        # # collect submission in 1,40
        # for ix in range(1,40):
        #     pass
        scoring_stacking(
            artifact=unique_model_name,
            event=EVENT,
            week_data="w2",
            week_model="w2",
        )

        unique_model_names.append(unique_model_name)

    # after finish scoring all events, then make submission
    make_submission(
        click_model=unique_model_names[0],
        cart_model=unique_model_names[1],
        order_model=unique_model_names[2],
        week_data="w2",
        week_model="w2",
    )

    # evaluate submission in 1,40
    eval_submission(
        click_model=unique_model_names[0],
        cart_model=unique_model_names[1],
        order_model=unique_model_names[2],
        week_data="w2",
        week_model="w2",
    )


@click.command()
@click.option(
    "--algo",
    help="avaiable mode: logreg/mlp",
)
def main(algo: str):
    train_stacking(algo=algo)


if __name__ == "__main__":
    main()
