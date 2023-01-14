import click
import polars as pl
import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import joblib
import datetime
from typing import Union
from pathlib import Path
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from src.training.export_treelite import export_treelite_model

from src.utils.constants import (
    CFG,
    write_json,
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_training_test_dataset_dir,
    get_processed_scoring_train_dataset_dir,
)
from src.model.model import (
    EnsembleModels,
    CatClassifier,
    CATRanker,
    ClassifierModel,
    LGBClassifier,
    LGBRanker,
    RankingModel,
)
from src.metrics.model_evaluation import (
    summarise_feature_importance,
    plot_and_save_feature_importance,
    plot_and_save_score_distribution,
)

# for enabling training + scoring
from src.scoring.score import scoring
from src.scoring.score_treelite import scoring_treelite
from src.scoring.make_submission import make_submission
from src.scoring.eval_submission import eval_submission

from src.utils.constants import get_artifacts_training_dir, check_directory

from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def downsample(df: pd.DataFrame):
    desired_ratio = 10
    positive_class = df[df[TARGET] == 1]
    negative_class = df[df[TARGET] == 0]
    negative_downsample = resample(
        negative_class,
        replace=False,
        n_samples=len(positive_class) * desired_ratio,
        random_state=777,
    )

    df = pd.concat([positive_class, negative_downsample], ignore_index=True)
    df = df.sort_values(by=["session", TARGET], ascending=[True, True])

    return df


def train(algo: str, events: list, week: str, n: int, eval: int):
    # for each event
    # initiate ensemble model
    # iterate N
    # for each N: read training data, train_test_split, fit model
    # append to ensemble_model, save feature importance, measure ROC/PR AUC per chunk
    # save ensemble model for particular event,
    # score validation on particular event
    # measure recall@20 in that event
    # continue on next event
    # if eval == 1 -> perform scoring in validation & eval submission

    input_path: Path
    val_path: Path
    if week == "w1":
        input_path = get_processed_scoring_train_dataset_dir()
        if eval == 1:
            raise ValueError("Can't eval submission for training with data w1")
    elif week == "w2":
        input_path = get_processed_training_train_dataset_dir()
        val_path = get_processed_training_test_dataset_dir()
    else:
        raise NotImplementedError("week not implemented! (w1/w2)")

    event_train_roc_aucs, event_val_roc_aucs = [], []
    event_train_pr_aucs, event_val_pr_aucs = [], []
    unique_model_names = []
    for EVENT in events:
        performance, hyperparams = {}, {}
        logging.info(f"init ensemble for event {EVENT.upper()}")
        ensemble_model = EnsembleModels()
        train_roc_aucs, train_pr_aucs = [], []
        val_roc_aucs, val_pr_aucs = [], []
        feature_importances = []
        train_score_dists = []
        val_score_dists = []
        for IX in tqdm(range(n)):
            model: Union[ClassifierModel, RankingModel]
            # hyperparams click

            hyperparams = {}
            if algo == "lgbm_classifier":
                if EVENT == "orders":
                    # LB 0.564 fea 99
                    # hyperparams = {
                    #     "n_estimators": 500,
                    #     "learning_rate": 0.05479523964757477,
                    #     "max_depth": 7,
                    #     "num_leaves": 40,
                    #     "min_data_in_leaf": 560,
                    #     "feature_fraction": 0.849801842682232,
                    #     "bagging_fraction": 0.8599407686777631,
                    # }
                    hyperparams = {
                        "n_estimators": 1100,
                        "learning_rate": 0.052824552063657305,
                        "max_depth": 7,
                        "num_leaves": 98,
                        "min_data_in_leaf": 536,
                        "feature_fraction": 0.9373038392898101,
                        "bagging_fraction": 0.926452587658148,
                    }
                elif EVENT == "carts":
                    # LB 0.564 fea 99
                    # hyperparams = {
                    #     "n_estimators": 500,
                    #     "learning_rate": 0.023284940416743508,
                    #     "max_depth": 2,
                    #     "num_leaves": 39,
                    #     "min_data_in_leaf": 435,
                    #     "feature_fraction": 0.7118135289194919,
                    #     "bagging_fraction": 0.8703404040978261,
                    # }
                    hyperparams = {
                        "n_estimators": 1100,
                        "learning_rate": 0.034626607160951436,
                        "max_depth": 6,
                        "num_leaves": 34,
                        "min_data_in_leaf": 420,
                        "feature_fraction": 0.948462644663103,
                        "bagging_fraction": 0.7636737045356861,
                    }
                elif EVENT == "clicks":
                    # LB 0.564 fea 99
                    # hyperparams = {
                    #     "n_estimators": 500,
                    #     "learning_rate": 0.03165988766745295,
                    #     "max_depth": 5,
                    #     "num_leaves": 102,
                    #     "min_data_in_leaf": 416,
                    #     "feature_fraction": 0.7231413635494773,
                    #     "bagging_fraction": 0.7577740460272675,
                    # }
                    hyperparams = {
                        "n_estimators": 1100,
                        "learning_rate": 0.04045305955075708,
                        "max_depth": 6,
                        "num_leaves": 49,
                        "min_data_in_leaf": 1138,
                        "feature_fraction": 0.8756392153185409,
                        "bagging_fraction": 0.8882829042882201,
                    }
            elif algo == "lgbm_ranker":
                if EVENT == "orders":
                    hyperparams = {
                        "n_estimators": 2000,
                        "learning_rate": 0.06686942900752924,
                        "max_depth": 2,
                        "num_leaves": 19,
                        "min_data_in_leaf": 1258,
                        "feature_fraction": 0.8126707937968762,
                        "bagging_fraction": 0.9120592736711024,
                    }
                elif EVENT == "carts":
                    hyperparams = {
                        "n_estimators": 2000,
                        "learning_rate": 0.0489842694434591,
                        "max_depth": 5,
                        "num_leaves": 83,
                        "min_data_in_leaf": 1358,
                        "feature_fraction": 0.7825532461598741,
                        "bagging_fraction": 0.7240384385328306,
                    }
                elif EVENT == "clicks":
                    hyperparams = {
                        "n_estimators": 2000,
                        "learning_rate": 0.04184392968947091,
                        "max_depth": 6,
                        "num_leaves": 106,
                        "min_data_in_leaf": 780,
                        "feature_fraction": 0.932581545103427,
                        "bagging_fraction": 0.8143391317097348,
                    }

            if algo == "lgbm_classifier":
                model = LGBClassifier(**hyperparams)
            elif algo == "cat_classifier":
                model = CatClassifier(**hyperparams)
            elif algo == "lgbm_ranker":
                model = LGBRanker(**hyperparams)
            elif algo == "cat_ranker":
                model = CATRanker(**hyperparams)
            else:
                raise NotImplementedError("algorithm not implemented! (lgbm/catboost)")

            logging.info(f"read training & validation data for chunk: {IX}")

            train_df = pl.DataFrame()
            val_df = pl.DataFrame()
            if EVENT == "orders":
                for i in range(CFG.N_train):
                    filepath = f"{input_path}/train_{i}_{EVENT}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    # df_chunk = df_chunk.to_pandas()
                    # df_chunk = downsample(df_chunk)
                    # df_chunk = pl.from_pandas(df_chunk)
                    train_df = pl.concat([train_df, df_chunk])

                for i in range(5):
                    filepath = f"{val_path}/test_{i}_{EVENT}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    df_chunk = df_chunk.to_pandas()
                    df_chunk = downsample(df_chunk)
                    df_chunk = pl.from_pandas(df_chunk)
                    val_df = pl.concat([val_df, df_chunk])

            elif EVENT == "carts":
                train_df = pl.DataFrame()
                for i in range(CFG.N_train):
                    filepath = f"{input_path}/train_{i}_{EVENT}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    # df_chunk = df_chunk.to_pandas()
                    # df_chunk = downsample(df_chunk)
                    # df_chunk = pl.from_pandas(df_chunk)
                    train_df = pl.concat([train_df, df_chunk])

                for i in range(3):
                    filepath = f"{val_path}/test_{i}_{EVENT}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    df_chunk = df_chunk.to_pandas()
                    df_chunk = downsample(df_chunk)
                    df_chunk = pl.from_pandas(df_chunk)
                    val_df = pl.concat([val_df, df_chunk])

            else:
                for i in range(int(CFG.N_train / 5)):
                    filepath = f"{input_path}/train_{i}_{EVENT}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    # df_chunk = df_chunk.to_pandas()
                    # df_chunk = downsample(df_chunk)
                    # df_chunk = pl.from_pandas(df_chunk)
                    train_df = pl.concat([train_df, df_chunk])

                for i in range(1):
                    filepath = f"{val_path}/test_{i}_{EVENT}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    df_chunk = df_chunk.to_pandas()
                    df_chunk = downsample(df_chunk)
                    df_chunk = pl.from_pandas(df_chunk)
                    val_df = pl.concat([val_df, df_chunk])

            logging.info(f"train shape {train_df.shape}")
            logging.info(f"val shape {val_df.shape}")
            # sort data based on session & label
            train_df = train_df.sort(
                by=["session", "candidate_aid"], reverse=[True, False]
            )
            train_df = train_df.to_pandas()

            val_df = val_df.sort(by=["session", "candidate_aid"], reverse=[True, False])
            val_df = val_df.to_pandas()

            selected_features = list(train_df.columns)
            selected_features.remove("session")
            selected_features.remove("candidate_aid")
            selected_features.remove(TARGET)

            # unimportant features in order models
            selected_features.remove("rank_word2vec")
            selected_features.remove("rank_word2vec_dur")
            selected_features.remove("rank_word2vec_wgtd_dur")
            selected_features.remove("rank_word2vec_wgtd_rec")
            selected_features.remove("rank_popular_week")
            selected_features.remove("rank_matrix_fact")
            selected_features.remove("rank_fasttext")
            selected_features.remove("retrieval_fasttext")
            selected_features.remove("retrieval_popular_week")
            selected_features.remove("retrieval_word2vec")
            selected_features.remove("retrieval_word2vec_dur")
            selected_features.remove("retrieval_word2vec_wgtd_dur")
            selected_features.remove("retrieval_word2vec_wgtd_rec")
            selected_features.remove("retrieval_matrix_fact")

            # remove word2vec related fea
            selected_features.remove(
                "diff_w_mean_word2vec_skipgram_last_event_cosine_distance"
            )

            X_train = train_df[selected_features]
            group_train = train_df["session"]
            y_train = train_df[TARGET]

            X_val = val_df[selected_features]
            group_val = val_df["session"]
            y_val = val_df[TARGET]

            del val_df
            gc.collect()

            # split train and validation using StratifiedGroupKFold
            # X_train, X_val, y_train, y_val, group_train, group_val = train_test_split(
            #     X, y, group, test_size=0.2, stratify=y, random_state=745
            # )

            # X = train_df[selected_features]
            # group = train_df["session"]
            # y = train_df[TARGET]

            del train_df
            gc.collect()

            # skgfold = StratifiedGroupKFold(n_splits=5)
            # train_idx, val_idx = [], []

            # for tidx, vidx in skgfold.split(X, y, groups=group):
            #     train_idx, val_idx = tidx, vidx

            # X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
            # y_train, y_val = y[train_idx], y[val_idx]
            # group_train, group_val = group[train_idx], group[val_idx]

            # del X, y, group
            # gc.collect()

            # calculate num samples per group
            logging.info("calculate num samples per group")
            n_group_train = list(group_train.value_counts())
            n_group_val = list(group_val.value_counts())

            logging.info("distribution of n_candidate in train")
            logging.info(group_train.value_counts().value_counts())
            logging.info("distribution of n_candidate in val")
            logging.info(group_val.value_counts().value_counts())

            logging.info(f"train shape {X_train.shape} sum groups {sum(n_group_train)}")
            logging.info(f"val shape {X_val.shape} sum groups {sum(n_group_val)}")
            logging.info(f"y_train {np.mean(y_train)} | y_val {np.mean(y_val)}")

            if algo == "lgbm_classifier":
                model.fit(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
            elif algo == "cat_classifier":
                model.fit(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
            elif algo == "lgbm_ranker":
                model.fit(
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                    group_train=n_group_train,
                    group_val=n_group_val,
                    eval_at=20,
                )
            elif algo == "cat_ranker":
                model.fit(
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                    group_train=group_train,
                    group_val=group_val,
                )

            hyperparams = model.get_params()
            ensemble_model.append(model)
            # predict train
            y_proba = model.predict(X_train)
            roc_auc = roc_auc_score(y_true=y_train, y_score=y_proba)
            pr_auc = average_precision_score(y_true=y_train, y_score=y_proba)
            train_roc_aucs.append(roc_auc)
            train_pr_aucs.append(pr_auc)
            logging.info(
                f"TRAIN {EVENT.upper()}:{IX} ROC AUC {roc_auc} | PR AUC {pr_auc}"
            )
            train_score = {"y": y_train, "y_hat": y_proba}
            train_score = pd.DataFrame(train_score)
            train_score_dists.append(train_score)
            # predict val
            y_proba = model.predict(X_val)
            roc_auc = roc_auc_score(y_true=y_val, y_score=y_proba)
            pr_auc = average_precision_score(y_true=y_val, y_score=y_proba)
            val_roc_aucs.append(roc_auc)
            val_pr_aucs.append(pr_auc)
            logging.info(
                f"VAL {EVENT.upper()}:{IX} ROC AUC {roc_auc} | PR AUC {pr_auc}"
            )
            # save feature importance
            feature_importances.append(model.feature_importances_)
            val_score = {"y": y_val, "y_hat": y_proba}
            val_score = pd.DataFrame(val_score)
            val_score_dists.append(val_score)

            del X_train, X_val, y_train, y_val, group_train, group_val
            gc.collect()

        # save to dict per event
        performance["train_roc_auc"] = train_roc_aucs
        performance["train_pr_auc"] = train_pr_aucs
        performance["val_roc_auc"] = val_roc_aucs
        performance["val_pr_auc"] = val_pr_aucs
        mean_val_roc_auc = np.mean(val_roc_aucs)
        mean_val_pr_auc = np.mean(val_pr_aucs)
        mean_train_roc_auc = np.mean(train_roc_aucs)
        mean_train_pr_auc = np.mean(train_pr_aucs)

        event_train_roc_aucs.append(mean_train_roc_auc)
        event_train_pr_aucs.append(mean_train_pr_auc)
        event_val_roc_aucs.append(mean_val_roc_auc)
        event_val_pr_aucs.append(mean_val_pr_auc)

        logging.info(
            f"TRAIN {EVENT.upper()}:MEAN ROC AUC {mean_train_roc_auc} | PR AUC {mean_train_pr_auc}"
        )
        logging.info(
            f"VAL {EVENT.upper()}:MEAN ROC AUC {mean_val_roc_auc} | PR AUC {mean_val_pr_auc}"
        )

        # save artifacts to
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # get current date
        # naming unique model
        mean_val_roc_auc = int(mean_val_roc_auc * 100000)
        mean_val_pr_auc = int(mean_val_pr_auc * 100000)
        unique_model_name = (
            f"{current_date}_{EVENT}_{algo}_{mean_val_pr_auc}_{mean_val_roc_auc}"
        )
        artifact_path = get_artifacts_training_dir(event=EVENT, week=week)
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
            dfs=train_score_dists, filepath=filepath, dataset="train"
        )
        plot_and_save_score_distribution(
            dfs=val_score_dists, filepath=filepath, dataset="val"
        )

        # pickle the model
        model_name = f"{filepath}/model.pkl"
        joblib.dump(ensemble_model, model_name)

        # output hyperparams model
        hyper_path = f"{filepath}/hyperparams.json"
        write_json(filepath=hyper_path, data=hyperparams)

        # output performance metrics
        perf_path = f"{filepath}/performance_metric.json"
        write_json(filepath=perf_path, data=performance)

        if eval == 1:
            # perform scoring
            if algo in ["lgbm_classifier", "lgbm_ranker"]:
                # export treelite model
                export_treelite_model(
                    artifact=unique_model_name, event=EVENT, week_model="w2"
                )
                # scoring using treelite model
                scoring_treelite(
                    artifact=unique_model_name,
                    event=EVENT,
                    week_data="w2",
                    week_model="w2",
                )
            else:
                scoring(
                    artifact=unique_model_name,
                    event=EVENT,
                    week_data="w2",
                    week_model="w2",
                )
            # append unique_model_names for make & eval submission
            unique_model_names.append(unique_model_name)

    if eval == 1:
        # make submission
        # append unique_model_names for make & eval submission
        make_submission(
            click_model=unique_model_names[2],
            cart_model=unique_model_names[1],
            order_model=unique_model_names[0],
            week_data="w2",
            week_model="w2",
        )

    logging.info("complete training models!")
    logging.info("=========== SUMMARY ===========")
    for ix, EVENT in enumerate(events):
        logging.info(f"=========== {EVENT.upper()} ===========")
        logging.info(
            f"TRAIN MEAN ROC AUC {event_train_roc_aucs[ix]} | PR AUC {event_train_pr_aucs[ix]}"
        )
        logging.info(
            f"VAL MEAN ROC AUC {event_val_roc_aucs[ix]} | PR AUC {event_val_pr_aucs[ix]}"
        )
    logging.info("============= END =============")

    if eval == 1:
        # eval submission
        logging.info("start eval submission")
        eval_submission(
            click_model=unique_model_names[2],
            cart_model=unique_model_names[1],
            order_model=unique_model_names[0],
            week_data="w2",
            week_model="w2",
        )


@click.command()
@click.option(
    "--event",
    default="all",
    help="avaiable event: clicks/carts/orders/all",
)
@click.option(
    "--algo",
    default="lgbm",
    help="algorithm for training; lgbm_classifier/lgbm_ranker/cat_classifier/cat_ranker",
)
@click.option(
    "--week",
    default="w2",
    help="subset of training data, w1/w2; w1:scoring dir, w2:training dir",
)
@click.option(
    "--n",
    default=1,
    help="number of chunk for training; between 1-10",
)
@click.option(
    "--eval",
    default=1,
    help="number of chunk for training; between 1-10",
)
def main(
    event: str = "all", algo: str = "lgbm", week: str = "w2", n: int = 1, eval: int = 1
):
    events = ["orders", "carts", "clicks"]
    if event != "all":
        events = [event]
        if eval == 1:
            raise ValueError("Can't eval if not all events trained on")
    train(algo=algo, events=events, week=week, n=n, eval=eval)


if __name__ == "__main__":
    main()
