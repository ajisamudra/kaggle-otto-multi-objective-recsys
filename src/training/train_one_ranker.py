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
from src.scoring.score_one_ranker import scoring
from src.scoring.make_submission import make_submission
from src.scoring.eval_submission import eval_submission

from src.utils.constants import get_artifacts_training_dir, check_directory

from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def downsample(df: pd.DataFrame):
    desired_ratio = 20
    positive_class = df[df[TARGET] >= 1]
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

    unique_model_names = []
    performance, hyperparams = {}, {}
    logging.info("init ensemble one ranker")
    ensemble_model = EnsembleModels()
    feature_importances = []
    model: Union[ClassifierModel, RankingModel]
    hyperparams = {}
    if algo == "lgbm_ranker":
        hyperparams = {
            "n_estimators": 1000,
            # "learning_rate": 0.03683453215722998,
            # "max_depth": 7,
            # "num_leaves": 78,
            # "min_data_in_leaf": 1154,
            # "feature_fraction": 0.7977584122891811,
            # "bagging_fraction": 0.712445779238389,
        }
    elif algo == "cat_ranker":
        hyperparams = {"n_estimators": 1000}

    if algo == "lgbm_ranker":
        model = LGBRanker(**hyperparams)
    elif algo == "cat_ranker":
        model = CATRanker(**hyperparams)
    else:
        raise NotImplementedError("algorithm not implemented! (lgbm_ranker/cat_ranker)")

    logging.info(f"read training data for one ranker")
    train_df = pl.DataFrame()

    for i in range(2):
        filepath = f"{input_path}/train_{i}_one_ranker_combined.parquet"
        df_chunk = pl.read_parquet(filepath)
        # df_chunk = df_chunk.to_pandas()
        # df_chunk = downsample(df_chunk)
        # df_chunk = pl.from_pandas(df_chunk)
        train_df = pl.concat([train_df, df_chunk])

    logging.info(f"train shape {train_df.shape}")
    # logging.info(f"val shape {val_df.shape}")
    # sort data based on session & label
    train_df = train_df.sort(by=["session", TARGET], reverse=[True, True])
    train_df = train_df.to_pandas()
    # replace inf with 0
    # and make sure there's no None
    train_df = train_df.replace([np.inf, -np.inf], 0)
    train_df = train_df.fillna(0)

    selected_features = list(train_df.columns)
    selected_features.remove("session")
    selected_features.remove("candidate_aid")
    selected_features.remove(TARGET)

    X = train_df[selected_features]
    group = train_df["session"]
    y = train_df[TARGET]

    del train_df
    gc.collect()

    skgfold = StratifiedGroupKFold(n_splits=5)
    train_idx, val_idx = [], []

    for tidx, vidx in skgfold.split(X, y, groups=group):
        train_idx, val_idx = tidx, vidx

    X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
    y_train, y_val = y[train_idx], y[val_idx]
    group_train, group_val = group[train_idx], group[val_idx]

    del X, y, group
    gc.collect()

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

    if algo == "lgbm_ranker":
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

    del X_train, X_val, y_train, y_val, group_train, group_val
    gc.collect()

    event_val_roc_aucs = []
    event_val_pr_aucs = []
    val_score_dists = []
    for EVENT in events:
        logging.info(f"read validation data for chunk for event {EVENT.upper()}")
        test_df = pl.DataFrame()
        if EVENT == "orders":
            for i in range(10):
                filepath = f"{val_path}/test_{i}_{EVENT}_combined.parquet"
                df_chunk = pl.read_parquet(filepath)
                df_chunk = df_chunk.to_pandas()
                df_chunk = downsample(df_chunk)
                df_chunk = pl.from_pandas(df_chunk)
                test_df = pl.concat([test_df, df_chunk])

        elif EVENT == "carts":
            for i in range(10):
                filepath = f"{val_path}/test_{i}_{EVENT}_combined.parquet"
                df_chunk = pl.read_parquet(filepath)
                df_chunk = df_chunk.to_pandas()
                df_chunk = downsample(df_chunk)
                df_chunk = pl.from_pandas(df_chunk)
                test_df = pl.concat([test_df, df_chunk])

        else:
            for i in range(3):
                filepath = f"{val_path}/test_{i}_{EVENT}_combined.parquet"
                df_chunk = pl.read_parquet(filepath)
                df_chunk = df_chunk.to_pandas()
                df_chunk = downsample(df_chunk)
                df_chunk = pl.from_pandas(df_chunk)
                test_df = pl.concat([test_df, df_chunk])

        test_df = test_df.sort(by=["session", TARGET], reverse=[True, True])
        test_df = test_df.to_pandas()
        # replace inf with 0
        # and make sure there's no None
        test_df = test_df.replace([np.inf, -np.inf], 0)
        test_df = test_df.fillna(0)

        X_test = test_df[selected_features]
        y_test = test_df[TARGET]

        # predict val
        y_proba = model.predict(X_test)
        roc_auc = roc_auc_score(y_true=y_test, y_score=y_proba)
        pr_auc = average_precision_score(y_true=y_test, y_score=y_proba)
        event_val_roc_aucs.append(roc_auc)
        event_val_pr_aucs.append(pr_auc)
        logging.info(f"VAL {EVENT.upper()} ROC AUC {roc_auc} | PR AUC {pr_auc}")

        del test_df, X_test
        gc.collect()
        val_score = {"y": y_test, "y_hat": y_proba}
        val_score = pd.DataFrame(val_score)
        val_score_dists.append(val_score)

        # save to dict per event
        performance[f"{EVENT}_val_roc_auc"] = roc_auc
        performance[f"{EVENT}_val_pr_auc"] = pr_auc

    # save feature importance
    feature_importances.append(model.feature_importances_)

    # save artifacts to
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # get current date
    # naming unique model
    order_pr_auc = int(performance["orders_val_pr_auc"] * 100000)
    cart_pr_auc = int(performance["carts_val_pr_auc"] * 100000)
    click_pr_auc = int(performance["clicks_val_pr_auc"] * 100000)
    unique_model_name = (
        f"{current_date}_one_ranker_{algo}_{order_pr_auc}_{cart_pr_auc}_{click_pr_auc}"
    )
    artifact_path = get_artifacts_training_dir(event="one_ranker", week=week)
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
    joblib.dump(ensemble_model, model_name)

    # output hyperparams model
    hyper_path = f"{filepath}/hyperparams.json"
    write_json(filepath=hyper_path, data=hyperparams)

    # output performance metrics
    perf_path = f"{filepath}/performance_metric.json"
    write_json(filepath=perf_path, data=performance)

    if eval == 1:
        for EVENT in events:
            # perform scoring
            scoring(
                artifact=unique_model_name, event=EVENT, week_data="w2", week_model="w2"
            )

        # make submission
        make_submission(
            click_model=unique_model_name,
            cart_model=unique_model_name,
            order_model=unique_model_name,
            week_data="w2",
            week_model="w2",
        )

    logging.info("complete training models!")
    logging.info("=========== SUMMARY ===========")
    for ix, EVENT in enumerate(events):
        logging.info(f"=========== {EVENT.upper()} ===========")
        logging.info(
            f"VAL MEAN ROC AUC {event_val_roc_aucs[ix]} | PR AUC {event_val_pr_aucs[ix]}"
        )
    logging.info("============= END =============")

    if eval == 1:
        # eval submission
        logging.info("start eval submission")
        eval_submission(
            click_model=unique_model_name,
            cart_model=unique_model_name,
            order_model=unique_model_name,
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
    default=0,
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
