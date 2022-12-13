import click
import polars as pl
import pandas as pd
import numpy as np
import gc
import datetime
from typing import Union
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from src.training.train import downsample
from src.utils.constants import (
    CFG,
    write_json,
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_training_test_dataset_dir,
)
from src.model.model import (
    CatClassifier,
    CATRanker,
    ClassifierModel,
    LGBClassifier,
    LGBRanker,
    RankingModel,
)


from src.utils.constants import (
    get_artifacts_tuning_dir,
    check_directory,
)

from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def cross_validation_ap_score(
    model: Union[ClassifierModel, RankingModel],
    algo: str,
    event: str,
    k: int,
) -> float:

    input_path = get_processed_training_train_dataset_dir()
    val_path = get_processed_training_test_dataset_dir()

    val_pr_aucs = []
    for _ in range(k):
        train_df = pl.DataFrame()
        val_df = pl.DataFrame()
        if event == "orders":
            for i in range(CFG.N_train):
                filepath = f"{input_path}/train_{i}_{event}_combined.parquet"
                df_chunk = pl.read_parquet(filepath)
                df_chunk = df_chunk.to_pandas()
                df_chunk = downsample(df_chunk)
                df_chunk = pl.from_pandas(df_chunk)
                train_df = pl.concat([train_df, df_chunk])

            for i in range(10):
                filepath = f"{val_path}/test_{i}_{event}_combined.parquet"
                df_chunk = pl.read_parquet(filepath)
                df_chunk = df_chunk.to_pandas()
                df_chunk = downsample(df_chunk)
                df_chunk = pl.from_pandas(df_chunk)
                val_df = pl.concat([val_df, df_chunk])

        elif event == "carts":
            train_df = pl.DataFrame()
            for i in range(CFG.N_train):
                filepath = f"{input_path}/train_{i}_{event}_combined.parquet"
                df_chunk = pl.read_parquet(filepath)
                df_chunk = df_chunk.to_pandas()
                df_chunk = downsample(df_chunk)
                df_chunk = pl.from_pandas(df_chunk)
                train_df = pl.concat([train_df, df_chunk])

            for i in range(10):
                filepath = f"{val_path}/test_{i}_{event}_combined.parquet"
                df_chunk = pl.read_parquet(filepath)
                df_chunk = df_chunk.to_pandas()
                df_chunk = downsample(df_chunk)
                df_chunk = pl.from_pandas(df_chunk)
                val_df = pl.concat([val_df, df_chunk])
        else:
            for i in range(int(CFG.N_train / 10) + 1):
                filepath = f"{input_path}/train_{i}_{event}_combined.parquet"
                df_chunk = pl.read_parquet(filepath)
                df_chunk = df_chunk.to_pandas()
                df_chunk = downsample(df_chunk)
                df_chunk = pl.from_pandas(df_chunk)
                train_df = pl.concat([train_df, df_chunk])

            for i in range(5):
                filepath = f"{val_path}/test_{i}_{event}_combined.parquet"
                df_chunk = pl.read_parquet(filepath)
                df_chunk = df_chunk.to_pandas()
                df_chunk = downsample(df_chunk)
                df_chunk = pl.from_pandas(df_chunk)
                val_df = pl.concat([val_df, df_chunk])

        # sort data based on session & label
        train_df = train_df.sort(by=["session", TARGET], reverse=[True, True])
        val_df = val_df.sort(by=["session", TARGET], reverse=[True, True])

        train_df = train_df.to_pandas()
        val_df = val_df.to_pandas()

        selected_features = list(train_df.columns)
        selected_features.remove("session")
        selected_features.remove("candidate_aid")
        selected_features.remove(TARGET)

        # X_train = train_df[selected_features]
        # group_train = train_df["session"]
        # y_train = train_df[TARGET]

        # X_val = val_df[selected_features]
        # group_val = val_df["session"]
        # y_val = val_df[TARGET]

        X = train_df[selected_features]
        group = train_df["session"]
        y = train_df[TARGET]

        skgfold = StratifiedGroupKFold(n_splits=5)
        train_idx, val_idx = [], []

        for tidx, vidx in skgfold.split(X, y, groups=group):
            train_idx, val_idx = tidx, vidx

        X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
        y_train, y_val = y[train_idx], y[val_idx]
        group_train, group_val = group[train_idx], group[val_idx]

        # calculate num samples per group
        n_group_train = list(group_train.value_counts())
        n_group_val = list(group_val.value_counts())
        logging.info(f"train shape {X_train.shape} sum groups {sum(n_group_train)}")
        logging.info(f"val shape {X_val.shape} sum groups {sum(n_group_val)}")
        logging.info(f"y_train {np.mean(y_train)} | y_val {np.mean(y_val)}")

        if algo == "lgbm_classifier":
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

        # predict val
        y_proba = model.predict(X_val)
        pr_auc = average_precision_score(y_true=y_val, y_score=y_proba)
        val_pr_aucs.append(pr_auc)

        del train_df, val_df, X_train, X_val, y_train, y_val, group_train, group_val
        gc.collect()

    return float(np.mean(val_pr_aucs))


class ObjectiveLGBModel:
    def __init__(self, algo: str, n_estimators: int, k: int, event: str):
        self.n_estimators = n_estimators
        self.algo = algo
        self.event = event
        self.k = k

    def __call__(self, trial):
        hyperparams = {
            "n_estimators": trial.suggest_int(
                "n_estimators", self.n_estimators, self.n_estimators
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 400, 1500),
            # "feature_fraction": trial.suggest_float("feature_fraction", 0.9, 1),
            # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.9, 1),
            "random_state": 747,
            "verbose": 0,
        }
        model: Union[ClassifierModel, RankingModel]
        if self.algo == "lgbm_classifier":
            model = LGBClassifier(**hyperparams)
        elif self.algo == "lgbm_ranker":
            model = LGBRanker(**hyperparams)
        # measure performance in CV
        cv_ap = cross_validation_ap_score(
            model, k=self.k, event=self.event, algo=self.algo
        )

        return cv_ap


def perform_tuning(algo: str, events: list, k: int, n_estimators: int, n_trial: int):
    for EVENT in events:
        logging.info("measure AP before tuning")

        hyperparams = {}
        if algo == "lgbm_classifier":
            if EVENT == "orders":
                hyperparams = {
                    "n_estimators": 500,
                    "learning_rate": 0.052824552063657305,
                    "max_depth": 7,
                    "num_leaves": 98,
                    "min_data_in_leaf": 536,
                    "feature_fraction": 0.9373038392898101,
                    "bagging_fraction": 0.926452587658148,
                }
            elif EVENT == "carts":
                hyperparams = hyperparams = {
                    "n_estimators": 500,
                    "learning_rate": 0.034626607160951436,
                    "max_depth": 6,
                    "num_leaves": 34,
                    "min_data_in_leaf": 420,
                    "feature_fraction": 0.948462644663103,
                    "bagging_fraction": 0.7636737045356861,
                }
            elif EVENT == "clicks":
                hyperparams = {
                    "n_estimators": 500,
                    "learning_rate": 0.04045305955075708,
                    "max_depth": 6,
                    "num_leaves": 49,
                    "min_data_in_leaf": 1138,
                    "feature_fraction": 0.8756392153185409,
                    "bagging_fraction": 0.8882829042882201,
                }
        elif algo == "lgbm_ranker":
            if EVENT == "orders":
                hyperparams = {}
            elif EVENT == "carts":
                hyperparams = {}
            elif EVENT == "clicks":
                hyperparams = {}

        # check performance before tuning
        model: Union[ClassifierModel, RankingModel]
        if algo == "lgbm_classifier":
            model = LGBClassifier(**hyperparams)
        elif algo == "lgbm_ranker":
            model = LGBRanker(**hyperparams)
        mean_val_pr_auc_before = cross_validation_ap_score(
            model=model, algo=algo, event=EVENT, k=k
        )
        logging.info(f"VAL AP with {k} fold before tuning: {mean_val_pr_auc_before}")

        logging.info("perform tuning")
        study = optuna.create_study(direction="maximize", sampler=TPESampler())
        if algo in ["lgbm_classifier", "lgbm_ranker"]:
            study.optimize(
                ObjectiveLGBModel(
                    algo=algo, n_estimators=n_estimators, k=k, event=EVENT
                ),
                n_trials=n_trial,
            )

        logging.info("complete tuning!")
        # find best hyperparams
        mean_val_pr_auc_after = study.best_value
        best_hyperparams = study.best_params
        logging.info(f"Previous value: {mean_val_pr_auc_before}")
        logging.info(f"Found best tuned value {mean_val_pr_auc_after}")
        logging.info(f"Best hyperparams {best_hyperparams}")
        # if better, save to artifact dir
        performance_metric = {
            "before": mean_val_pr_auc_before,
            "after": mean_val_pr_auc_after,
        }
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # get current date
        unique_model_name = f"{current_date}_{EVENT}_{algo}_{mean_val_pr_auc_before}_{mean_val_pr_auc_after}"
        artifact_path = get_artifacts_tuning_dir(event=EVENT, week="w2")
        # create artifact dir
        filepath = artifact_path / unique_model_name
        check_directory(filepath)
        logging.info(f"saving artifacts to: {filepath}")
        hyper_path = f"{filepath}/tuned_hyperparams.json"
        write_json(filepath=hyper_path, data=best_hyperparams)

        # output performance metrics
        perf_path = f"{filepath}/performance_metric.json"
        write_json(filepath=perf_path, data=performance_metric)

        if mean_val_pr_auc_after > mean_val_pr_auc_before:
            logging.info(
                "Tuned value is higher than previous value! you should update hyperparams in training!"
            )
            logging.info(performance_metric)


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
    "--k",
    default=1,
    help="number of K in cross-validation; between 1-10",
)
@click.option(
    "--n_estimators",
    default=500,
    help="number of K in cross-validation; between 1-10",
)
@click.option(
    "--n_trial",
    default=50,
    help="number of K in cross-validation; between 1-10",
)
def main(
    event: str = "all",
    algo: str = "lgbm",
    k: int = 1,
    n_estimators: int = 500,
    n_trial: int = 50,
):
    events = ["orders", "carts", "clicks"]
    if event != "all":
        events = [event]
    perform_tuning(
        algo=algo, events=events, k=k, n_estimators=n_estimators, n_trial=n_trial
    )


if __name__ == "__main__":
    main()
