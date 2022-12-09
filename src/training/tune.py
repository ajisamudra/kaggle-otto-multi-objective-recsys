import click
import polars as pl
import pandas as pd
import numpy as np
import gc
import datetime
from typing import Union
import optuna
from optuna.samplers import TPESampler
from sklearn.utils import resample
from sklearn.metrics import average_precision_score
from src.utils.constants import (
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
    for IX in range(k):
        filepath = f"{input_path}/train_{IX}_{event}_combined.parquet"
        train_df = pl.read_parquet(filepath)
        filepath = f"{val_path}/test_{IX}_{event}_combined.parquet"
        val_df = pl.read_parquet(filepath)

        # sort data based on session & label
        train_df = train_df.sort(by=["session", TARGET], reverse=[True, True])
        val_df = val_df.sort(by=["session", TARGET], reverse=[True, True])

        train_df = train_df.to_pandas()
        val_df = val_df.to_pandas()

        selected_features = list(train_df.columns)
        selected_features.remove("session")
        selected_features.remove("candidate_aid")
        selected_features.remove(TARGET)

        # downsample training data so negative class 20:1 positive class
        desired_ratio = 20
        positive_class = train_df[train_df[TARGET] == 1]
        negative_class = train_df[train_df[TARGET] == 0]
        negative_downsample = resample(
            negative_class,
            replace=False,
            n_samples=len(positive_class) * desired_ratio,
            random_state=777,
        )

        train_df = pd.concat([positive_class, negative_downsample], ignore_index=True)
        train_df = train_df.sort_values(by=["session", TARGET], ascending=[True, True])
        logging.info(train_df.shape)

        del positive_class, negative_class, negative_downsample
        gc.collect()

        X_train = train_df[selected_features]
        group_train = train_df["session"]
        y_train = train_df[TARGET]

        X_val = val_df[selected_features]
        group_val = val_df["session"]
        y_val = val_df[TARGET]

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
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.008, 0.1),
            "max_depth": trial.suggest_int("max_depth", 1, 7),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.9),
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
        # check performance before tuning
        model: Union[ClassifierModel, RankingModel]
        if algo == "lgbm_classifier":
            model = LGBClassifier()
        elif algo == "lgbm_ranker":
            model = LGBRanker()
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
    default=2,
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
    k: int = 2,
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
