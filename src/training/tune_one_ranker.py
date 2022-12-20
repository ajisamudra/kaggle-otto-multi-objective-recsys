import click
import polars as pl
import pandas as pd
import numpy as np
import gc
import datetime
from typing import Union
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import ndcg_score
from src.training.train_one_ranker import downsample
from src.utils.constants import (
    CFG,
    write_json,
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_training_test_dataset_dir,
    get_processed_local_validation_dir,
)
from src.model.model import (
    CATRanker,
    ClassifierModel,
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


def measure_recall(df_pred: pd.DataFrame, df_truth: pd.DataFrame, K: int = 20):
    """
    df_pred is in submission format
    session | type | labels
    123 | clicks | [AID1 AID2 AID3]
    123 | carts | [AID1 AID2]
    123 | orders | [AID1]

    df_truth is in following format
    session | type | ground_truth
    123 | clicks | [AID1 AID4]
    123 | carts | [AID1]
    123 | orders | [AID1]

    Ks list of recall@K want to measure

    """
    # competition eval metrics
    score = 0
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    for t in ["clicks", "carts", "orders"]:
        # logging.info(f"filter submission event {t}")
        sub_session = (
            df_pred.loc[df_pred["type"] == t]["session"].apply(lambda x: int(x)).values
        )
        sub_labels = df_pred.loc[df_pred["type"] == t]["labels"].values
        # logging.info(f"filter label event {t}")
        label_session = df_truth.loc[df_truth["type"] == t]["session"].values
        label_truth = df_truth.loc[df_truth["type"] == t]["ground_truth"].values

        sub_dict = dict(zip(sub_session, sub_labels))
        tmp_labels_dict = dict(zip(label_session, label_truth))

        recall_scores, hit_scores, gt_counts = [], [], []
        for session_id in list(sub_session):
            targets = set(tmp_labels_dict[session_id])
            preds = set(sub_dict[session_id])

            # calc
            hit = len((targets & preds))
            len_truth = min(len(targets), 20)
            recall_score = hit / len_truth

            recall_scores.append(recall_score)
            hit_scores.append(hit)
            gt_counts.append(len_truth)

        recall_scores = np.array(recall_scores)
        hit_scores = np.array(hit_scores)
        gt_counts = np.array(gt_counts)

        n_hits = np.sum(hit_scores)
        n_gt = np.sum(gt_counts)
        recall = np.sum(hit_scores) / np.sum(gt_counts)
        score += weights[t] * recall
        logging.info(f"{t} hits@{K} = {n_hits} / gt@{K} = {n_gt}")
        logging.info(f"{t} recall@{K} = {recall}")

    logging.info("=============")
    logging.info(f"Overall Recall@{K} = {score}")
    logging.info("=============")

    return score


def cross_validation_ap_score(
    model: Union[ClassifierModel, RankingModel],
    algo: str,
    k: int,
) -> float:

    input_path = get_processed_training_train_dataset_dir()
    val_path = get_processed_training_test_dataset_dir()
    gt_path = get_processed_local_validation_dir()

    recall20s = []
    for _ in range(k):
        train_df = pl.DataFrame()

        for i in range(2):
            filepath = f"{input_path}/train_{i}_one_ranker_combined.parquet"
            df_chunk = pl.read_parquet(filepath)
            df_chunk = df_chunk.to_pandas()
            df_chunk = downsample(df_chunk)
            df_chunk = pl.from_pandas(df_chunk)
            train_df = pl.concat([train_df, df_chunk])

        logging.info(f"train shape {train_df.shape}")

        train_df = train_df.sort(by=["session", TARGET], reverse=[True, True])
        train_df = train_df.to_pandas()
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

        logging.info("split train and val dataset")
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
        n_group_train = list(group_train.value_counts())
        n_group_val = list(group_val.value_counts())
        logging.info(f"train shape {X_train.shape} sum groups {sum(n_group_train)}")
        logging.info(f"val shape {X_val.shape} sum groups {sum(n_group_val)}")
        logging.info(f"y_train {np.mean(y_train)} | y_val {np.mean(y_val)}")

        logging.info("fit ranker")
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

        del X_train, y_train, group_train, X_val, y_val, group_val
        gc.collect()

        # measure recall@20
        logging.info("score candidates and pick top 20")
        df_pred = pl.DataFrame()
        for EVENT in ["orders", "carts", "clicks"]:
            logging.info(f"event: {EVENT.upper()}")
            test_df = pl.DataFrame()
            df_chunk = pl.DataFrame()
            if EVENT == "orders":
                for i in range(8):
                    filepath = f"{val_path}/test_{i}_{EVENT}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    test_df = pl.concat([test_df, df_chunk])

            elif EVENT == "carts":
                for i in range(6):
                    filepath = f"{val_path}/test_{i}_{EVENT}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    test_df = pl.concat([test_df, df_chunk])

            else:
                for i in range(3):
                    filepath = f"{val_path}/test_{i}_{EVENT}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    test_df = pl.concat([test_df, df_chunk])

            test_df = test_df.to_pandas()
            # replace inf with 0
            # and make sure there's no None
            test_df = test_df.replace([np.inf, -np.inf], 0)
            test_df = test_df.fillna(0)

            X_test = test_df[selected_features]

            del df_chunk
            gc.collect()

            # predict val
            scores = model.predict(X_test)
            # select only session & candidate_aid cols
            test_df = pl.from_pandas(test_df)
            test_df = test_df.select([pl.col(["session", "candidate_aid", "label"])])
            # add scores columns
            # logging.info("merge with test_df")
            test_df = test_df.with_columns([pl.Series(name="score", values=scores)])

            del X_test
            gc.collect()

            test_predictions = (
                test_df.sort(["session", "score"], reverse=True)
                .groupby("session")
                .agg([pl.col("candidate_aid").limit(20).list().alias("labels")])
            )

            del test_df
            gc.collect()

            test_predictions = test_predictions.select([pl.col(["session", "labels"])])
            test_predictions = test_predictions.with_columns(
                [pl.lit(EVENT).alias("type")]
            )
            test_predictions = test_predictions.select(
                [pl.col(["session", "type", "labels"])]
            )

            df_pred = pl.concat([df_pred, test_predictions])
            logging.info(f"df_pred shape {df_pred.shape}")

            del test_predictions
            gc.collect()

        logging.info("measure recall@20")
        gt_path = f"{gt_path}/test_labels.parquet"
        df_truth = pd.read_parquet(gt_path)
        recall20 = measure_recall(df_pred=df_pred.to_pandas(), df_truth=df_truth, K=20)
        recall20s.append(recall20)

        del df_truth, df_pred
        gc.collect()
        # # predict val
        # y_proba = model.predict(X_val)
        # true_relevance = np.asarray([y_val])
        # score_relevance = np.asarray([y_proba])
        # pr_auc = ndcg_score(y_true=true_relevance, y_score=score_relevance, k=20)
        # val_pr_aucs.append(pr_auc)

    return float(np.mean(recall20s))


class ObjectiveLGBOneRankerModel:
    def __init__(self, algo: str, n_estimators: int, k: int):
        self.n_estimators = n_estimators
        self.algo = algo
        self.k = k

    def __call__(self, trial):
        hyperparams = {
            "n_estimators": trial.suggest_int(
                "n_estimators", self.n_estimators, self.n_estimators
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 400, 1500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1),
            "random_state": 747,
            "verbose": 0,
        }
        model: Union[ClassifierModel, RankingModel]
        if self.algo == "lgbm_ranker":
            model = LGBRanker(**hyperparams)
        # measure performance in CV
        cv_ap = cross_validation_ap_score(model, k=self.k, algo=self.algo)

        return cv_ap


class ObjectiveCATOneRankerModel:
    def __init__(self, algo: str, n_estimators: int, k: int):
        self.n_estimators = n_estimators
        self.algo = algo
        self.k = k

    def __call__(self, trial):
        hyperparams = {
            "n_estimators": trial.suggest_int(
                "n_estimators", self.n_estimators, self.n_estimators
            ),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("depth", 6, 10),
            "num_leaves": trial.suggest_float("l2_leaf_reg", 0, 1),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000),
            "feature_fraction": trial.suggest_float("max_leaves", 64, 200),
            "random_state": 747,
            "verbose": 0,
        }
        model: Union[ClassifierModel, RankingModel]
        if self.algo == "cat_ranker":
            model = CATRanker(**hyperparams)
        # measure performance in CV
        cv_ap = cross_validation_ap_score(model, k=self.k, algo=self.algo)

        return cv_ap


def perform_tuning(algo: str, events: list, k: int, n_estimators: int, n_trial: int):
    logging.info("measure Recall@20 before tuning")
    hyperparams = {"n_estimators": 1000}

    # check performance before tuning
    model: RankingModel
    if algo == "lgbm_ranker":
        model = LGBRanker(**hyperparams)

    mean_val_ap_before = cross_validation_ap_score(model=model, algo=algo, k=k)
    logging.info(f"VAL Recall@20 with {k} fold before tuning: {mean_val_ap_before}")

    logging.info("perform tuning")
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    if algo in ["lgbm_classifier", "lgbm_ranker"]:
        study.optimize(
            ObjectiveLGBOneRankerModel(algo=algo, n_estimators=n_estimators, k=k),
            n_trials=n_trial,
        )

    logging.info("complete tuning!")
    # find best hyperparams
    mean_val_ap_after = study.best_value
    best_hyperparams = study.best_params
    logging.info(f"Previous value: {mean_val_ap_before}")
    logging.info(f"Found best tuned value {mean_val_ap_after}")
    logging.info(f"Best hyperparams {best_hyperparams}")
    # if better, save to artifact dir
    performance_metric = {
        "before": mean_val_ap_before,
        "after": mean_val_ap_after,
    }
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # get current date
    unique_model_name = (
        f"{current_date}_one_ranker_{algo}_{mean_val_ap_before}_{mean_val_ap_after}"
    )
    artifact_path = get_artifacts_tuning_dir(event="one_ranker", week="w2")
    # create artifact dir
    filepath = artifact_path / unique_model_name
    check_directory(filepath)
    logging.info(f"saving artifacts to: {filepath}")
    hyper_path = f"{filepath}/tuned_hyperparams.json"
    write_json(filepath=hyper_path, data=best_hyperparams)

    # output performance metrics
    perf_path = f"{filepath}/performance_metric.json"
    write_json(filepath=perf_path, data=performance_metric)

    if mean_val_ap_after > mean_val_ap_before:
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
    default=1000,
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
    n_estimators: int = 1000,
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
