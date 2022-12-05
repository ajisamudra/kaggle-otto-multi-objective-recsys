import click
import polars as pl
import pandas as pd
from tqdm import tqdm
import numpy as np
import json
import gc
import joblib
import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.constants import (
    write_json,
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_scoring_train_dataset_dir,
    get_processed_training_test_dataset_dir,
)
from src.model.model import (
    EnsembleClassifierModels,
    ClassifierModel,
    LGBClassifier,
    CatClassifier,
)
from src.metrics.model_evaluation import (
    summarise_feature_importance,
    plot_and_save_feature_importance,
)
from src.utils.constants import get_artifacts_training_dir, check_directory

from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def train(algo: str, events: list, week: str, n: int):
    # for each event
    # initiate ensemble model
    # iterate N
    # for each N: read training data, train_test_split, fit model
    # append to ensemble_model, save feature importance, measure ROC/PR AUC per chunk
    # save ensemble model for particular event,
    # score validation on particular event
    # measure recall@20 in that event
    # continue on next event

    input_path: Path
    if week == "w1":
        input_path = get_processed_scoring_train_dataset_dir()
    elif week == "w2":
        input_path = get_processed_training_train_dataset_dir()
    else:
        raise NotImplementedError("week not implemented! (w1/w2)")

    event_train_roc_aucs, event_val_roc_aucs = [], []
    event_train_pr_aucs, event_val_pr_aucs = [], []
    for EVENT in events:
        performance, hyperparams = {}, {}
        logging.info(f"init ensemble for event {EVENT.upper()}")
        ensemble_model = EnsembleClassifierModels()
        train_roc_aucs, train_pr_aucs = [], []
        val_roc_aucs, val_pr_aucs = [], []
        feature_importances = []
        for IX in tqdm(range(n)):
            model: ClassifierModel
            # hyperparams click

            if algo == "lgbm":
                model = LGBClassifier()
            elif algo == "catboost":
                model = CatClassifier()
            else:
                raise NotImplementedError("algorithm not implemented! (lgbm/catboost)")

            logging.info(f"read training data for chunk: {IX}")
            filepath = f"{input_path}/train_{IX}_{EVENT}_combined.parquet"
            train_df = pl.read_parquet(filepath)
            logging.info(train_df.shape)

            selected_features = train_df.columns
            selected_features.remove("session")
            selected_features.remove("candidate_aid")
            selected_features.remove(TARGET)

            X = train_df[selected_features].to_pandas()
            y = train_df[TARGET].to_pandas()

            # split train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=745
            )
            logging.info(f"y_train {np.mean(y_train)} | y_val {np.mean(y_val)}")
            model.fit(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
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

            del train_df, X, y, X_train, X_val, y_train, y_val
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
            f"{current_date}_{algo}_{mean_val_pr_auc}_{mean_val_roc_auc}"
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

        # pickle the model
        model_name = f"{filepath}/model.pkl"
        joblib.dump(ensemble_model, model_name)

        # output hyperparams model
        hyper_path = f"{filepath}/hyperparams.json"
        write_json(filepath=hyper_path, data=hyperparams)

        # output performance metrics
        perf_path = f"{filepath}/performance_metric.json"
        write_json(filepath=perf_path, data=performance)

        # # score validation and measure performance
        # logging.info(f"SCORE VALIDATION RECALL@20")
        # for i in range(10):
        #     # read chunk
        #     path = get_processed_training_test_dataset_dir()
        #     filename = f"{path}/test_{i}_{EVENT}_combined.parquet"
        #     val_df = pd.read_parquet(filename)
        #     X = val_df[selected_features]

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


@click.command()
@click.option(
    "--event",
    default="all",
    help="avaiable event: clicks/carts/orders/all",
)
@click.option(
    "--algo",
    default="lgbm",
    help="algorithm for training; lgbm/catboost",
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
def main(event: str = "all", algo: str = "lgbm", week: str = "w2", n: int = 1):
    events = ["clicks", "carts", "orders"]
    if event != "all":
        events = [event]
    train(algo=algo, events=events, week=week, n=n)


if __name__ == "__main__":
    main()