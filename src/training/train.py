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
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils.constants import (
    write_json,
    get_processed_training_train_dataset_dir,  # final dataset dir
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
)

# for enabling training + scoring
from src.scoring.score import scoring
from src.scoring.make_submission import make_submission
from src.scoring.eval_submission import eval_submission

from src.utils.constants import get_artifacts_training_dir, check_directory

from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


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
    if week == "w1":
        input_path = get_processed_scoring_train_dataset_dir()
        if eval == 1:
            raise ValueError("Can't eval submission for training with data w1")
    elif week == "w2":
        input_path = get_processed_training_train_dataset_dir()
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
        for IX in tqdm(range(n)):
            model: Union[ClassifierModel, RankingModel]
            # hyperparams click

            if algo == "lgbm_classifier":
                model = LGBClassifier()
            elif algo == "cat_classifier":
                model = CatClassifier()
            elif algo == "lgbm_ranker":
                model = LGBRanker()
            elif algo == "cat_ranker":
                model = CATRanker()
            else:
                raise NotImplementedError("algorithm not implemented! (lgbm/catboost)")

            logging.info(f"read training data for chunk: {IX}")
            filepath = f"{input_path}/train_{IX}_{EVENT}_combined.parquet"
            train_df = pl.read_parquet(filepath)
            logging.info(train_df.shape)

            # sort data based on session & label
            train_df = train_df.sort(by=["session", TARGET], reverse=[True, True])

            selected_features = train_df.columns
            selected_features.remove("session")
            selected_features.remove("candidate_aid")
            # # remove item features
            # selected_features.remove("item_all_events_count")
            # selected_features.remove("item_click_count")
            # selected_features.remove("item_cart_count")
            # selected_features.remove("item_order_count")
            # selected_features.remove("item_click_to_cart_cvr")
            # selected_features.remove("item_cart_to_order_cvr")
            # selected_features.remove("itemXhour_all_events_count")
            # selected_features.remove("itemXhour_click_count")
            # selected_features.remove("itemXhour_cart_count")
            # selected_features.remove("itemXhour_order_count")
            # selected_features.remove("itemXhour_click_to_cart_cvr")
            # selected_features.remove("itemXhour_cart_to_order_cvr")
            # selected_features.remove("itemXweekday_all_events_count")
            # selected_features.remove("itemXweekday_click_count")
            # selected_features.remove("itemXweekday_cart_count")
            # selected_features.remove("itemXweekday_order_count")
            # selected_features.remove("itemXweekday_click_to_cart_cvr")
            # selected_features.remove("itemXweekday_cart_to_order_cvr")

            selected_features.remove(TARGET)

            # select X & y per train & val
            X = train_df[selected_features].to_pandas()
            group = train_df["session"].to_pandas()
            y = train_df[TARGET].to_pandas()

            # split train and validation using StratifiedGroupKFold
            # X_train, X_val, y_train, y_val, group_train, group_val = train_test_split(
            #     X, y, group, test_size=0.2, stratify=y, random_state=745
            # )

            skgfold = StratifiedGroupKFold(n_splits=5)
            train_idx, val_idx = [], []

            for tidx, vidx in skgfold.split(X, y, groups=group):
                train_idx, val_idx = tidx, vidx

            X_train, X_val = X.iloc[train_idx, :], X.iloc[val_idx, :]
            y_train, y_val = y[train_idx], y[val_idx]
            group_train, group_val = group[train_idx], group[val_idx]

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
            scoring(
                artifact=unique_model_name, event=EVENT, week_data="w2", week_model="w2"
            )
            # append unique_model_names for make & eval submission
            unique_model_names.append(unique_model_name)

    if eval == 1:
        # make submission
        # append unique_model_names for make & eval submission
        make_submission(
            click_model=unique_model_names[0],
            cart_model=unique_model_names[1],
            order_model=unique_model_names[2],
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
            click_model=unique_model_names[0],
            cart_model=unique_model_names[1],
            order_model=unique_model_names[2],
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
    events = ["clicks", "carts", "orders"]
    if event != "all":
        events = [event]
        if eval == 1:
            raise ValueError("Can't eval if not all events trained on")
    train(algo=algo, events=events, week=week, n=n, eval=eval)


if __name__ == "__main__":
    main()
