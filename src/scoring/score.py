# step1: score input: model & test combined parquet | output -> rows & list (only take top 20)
# step2.a: make_submisssion input: scores click, cart, order | output -> concat all rows event
# step3: evaluate input: submission (all event)

# step2.b-2: optimize_ensemble_submisssion input: scores click, cart, order | intermiediate: add weight to model, create list per event |output -> concat all rows event
# step2.b-2: make_ensemble_submisssion input: scores click, cart, order | intermiediate: create list per event |output -> concat all rows event

import click
import polars as pl
from tqdm import tqdm
import numpy as np
import gc
import joblib
from pathlib import Path
from src.utils.constants import (
    get_processed_scoring_test_dataset_dir,
    get_processed_training_test_dataset_dir,
    get_data_output_local_submission_dir,  # scoring output dir
    get_data_output_submission_dir,
)

from src.utils.constants import get_artifacts_training_dir

from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def scoring(artifact: str, event: str, week: str):
    # for each event
    # read trained ensemble model
    # iterate 10 chunk of test data
    # for each N: read training data, predict, merge score to dataframe, save dataframe
    N_test = 10

    input_path: Path
    if week == "w1":
        input_path = get_processed_scoring_test_dataset_dir()
    elif week == "w2":
        input_path = get_processed_training_test_dataset_dir()
    else:
        raise NotImplementedError("week not implemented! (w1/w2)")

    EVENT = event
    artifact_path = get_artifacts_training_dir(event=EVENT, week=week)
    model_path = f"{artifact_path}/{artifact}/model.pkl"
    model = joblib.load(model_path)
    logging.info(f"ensemble model for event {EVENT.upper()} loaded!")

    logging.info("start scoring")
    for IX in tqdm(range(N_test)):
        logging.info(f"read test data for chunk: {IX}")
        filepath = f"{input_path}/test_{IX}_{EVENT}_combined.parquet"
        test_df = pl.read_parquet(filepath)
        logging.info(test_df.shape)

        logging.info("select features")
        selected_features = test_df.columns
        selected_features.remove("session")
        selected_features.remove("candidate_aid")
        selected_features.remove("label")

        # select features
        X_test = test_df[selected_features].to_pandas()
        # perform scoring
        logging.info("perform scoring")
        scores = model.predict(X_test)
        # select only session & candidate_aid cols
        test_df = test_df.select([pl.col(["session", "candidate_aid", "label"])])
        # add scores columns
        logging.info("merge with test_df")
        test_df = test_df.with_columns([pl.Series(name="score", values=scores)])

        # save to parquet
        if week == "w1":
            output_path = get_data_output_submission_dir(event=EVENT, model=artifact)
        elif week == "w2":
            output_path = get_data_output_local_submission_dir(
                event=EVENT, model=artifact
            )
        filepath = f"{output_path}/test_{IX}_{EVENT}_scores.parquet"
        test_df.write_parquet(f"{filepath}")
        logging.info(f"save chunk to: {filepath}")
        logging.info(f"output df shape {test_df.shape}")

        del X_test, test_df
        gc.collect()

    logging.info(f"scoring complete for {EVENT.upper()}!")


@click.command()
@click.option(
    "--artifact",
    default="2022-12-05_catboost_46313_82997",
    help="artifact folder for reading model.pkl",
)
@click.option(
    "--event",
    default="orders",
    help="avaiable event: clicks/carts/orders/all",
)
@click.option(
    "--week",
    default="w2",
    help="subset of test data, w1/w2; w1:scoring dir, w2:training dir",
)
def main(event: str = "all", artifact: str = "lgbm", week: str = "w2", n: int = 1):
    if event not in ["clicks", "carts", "orders"]:
        raise ValueError("available event: clicks, carts, orders")

    scoring(artifact=artifact, event=event, week=week)


if __name__ == "__main__":
    main()
