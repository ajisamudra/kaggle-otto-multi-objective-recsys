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
    CFG,
    get_processed_scoring_test_dataset_dir,
    get_processed_training_test_dataset_dir,
    get_data_output_local_submission_dir,  # scoring output dir
    get_data_output_submission_dir,
)

from src.utils.constants import get_artifacts_training_dir
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def scoring(artifact: str, event: str, week_data: str, week_model: str):
    # for each event
    # read trained ensemble model
    # iterate 10 chunk of test data
    # for each N: read training data, predict, merge score to dataframe, save dataframe
    N_test = CFG.N_test

    input_path: Path
    if week_data == "w1":
        input_path = get_processed_scoring_test_dataset_dir()
    elif week_data == "w2":
        input_path = get_processed_training_test_dataset_dir()
    else:
        raise NotImplementedError("week not implemented! (w1/w2)")

    EVENT = event
    artifact_path = get_artifacts_training_dir(event="one_ranker", week=week_model)
    model_path = f"{artifact_path}/{artifact}/model.pkl"
    model = joblib.load(model_path)
    logging.info(f"ensemble model for event {EVENT.upper()} loaded!")

    logging.info("start scoring")
    for IX in tqdm(range(N_test)):
        # logging.info(f"read test data for chunk: {IX}")
        filepath = f"{input_path}/test_{IX}_{EVENT}_combined.parquet"
        test_df = pl.read_parquet(filepath)
        logging.info(f"input shape {test_df.shape}")

        # logging.info("select features")
        selected_features = test_df.columns
        selected_features.remove("session")
        selected_features.remove("candidate_aid")

        selected_features.remove("matrix_fact_cart_order_last_event_cosine_distance")
        selected_features.remove("matrix_fact_cart_order_last_event_euclidean_distance")
        selected_features.remove("matrix_fact_cart_order_max_recency_cosine_distance")
        selected_features.remove(
            "matrix_fact_cart_order_max_recency_euclidean_distance"
        )
        selected_features.remove(
            "matrix_fact_cart_order_max_weighted_recency_cosine_distance"
        )
        selected_features.remove(
            "matrix_fact_cart_order_max_weighted_recency_euclidean_distance"
        )
        selected_features.remove("matrix_fact_cart_order_max_duration_cosine_distance")
        selected_features.remove(
            "matrix_fact_cart_order_max_duration_euclidean_distance"
        )
        selected_features.remove(
            "matrix_fact_cart_order_max_weighted_duration_cosine_distance"
        )
        selected_features.remove(
            "matrix_fact_cart_order_max_weighted_duration_euclidean_distance"
        )
        selected_features.remove("matrix_fact_buy2buy_last_event_cosine_distance")
        selected_features.remove("matrix_fact_buy2buy_last_event_euclidean_distance")
        selected_features.remove("matrix_fact_buy2buy_max_recency_cosine_distance")
        selected_features.remove("matrix_fact_buy2buy_max_recency_euclidean_distance")
        selected_features.remove(
            "matrix_fact_buy2buy_max_weighted_recency_cosine_distance"
        )
        selected_features.remove(
            "matrix_fact_buy2buy_max_weighted_recency_euclidean_distance"
        )
        selected_features.remove("matrix_fact_buy2buy_max_duration_cosine_distance")
        selected_features.remove("matrix_fact_buy2buy_max_duration_euclidean_distance")
        selected_features.remove(
            "matrix_fact_buy2buy_max_weighted_duration_cosine_distance"
        )
        selected_features.remove(
            "matrix_fact_buy2buy_max_weighted_duration_euclidean_distance"
        )

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

        # select features
        X_test = test_df[selected_features].to_pandas()

        # replace inf with 0
        X_test = X_test.replace([np.inf, -np.inf], 0)
        X_test = X_test.fillna(0)

        # perform scoring
        # logging.info("perform scoring")
        scores = model.predict(X_test)
        # select only session & candidate_aid cols
        test_df = test_df.select([pl.col(["session", "candidate_aid", "label"])])
        # add scores columns
        # logging.info("merge with test_df")
        test_df = test_df.with_columns([pl.Series(name="score", values=scores)])

        del X_test
        gc.collect()

        # save to parquet
        if week_data == "w1":
            output_path = get_data_output_submission_dir(
                event=EVENT, model=artifact, week_model=week_model
            )
        elif week_data == "w2":
            output_path = get_data_output_local_submission_dir(
                event=EVENT, model=artifact, week_model=week_model
            )

        filepath = f"{output_path}/test_{IX}_{EVENT}_scores.parquet"
        test_df = freemem(test_df)
        test_df.write_parquet(f"{filepath}")

        # logging.info(f"save prediction scores to: {filepath}")
        # logging.info(f"output df shape {test_df.shape}")

        # logging.info(f"rank & select top 20")
        # take top 20 candidate aid and save it as list
        test_predictions = (
            test_df.sort(["session", "score"], reverse=True)
            .groupby("session")
            .agg([pl.col("candidate_aid").limit(20).list().alias("labels")])
        )

        del test_df
        gc.collect()

        test_predictions = test_predictions.select([pl.col(["session", "labels"])])
        test_predictions = test_predictions.with_columns(
            [(pl.col(["session"]) + f"_{EVENT}").alias("session_type")]
        )
        test_predictions = test_predictions.select([pl.col(["session_type", "labels"])])
        test_predictions = test_predictions.with_columns(
            [pl.col("labels").apply(lambda x: " ".join(map(str, x))).alias("labels")]
        )

        filepath = f"{output_path}/test_{IX}_{EVENT}_submission.parquet"
        test_predictions = freemem(test_predictions)
        test_predictions.write_parquet(f"{filepath}")
        logging.info(f"save prediction submission to: {filepath}")
        logging.info(f"output df shape {test_predictions.shape}")

        del test_predictions
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
    "--week_data",
    default="w2",
    help="subset of test data, w1/w2; w1:scoring dir, w2:training dir",
)
@click.option(
    "--week_model",
    default="w2",
    help="on which training data the model was trained, w1/w2; w1:scoring dir, w2:training dir",
)
def main(
    event: str = "all",
    artifact: str = "lgbm",
    week_data: str = "w2",
    week_model: str = "w2",
):
    if event not in ["clicks", "carts", "orders"]:
        raise ValueError("available event: clicks, carts, orders")

    scoring(artifact=artifact, event=event, week_data=week_data, week_model=week_model)


if __name__ == "__main__":
    main()
