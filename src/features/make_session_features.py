from datetime import datetime
import click
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
from collections import Counter
import gc
from pathlib import Path
from src.utils.constants import (
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
    get_processed_training_train_sess_features_dir,
    get_processed_training_test_sess_features_dir,
    get_processed_scoring_train_sess_features_dir,
    get_processed_scoring_test_sess_features_dir,
)
from src.utils.logger import get_logger

logging = get_logger()


def count_distinct(arr):
    return len(Counter(arr).keys())


def get_hour_from_ts(x):
    return int(datetime.utcfromtimestamp(x).strftime("%H"))


def get_weekday_from_ts(x):
    """Return the day of the week as an integer, where Monday is 0 and Sunday is 6."""
    return datetime.utcfromtimestamp(x).weekday()


def get_datetime_from_ts(x):
    return datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def gen_user_features(data: pl.DataFrame):
    """
    df input
    session | type | ts | aid
    123 | 0 | 12313 | AID1
    123 | 1 | 12314 | AID1
    123 | 2 | 12345 | AID1
    """

    # sort session & ts ascendingly
    data = data.sort(["session", "ts"])

    # shift ts per session & get duration between event
    data = data.with_columns([pl.col("ts").shift().over("session").alias("prev_ts")])
    data = data.with_columns(
        [(pl.col("ts") - pl.col("prev_ts")).fill_null(0).alias("duration_second")]
    )

    # agg per session
    data_agg = data.groupby("session").agg(
        [
            pl.col("aid").count().alias("sess_all_events_count"),
            pl.col("aid").n_unique().alias("sess_aid_dcount"),
            # num of event type
            (pl.col("type") == 0).sum().alias("sess_click_count"),
            (pl.col("type") == 1).sum().alias("sess_cart_count"),
            (pl.col("type") == 2).sum().alias("sess_order_count"),
            # aid dcount per event type
            pl.col("aid")
            .filter(pl.col("type") == 0)
            .n_unique()
            .fill_null(0)
            .alias("sess_clicked_aid_dcount"),
            pl.col("aid")
            .filter(pl.col("type") == 1)
            .n_unique()
            .fill_null(0)
            .alias("sess_carted_aid_dcount"),
            pl.col("aid")
            .filter(pl.col("type") == 2)
            .n_unique()
            .fill_null(0)
            .alias("sess_ordered_aid_dcount"),
            ((pl.col("ts").max() - pl.col("ts").min()) / 60).alias(
                "sess_duration_mins"
            ),
            ((pl.col("ts").max() - pl.col("ts").min()) / 60 / 60).alias(
                "sess_duration_hour"
            ),
            # duration per event type
            pl.col("duration_second")
            .filter(pl.col("type") == 0)
            .mean()
            .fill_null(0)
            .alias("sess_avg_click_dur_sec"),
            pl.col("duration_second")
            .filter(pl.col("type") == 1)
            .mean()
            .fill_null(0)
            .alias("sess_avg_cart_dur_sec"),
            pl.col("duration_second")
            .filter(pl.col("type") == 2)
            .mean()
            .fill_null(0)
            .alias("sess_avg_order_dur_sec"),
            # event type
            pl.col("type").n_unique().alias("sess_type_dcount"),
            # for extracting hour and day of last event
            pl.col("ts").last().alias("curr_ts"),
        ]
    )

    data_agg = data_agg.with_columns(
        [
            pl.col("curr_ts").apply(lambda x: get_hour_from_ts(x)).alias("sess_hour"),
            pl.col("curr_ts")
            .apply(lambda x: get_weekday_from_ts(x))
            .alias("sess_weekday"),
        ],
    )

    # drop cols
    data_agg = data_agg.drop(columns=["curr_ts"])

    return data_agg


def make_session_features(
    name: str,
    input_path: Path,
    output_path: Path,
):

    n = 10
    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(n)):
        for event in ["clicks", "carts", "orders"]:
            # logging.info(f"chunk {ix}: read input")
            filepath = f"{input_path}/{name}_{ix}.parquet"
            df = pl.read_parquet(filepath)

            logging.info(f"start creating session features")
            df_output = gen_user_features(data=df)

            filepath = output_path / f"{name}_{ix}_session_feas.parquet"
            logging.info(f"save chunk to: {filepath}")
            df_output.write_parquet(f"{filepath}")
            logging.info(f"output df shape {df_output.shape}")

            del df, df_output
            gc.collect()


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    if mode == "training_train":
        input_path = get_processed_training_train_splitted_dir()
        output_path = get_processed_training_train_sess_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="train",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_sess_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="test",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_sess_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="train",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_sess_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="test",
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
