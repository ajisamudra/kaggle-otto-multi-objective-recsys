import click
import polars as pl
from tqdm import tqdm
import gc
import numpy as np
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
    get_processed_training_train_item_hour_features_dir,
    get_processed_training_test_item_hour_features_dir,
    get_processed_scoring_train_item_hour_features_dir,
    get_processed_scoring_test_item_hour_features_dir,
)
from src.features.preprocess_events import preprocess_events
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def gen_item_hour_features(data: pl.DataFrame):
    """
    df input
    session | type | ts | aid
    123 | 0 | 12313 | AID1
    123 | 1 | 12314 | AID1
    123 | 2 | 12345 | AID1
    """

    # START: event data preprocess
    data = preprocess_events(data)
    # END: event data preprocess

    # agg per aid
    data_agg = data.groupby(["aid", "hour"]).agg(
        [
            pl.col("type").count().alias("itemXhour_all_events_count"),
            # num of event type
            (pl.col("type") == 0).sum().alias("itemXhour_click_count"),
            (pl.col("type") == 1).sum().alias("itemXhour_cart_count"),
            (pl.col("type") == 2).sum().alias("itemXhour_order_count"),
        ]
    )

    # conversion rate per event
    data_agg = data_agg.with_columns(
        [
            # window aid
            pl.col("itemXhour_click_count")
            .sum()
            .over("aid")
            .alias("itemXhour_all_click_count"),
            pl.col("itemXhour_cart_count")
            .sum()
            .over("aid")
            .alias("itemXhour_all_cart_count"),
            pl.col("itemXhour_order_count")
            .sum()
            .over("aid")
            .alias("itemXhour_all_order_count"),
            # window hour
            pl.col("itemXhour_click_count")
            .sum()
            .over("hour")
            .alias("itemXhour_all_hour_click_count"),
            pl.col("itemXhour_cart_count")
            .sum()
            .over("hour")
            .alias("itemXhour_all_hour_cart_count"),
            pl.col("itemXhour_order_count")
            .sum()
            .over("hour")
            .alias("itemXhour_all_hour_order_count"),
            (pl.col("itemXhour_cart_count") / pl.col("itemXhour_click_count"))
            .fill_nan(0)
            .alias("itemXhour_click_to_cart_cvr"),
            # (pl.col("itemXhour_order_count") / pl.col("itemXhour_cart_count"))
            # .fill_nan(0)
            # .alias("itemXhour_cart_to_order_cvr"),
        ],
    )

    # frac event per hour
    data_agg = data_agg.with_columns(
        [
            # frac compare to its aid
            (pl.col("itemXhour_click_count") / pl.col("itemXhour_all_click_count"))
            .fill_nan(0)
            .alias("itemXhour_frac_click_all_click_count"),
            # (pl.col("itemXhour_cart_count") / pl.col("itemXhour_all_cart_count"))
            # .fill_nan(0)
            # .alias("itemXhour_frac_cart_all_cart_count"),
            # (pl.col("itemXhour_order_count") / pl.col("itemXhour_all_order_count"))
            # .fill_nan(0)
            # .alias("itemXhour_frac_order_all_order_count"),
            # frac compare to total event in particular hour
            # represent popularity at that hour
            (pl.col("itemXhour_click_count") / pl.col("itemXhour_all_hour_click_count"))
            .fill_nan(0)
            .alias("itemXhour_frac_click_all_hour_click_count"),
            # (pl.col("itemXhour_cart_count") / pl.col("itemXhour_all_hour_cart_count"))
            # .fill_nan(0)
            # .alias("itemXhour_frac_cart_all_hour_cart_count"),
            # (pl.col("itemXhour_order_count") / pl.col("itemXhour_all_hour_order_count"))
            # .fill_nan(0)
            # .alias("itemXhour_frac_order_all_hour_order_count"),
        ],
    )

    # drop cols
    data_agg = data_agg.drop(
        columns=[
            "itemXhour_all_click_count",
            "itemXhour_all_cart_count",
            "itemXhour_all_order_count",
            "itemXhour_all_hour_click_count",
            "itemXhour_all_hour_cart_count",
            "itemXhour_all_hour_order_count",
            "itemXhour_cart_count",
            "itemXhour_order_count",
        ]
    )

    return data_agg


def make_session_features(
    name: str,
    mode: str,
    input_path: Path,
    output_path: Path,
):

    if mode == "training_train":
        n = CFG.N_train
    elif mode == "training_test":
        n = CFG.N_local_test
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    df = pl.DataFrame()
    for ix in tqdm(range(n)):
        filepath = f"{input_path}/{name}_{ix}.parquet"
        df_chunk = pl.read_parquet(filepath)
        df = pl.concat([df, df_chunk])

    logging.info(f"input df shape {df.shape}")
    logging.info(f"start creating itemXhour features")
    df_output = gen_item_hour_features(data=df)
    df_output = freemem(df_output)

    filepath = output_path / f"{name}_item_hour_feas.parquet"
    logging.info(f"save chunk to: {filepath}")
    # replace inf with 0
    # and make sure there's no None
    df_output = df_output.to_pandas()
    df_output = df_output.replace([np.inf, -np.inf], 0)
    df_output = df_output.fillna(0)
    df_output = pl.from_pandas(df_output)
    df_output.write_parquet(f"{filepath}")
    logging.info(f"output df shape {df_output.shape}")


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    if mode == "training_train":
        input_path = get_processed_training_train_splitted_dir()
        output_path = get_processed_training_train_item_hour_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_item_hour_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_item_hour_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_item_hour_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
