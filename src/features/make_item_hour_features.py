import click
import polars as pl
from tqdm import tqdm
import gc
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
            (pl.col("itemXhour_cart_count") / pl.col("itemXhour_click_count"))
            .fill_nan(0)
            .alias("itemXhour_click_to_cart_cvr"),
            (pl.col("itemXhour_order_count") / pl.col("itemXhour_cart_count"))
            .fill_nan(0)
            .alias("itemXhour_cart_to_order_cvr"),
        ],
    )

    return data_agg


def make_session_features(
    name: str,
    input_path: Path,
    output_path: Path,
):

    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(n)):
        # logging.info(f"chunk {ix}: read input")
        filepath = f"{input_path}/{name}_{ix}.parquet"
        df = pl.read_parquet(filepath)

        logging.info(f"start creating item features")
        df_output = gen_item_hour_features(data=df)

        filepath = output_path / f"{name}_{ix}_item_hour_feas.parquet"
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
        output_path = get_processed_training_train_item_hour_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="train",
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
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
