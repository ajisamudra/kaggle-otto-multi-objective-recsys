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
    get_processed_training_train_item_weekday_features_dir,
    get_processed_training_test_item_weekday_features_dir,
    get_processed_scoring_train_item_weekday_features_dir,
    get_processed_scoring_test_item_weekday_features_dir,
)
from src.features.preprocess_events import preprocess_events
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def gen_item_weekday_features(data: pl.DataFrame):
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
    data_agg = data.groupby(["aid", "weekday"]).agg(
        [
            pl.col("type").count().alias("itemXweekday_all_events_count"),
            # num of event type
            (pl.col("type") == 0).sum().alias("itemXweekday_click_count"),
            (pl.col("type") == 1).sum().alias("itemXweekday_cart_count"),
            (pl.col("type") == 2).sum().alias("itemXweekday_order_count"),
        ]
    )

    # conversion rate per event
    data_agg = data_agg.with_columns(
        [
            # window aid
            pl.col("itemXweekday_click_count")
            .sum()
            .over("aid")
            .alias("itemXweekday_all_click_count"),
            pl.col("itemXweekday_cart_count")
            .sum()
            .over("aid")
            .alias("itemXweekday_all_cart_count"),
            pl.col("itemXweekday_order_count")
            .sum()
            .over("aid")
            .alias("itemXweekday_all_order_count"),
            # window weekday
            pl.col("itemXweekday_click_count")
            .sum()
            .over("weekday")
            .alias("itemXweekday_all_weekday_click_count"),
            pl.col("itemXweekday_cart_count")
            .sum()
            .over("weekday")
            .alias("itemXweekday_all_weekday_cart_count"),
            pl.col("itemXweekday_order_count")
            .sum()
            .over("weekday")
            .alias("itemXweekday_all_weekday_order_count"),
            (pl.col("itemXweekday_cart_count") / pl.col("itemXweekday_click_count"))
            .fill_nan(0)
            .alias("itemXweekday_click_to_cart_cvr"),
            (pl.col("itemXweekday_order_count") / pl.col("itemXweekday_cart_count"))
            .fill_nan(0)
            .alias("itemXweekday_cart_to_order_cvr"),
        ],
    )

    # frac event per hourda
    data_agg = data_agg.with_columns(
        [
            # frac compare to its aid
            (
                pl.col("itemXweekday_click_count")
                / pl.col("itemXweekday_all_click_count")
            )
            .fill_nan(0)
            .alias("itemXweekday_frac_click_all_click_count"),
            (pl.col("itemXweekday_cart_count") / pl.col("itemXweekday_all_cart_count"))
            .fill_nan(0)
            .alias("itemXweekday_frac_cart_all_cart_count"),
            (
                pl.col("itemXweekday_order_count")
                / pl.col("itemXweekday_all_order_count")
            )
            .fill_nan(0)
            .alias("itemXweekday_frac_order_all_order_count"),
            # frac compare to total event in particular weekday
            # represent popularity at that weekday
            (
                pl.col("itemXweekday_click_count")
                / pl.col("itemXweekday_all_weekday_click_count")
            )
            .fill_nan(0)
            .alias("itemXweekday_frac_click_all_weekday_click_count"),
            (
                pl.col("itemXweekday_cart_count")
                / pl.col("itemXweekday_all_weekday_cart_count")
            )
            .fill_nan(0)
            .alias("itemXweekday_frac_cart_all_weekday_cart_count"),
            (
                pl.col("itemXweekday_order_count")
                / pl.col("itemXweekday_all_weekday_order_count")
            )
            .fill_nan(0)
            .alias("itemXweekday_frac_order_all_weekday_order_count"),
        ],
    )

    # drop cols
    data_agg = data_agg.drop(
        columns=[
            "itemXweekday_all_click_count",
            "itemXweekday_all_cart_count",
            "itemXweekday_all_order_count",
            "itemXweekday_all_weekday_click_count",
            "itemXweekday_all_weekday_cart_count",
            "itemXweekday_all_weekday_order_count",
        ]
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
    df = pl.DataFrame()
    for ix in tqdm(range(n)):
        filepath = f"{input_path}/{name}_{ix}.parquet"
        df_chunk = pl.read_parquet(filepath)
        df = pl.concat([df, df_chunk])

    logging.info(f"input df shape {df.shape}")
    logging.info(f"start creating itemXweekday features")
    df_output = gen_item_weekday_features(data=df)
    df_output = freemem(df_output)

    filepath = output_path / f"{name}_item_weekday_feas.parquet"
    logging.info(f"save chunk to: {filepath}")
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
        output_path = get_processed_training_train_item_weekday_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="train",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_item_weekday_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="test",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_item_weekday_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="train",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_item_weekday_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_features(
            name="test",
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
