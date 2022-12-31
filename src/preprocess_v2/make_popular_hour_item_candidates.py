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
    get_processed_training_train_tmp_candidates_dir,
    get_processed_training_test_tmp_candidates_dir,
    get_processed_scoring_train_tmp_candidates_dir,
    get_processed_scoring_test_tmp_candidates_dir,
)
from src.features.preprocess_events import preprocess_ts
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def gen_popular_hour_item(data: pl.DataFrame):
    """
    df input
    session | type | ts | aid
    123 | 0 | 12313 | AID1
    123 | 1 | 12314 | AID1
    123 | 2 | 12345 | AID1
    """

    # START: event data preprocess
    data = preprocess_ts(data)
    # END: event data preprocess

    # agg per date
    data_agg = data.groupby(["hour", "aid"]).agg(
        [
            # num of event type
            (pl.col("type") == 0).sum().alias("item_click_count"),
            (pl.col("type") == 1).sum().alias("item_cart_count"),
            (pl.col("type") == 2).sum().alias("item_order_count"),
        ]
    )

    data_agg = data_agg.with_columns(
        [
            pl.col("item_click_count")
            .rank(reverse=True, method="ordinal")
            .over("hour")
            .alias("click_rank"),
            pl.col("item_cart_count")
            .rank(reverse=True, method="ordinal")
            .over("hour")
            .alias("cart_rank"),
            pl.col("item_order_count")
            .rank(reverse=True, method="ordinal")
            .over("hour")
            .alias("order_rank"),
        ]
    )

    # return top 20 popular click/cart/order
    popular_click = data_agg.filter(pl.col("click_rank") <= 20)
    popular_click = popular_click.sort(by="click_rank")
    popular_cart = data_agg.filter(pl.col("cart_rank") <= 20)
    popular_cart = popular_cart.sort(by="cart_rank")
    popular_order = data_agg.filter(pl.col("order_rank") <= 20)
    popular_order = popular_order.sort(by="order_rank")

    # agg for candidate list
    popular_click = popular_click.groupby("hour").agg(
        [pl.col("aid").list().alias("labels")]
    )
    popular_cart = popular_cart.groupby("hour").agg(
        [pl.col("aid").list().alias("labels")]
    )
    popular_order = popular_order.groupby("hour").agg(
        [pl.col("aid").list().alias("labels")]
    )

    return popular_click, popular_cart, popular_order


def make_popular_hour_candidates(
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
    logging.info(f"start creating popular item candidates")
    popular_click, popular_cart, popular_order = gen_popular_hour_item(data=df)

    filepath = output_path / f"{name}_popular_hour_clicks.parquet"
    logging.info(f"save chunk to: {filepath}")
    popular_click.write_parquet(f"{filepath}")
    logging.info(f"output df shape {popular_click.shape}")

    filepath = output_path / f"{name}_popular_hour_carts.parquet"
    logging.info(f"save chunk to: {filepath}")
    popular_cart.write_parquet(f"{filepath}")
    logging.info(f"output df shape {popular_cart.shape}")

    filepath = output_path / f"{name}_popular_hour_orders.parquet"
    logging.info(f"save chunk to: {filepath}")
    popular_order.write_parquet(f"{filepath}")
    logging.info(f"output df shape {popular_order.shape}")


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    if mode == "training_train":
        input_path = get_processed_training_train_splitted_dir()
        output_path = get_processed_training_train_tmp_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save tmp candidates to: {output_path}")
        make_popular_hour_candidates(
            name="train",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_tmp_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save tmp candidates to: {output_path}")
        make_popular_hour_candidates(
            name="test",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_tmp_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save tmp candidates to: {output_path}")
        make_popular_hour_candidates(
            name="train",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_tmp_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save tmp candidates to: {output_path}")
        make_popular_hour_candidates(
            name="test",
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
