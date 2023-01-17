import click
import polars as pl
from tqdm import tqdm
import numpy as np
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
    get_processed_training_train_item_covisitation_features_dir,  # output dir
    get_processed_training_test_item_covisitation_features_dir,
    get_processed_scoring_train_item_covisitation_features_dir,
    get_processed_scoring_test_item_covisitation_features_dir,
)
from src.utils.data import (
    get_top20_covisitation_click_df,
    get_top15_covisitation_buy2buy_df,
    get_top15_covisitation_buys_df,
)
from src.utils.memory import freemem, convert_float_to_int
from src.features.preprocess_events import preprocess_events
from src.utils.logger import get_logger

logging = get_logger()


def gen_user_covisit_features(
    data: pl.DataFrame,
    top_15_buys: pl.DataFrame,
    top_15_buy2buy: pl.DataFrame,
    top_20_clicks: pl.DataFrame,
):
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

    # left join with click and agg_click_df
    data_click = data.join(top_20_clicks, how="left", left_on="aid", right_on="aid_x")
    # data_click = data_click.with_columns(
    #     [(pl.col("log_recency_score") * pl.col("wgt")).alias("log_wgt")]
    # )
    df_agg_click = data_click.groupby(["session", "aid_y"]).agg(
        [
            # pl.col("wgt").sum().alias("session_covisit_click_wgt"),
            pl.col("wgt").count().alias("session_covisit_click_cnt"),
            # pl.col("log_wgt").sum().alias("session_covisit_click_log_wgt"),
        ]
    )

    # left join with buy2buy and agg_buy2buy_df
    data_buy2buy = data.join(
        top_15_buy2buy, how="left", left_on="aid", right_on="aid_x"
    )
    # data_buy2buy = data_buy2buy.with_columns(
    #     [(pl.col("log_recency_score") * pl.col("wgt")).alias("log_wgt")]
    # )
    df_agg_buy2buy = data_buy2buy.groupby(["session", "aid_y"]).agg(
        [
            # pl.col("wgt").sum().alias("session_covisit_buy2buy_wgt"),
            pl.col("wgt").count().alias("session_covisit_buy2buy_cnt"),
            # pl.col("log_wgt").sum().alias("session_covisit_buy2buy_log_wgt"),
        ]
    )

    # left join with buys and agg_buys_df
    data_buys = data.join(top_15_buys, how="left", left_on="aid", right_on="aid_x")
    # data_buys = data_buys.with_columns(
    #     [(pl.col("log_recency_score") * pl.col("wgt")).alias("log_wgt")]
    # )
    df_agg_buys = data_buys.groupby(["session", "aid_y"]).agg(
        [
            # pl.col("wgt").sum().alias("session_covisit_buys_wgt"),
            pl.col("wgt").count().alias("session_covisit_buys_cnt"),
            # pl.col("log_wgt").sum().alias("session_covisit_buys_log_wgt"),
        ]
    )

    # outer join click, buy2buy, and buys
    df_agg_click = df_agg_click.join(
        df_agg_buys, how="outer", on=["session", "aid_y"]
    ).join(df_agg_buy2buy, how="outer", on=["session", "aid_y"])

    # fill_null(0)
    df_agg_click = df_agg_click.fill_null(0)

    del df_agg_buys, df_agg_buy2buy, data_click, data_buy2buy, data_buys
    gc.collect()

    return df_agg_click


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

    if mode in ["training_train", "training_test"]:
        logging.info("read local covisitation buys")
        top_15_buys = get_top15_covisitation_buys_df()
        logging.info("read local covisitation buy2buy")
        top_15_buy2buy = get_top15_covisitation_buy2buy_df()
        logging.info("read local covisitation click")
        top_20_clicks = get_top20_covisitation_click_df()
    else:
        logging.info("read scoring covisitation buys")
        top_15_buys = get_top15_covisitation_buys_df(mode="scoring")
        logging.info("read scoring covisitation buy2buy")
        top_15_buy2buy = get_top15_covisitation_buy2buy_df(mode="scoring")
        logging.info("read scoring covisitation click")
        top_20_clicks = get_top20_covisitation_click_df(mode="scoring")

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(n)):
        # logging.info(f"chunk {ix}: read input")
        filepath = f"{input_path}/{name}_{ix}.parquet"
        df = pl.read_parquet(filepath)

        logging.info(f"start creating session covisitation features")
        df_output = gen_user_covisit_features(
            data=df,
            top_15_buys=top_15_buys,
            top_15_buy2buy=top_15_buy2buy,
            top_20_clicks=top_20_clicks,
        )
        df_output = freemem(df_output)
        df_output = convert_float_to_int(df_output)

        filepath = output_path / f"{name}_{ix}_session_covisit_feas.parquet"
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
        output_path = get_processed_training_train_item_covisitation_features_dir()
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
        output_path = get_processed_training_test_item_covisitation_features_dir()
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
        output_path = get_processed_scoring_train_item_covisitation_features_dir()
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
        output_path = get_processed_scoring_test_item_covisitation_features_dir()
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
