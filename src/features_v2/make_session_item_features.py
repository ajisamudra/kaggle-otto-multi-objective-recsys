import click
import polars as pl
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
    get_processed_training_train_sess_item_features_dir,
    get_processed_training_test_sess_item_features_dir,
    get_processed_scoring_train_sess_item_features_dir,
    get_processed_scoring_test_sess_item_features_dir,
)
from src.features.preprocess_events import preprocess_events
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()

# input_path = get_processed_training_train_splitted_dir()
# cand_df = pl.read_parquet(f"{input_path}/train_0.parquet")
# cand_df.head()


def gen_session_item_features(data: pl.DataFrame):
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

    # get last ts per session
    data = data.with_columns([pl.col("ts").last().over("session").alias("curr_ts")])

    # agg per session X aid
    data_agg = data.groupby(["session", "aid"]).agg(
        [
            pl.col("aid").count().alias("sesXaid_events_count"),
            (pl.col("curr_ts") - pl.col("ts"))
            .mean()
            .fill_null(0)
            .alias("sesXaid_sec_from_last_event"),
            # num of event type
            (pl.col("type") == 0).sum().alias("sesXaid_click_count"),
            (pl.col("type") == 1).sum().alias("sesXaid_cart_count"),
            (pl.col("type") == 2).sum().alias("sesXaid_order_count"),
            # avg duration per event type
            pl.col("duration_second")
            .filter(pl.col("type") == 0)
            .mean()
            .fill_null(0)
            .alias("sesXaid_avg_click_dur_sec"),
            pl.col("duration_second")
            .filter(pl.col("type") == 1)
            .mean()
            .fill_null(0)
            .alias("sesXaid_avg_cart_dur_sec"),
            # pl.col("duration_second")
            # .filter(pl.col("type") == 2)
            # .mean()
            # .fill_null(0)
            # .alias("sesXaid_avg_order_dur_sec"),
            # sum duration per event type
            pl.col("duration_second")
            .filter(pl.col("type") == 0)
            .sum()
            .fill_null(0)
            .alias("sesXaid_sum_click_dur_sec"),
            pl.col("duration_second")
            .filter(pl.col("type") == 1)
            .sum()
            .fill_null(0)
            .alias("sesXaid_sum_cart_dur_sec"),
            pl.col("duration_second")
            .filter(pl.col("type") == 2)
            .sum()
            .fill_null(0)
            .alias("sesXaid_sum_order_dur_sec"),
            # total duration
            pl.col("duration_second").sum().fill_null(0).alias("sesXaid_sum_dur_sec"),
            # event type
            pl.col("type").n_unique().alias("sesXaid_type_dcount"),
            # sum of log_recency_score & type_weighted_log_recency_score
            pl.col("log_recency_score").sum().alias("sesXaid_log_recency_score"),
            pl.col("type_weighted_log_recency_score")
            .sum()
            .alias("sesXaid_type_weighted_log_recency_score"),
            # pl.col("log_duration_second").sum().alias("sesXaid_log_duration_second"),
            # pl.col("type_weighted_log_duration_second")
            # .sum()
            # .alias("sesXaid_type_weighted_log_duration_second"),
            pl.col("action_num_reverse_chrono")
            .min()
            .alias("sesXaid_action_num_reverse_chrono"),
            pl.col("type").last().alias("sesXaid_last_type_in_session"),
        ]
    )

    data_agg = data_agg.with_columns(
        [
            (pl.col("sesXaid_sec_from_last_event") / 60).alias(
                "sesXaid_mins_from_last_event"
            ),
            # (np.log2(1 + pl.col("sesXaid_log_duration_second"))).alias(
            #     "sesXaid_log_duration_second_log2p1"
            # ),
            # (np.log2(1 + pl.col("sesXaid_type_weighted_log_duration_second"))).alias(
            #     "sesXaid_type_weighted_log_duration_second_log2p1"
            # ),
        ],
    )

    data_agg = data_agg.with_columns(
        [
            (np.log1p(pl.col("sesXaid_mins_from_last_event"))).alias(
                "sesXaid_mins_from_last_event_log1p"
            ),
        ],
    )

    # fraction of event to all event in session
    data_agg = data_agg.with_columns(
        [
            # window event count session
            pl.col("sesXaid_click_count")
            .sum()
            .over("session")
            .alias("sesXaid_all_click_count"),
            pl.col("sesXaid_cart_count")
            .sum()
            .over("session")
            .alias("sesXaid_all_cart_count"),
            pl.col("sesXaid_order_count")
            .sum()
            .over("session")
            .alias("sesXaid_all_order_count"),
            # window duration in session
            pl.col("sesXaid_sum_dur_sec")
            .sum()
            .over("session")
            .alias("sesXaid_all_duration_second"),
            pl.col("sesXaid_sum_click_dur_sec")
            .sum()
            .over("session")
            .alias("sesXaid_all_sum_click_dur_sec"),
            pl.col("sesXaid_sum_cart_dur_sec")
            .sum()
            .over("session")
            .alias("sesXaid_all_sum_cart_dur_sec"),
            pl.col("sesXaid_sum_order_dur_sec")
            .sum()
            .over("session")
            .alias("sesXaid_all_sum_order_dur_sec"),
            # window log recency score
            pl.col("sesXaid_log_recency_score")
            .sum()
            .over("session")
            .alias("sesXaid_all_log_recency_score"),
            pl.col("sesXaid_type_weighted_log_recency_score")
            .sum()
            .over("session")
            .alias("sesXaid_all_type_weighted_log_recency_score"),
            # pl.col("sesXaid_log_duration_second")
            # .sum()
            # .over("session")
            # .alias("sesXaid_all_log_duration_second"),
            # pl.col("sesXaid_type_weighted_log_duration_second")
            # .sum()
            # .over("session")
            # .alias("sesXaid_all_type_weighted_log_duration_second"),
        ],
    )

    data_agg = data_agg.with_columns(
        [
            # frac compare to total event in particular session
            (pl.col("sesXaid_click_count") / pl.col("sesXaid_all_click_count"))
            .fill_nan(0)
            .alias("sesXaid_frac_click_all_click_count"),
            (pl.col("sesXaid_cart_count") / pl.col("sesXaid_all_cart_count"))
            .fill_nan(0)
            .alias("sesXaid_frac_cart_all_cart_count"),
            (pl.col("sesXaid_order_count") / pl.col("sesXaid_all_order_count"))
            .fill_nan(0)
            .alias("sesXaid_frac_order_all_order_count"),
            # frac compare to total event in particular sessionXaid
            (pl.col("sesXaid_click_count") / pl.col("sesXaid_events_count"))
            .fill_nan(0)
            .alias("sesXaid_frac_click_all_events_count"),
            (pl.col("sesXaid_cart_count") / pl.col("sesXaid_events_count"))
            .fill_nan(0)
            .alias("sesXaid_frac_cart_all_events_count"),
            (pl.col("sesXaid_order_count") / pl.col("sesXaid_events_count"))
            .fill_nan(0)
            .alias("sesXaid_frac_order_all_events_count"),
            # frac to all duration event
            (
                pl.col("sesXaid_sum_click_dur_sec")
                / pl.col("sesXaid_all_duration_second")
            )
            .fill_nan(0)
            .alias("sesXaid_frac_dur_click_all_dur_sec"),
            (pl.col("sesXaid_sum_cart_dur_sec") / pl.col("sesXaid_all_duration_second"))
            .fill_nan(0)
            .alias("sesXaid_frac_dur_cart_all_dur_sec"),
            # (
            #     pl.col("sesXaid_sum_order_dur_sec")
            #     / pl.col("sesXaid_all_duration_second")
            # )
            # .fill_nan(0)
            # .alias("sesXaid_frac_dur_order_all_dur_sec"),
            # frac to total duration specific event
            (
                pl.col("sesXaid_sum_click_dur_sec")
                / pl.col("sesXaid_all_sum_click_dur_sec")
            )
            .fill_nan(0)
            .alias("sesXaid_frac_dur_click_all_dur_sec"),
            (
                pl.col("sesXaid_sum_cart_dur_sec")
                / pl.col("sesXaid_all_sum_cart_dur_sec")
            )
            .fill_nan(0)
            .alias("sesXaid_frac_dur_cart_all_dur_sec"),
            # (
            #     pl.col("sesXaid_sum_order_dur_sec")
            #     / pl.col("sesXaid_all_sum_order_dur_sec")
            # )
            # .fill_nan(0)
            # .alias("sesXaid_frac_dur_order_all_dur_sec"),
            # frac to total log_recency_score
            (
                pl.col("sesXaid_log_recency_score")
                / pl.col("sesXaid_all_log_recency_score")
            )
            .fill_nan(0)
            .alias("sesXaid_frac_log_recency_to_all"),
            (
                pl.col("sesXaid_type_weighted_log_recency_score")
                / pl.col("sesXaid_all_type_weighted_log_recency_score")
            )
            .fill_nan(0)
            .alias("sesXaid_frac_type_weighted_log_recency_to_all"),
            # (
            #     pl.col("sesXaid_log_duration_second")
            #     / pl.col("sesXaid_all_log_duration_second")
            # )
            # .fill_nan(0)
            # .alias("sesXaid_frac_log_duration_sec_to_all"),
            # (
            #     pl.col("sesXaid_type_weighted_log_duration_second")
            #     / pl.col("sesXaid_all_type_weighted_log_duration_second")
            # )
            # .fill_nan(0)
            # .alias("sesXaid_frac_type_weighted_log_duration_sec_to_all"),
        ],
    )

    # drop cols
    data_agg = data_agg.drop(
        columns=[
            "sesXaid_sec_from_last_event",
            # "sesXaid_log_duration_second",
            # "sesXaid_type_weighted_log_duration_second",
            "sesXaid_all_click_count",
            "sesXaid_all_cart_count",
            "sesXaid_all_order_count",
            "sesXaid_all_duration_second",
            "sesXaid_all_sum_click_dur_sec",
            "sesXaid_all_sum_cart_dur_sec",
            "sesXaid_all_sum_order_dur_sec",
            "sesXaid_all_log_recency_score",
            "sesXaid_all_type_weighted_log_recency_score",
            # "sesXaid_all_log_duration_second",
            # "sesXaid_all_type_weighted_log_duration_second",
            "sesXaid_events_count",
            "sesXaid_order_count",
            "sesXaid_sum_cart_dur_sec",
        ]
    )

    return data_agg


def make_session_item_features(
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
    for ix in tqdm(range(n)):
        # logging.info(f"chunk {ix}: read input")
        filepath = f"{input_path}/{name}_{ix}.parquet"
        df = pl.read_parquet(filepath)

        logging.info(f"start creating sessionXaid features")
        df_output = gen_session_item_features(data=df)
        df_output = freemem(df_output)

        filepath = output_path / f"{name}_{ix}_session_item_feas.parquet"
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
        output_path = get_processed_training_train_sess_item_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_item_features(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_sess_item_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_item_features(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_sess_item_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_item_features(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_sess_item_features_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_item_features(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
