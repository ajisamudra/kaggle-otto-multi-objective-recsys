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
    get_processed_training_train_sess_features_dir,
    get_processed_training_test_sess_features_dir,
    get_processed_scoring_train_sess_features_dir,
    get_processed_scoring_test_sess_features_dir,
)
from src.utils.date_function import get_hour_from_ts, get_weekday_from_ts
from src.features.preprocess_events import preprocess_events
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def gen_user_features(data: pl.DataFrame):
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

    data = data.with_columns(
        [
            pl.when(pl.col("type") == 0).then(1).otherwise(None).alias("dummy_click"),
            pl.when(pl.col("type") == 1).then(1).otherwise(None).alias("dummy_cart"),
            pl.when(pl.col("type") == 2).then(1).otherwise(None).alias("dummy_order"),
        ],
    )

    data = data.with_columns(
        [
            (pl.col("dummy_click") * pl.col("hour")).alias("hour_click"),
            (pl.col("dummy_cart") * pl.col("hour")).alias("hour_cart"),
            (pl.col("dummy_order") * pl.col("hour")).alias("hour_order"),
            (pl.col("dummy_click") * pl.col("weekday")).alias("weekday_click"),
            (pl.col("dummy_cart") * pl.col("weekday")).alias("weekday_cart"),
            (pl.col("dummy_order") * pl.col("weekday")).alias("weekday_order"),
        ],
    )

    # agg per session
    data_agg = data.groupby("session").agg(
        [
            pl.col("aid").count().alias("sess_all_events_count"),
            pl.col("oneday_session").sum().alias("sess_num_real_session"),
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
            # avg duration per session
            pl.col("duration_second")
            .mean()
            .fill_null(0)
            .alias("sess_avg_all_events_dur_sec"),
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
            # avg hour & weekday of click/cart/order
            pl.col("hour_click").mean().alias("sess_avg_hour_click"),
            pl.col("hour_cart").mean().alias("sess_avg_hour_cart"),
            pl.col("hour_order").mean().alias("sess_avg_hour_order"),
            pl.col("weekday_click").mean().alias("sess_avg_weekday_click"),
            pl.col("weekday_cart").mean().alias("sess_avg_weekday_cart"),
            pl.col("weekday_order").mean().alias("sess_avg_weekday_order"),
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
            (pl.col("sess_num_real_session") + 1).alias("sess_num_real_session"),
        ],
    )

    data_agg = data_agg.with_columns(
        [
            # conversion rate
            (pl.col("sess_cart_count") / pl.col("sess_click_count"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_click_to_cart_cvr"),
            (pl.col("sess_order_count") / pl.col("sess_cart_count"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_cart_to_order_cvr"),
            (pl.col("sess_carted_aid_dcount") / pl.col("sess_clicked_aid_dcount"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_clicked_to_carted_aid_cvr"),
            (pl.col("sess_ordered_aid_dcount") / pl.col("sess_carted_aid_dcount"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_carted_to_ordered_aid_cvr"),
            # click / cart / order event proportion
            (pl.col("sess_click_count") / pl.col("sess_all_events_count"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_frac_click_to_all_events"),
            (pl.col("sess_cart_count") / pl.col("sess_all_events_count"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_frac_cart_to_all_events"),
            (pl.col("sess_order_count") / pl.col("sess_all_events_count"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_frac_order_to_all_events"),
            (pl.col("sess_clicked_aid_dcount") / pl.col("sess_aid_dcount"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_frac_clicked_aid_to_all_aid"),
            (pl.col("sess_carted_aid_dcount") / pl.col("sess_aid_dcount"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_frac_carted_aid_to_all_aid"),
            (pl.col("sess_ordered_aid_dcount") / pl.col("sess_aid_dcount"))
            .fill_nan(0)
            .fill_null(0)
            .alias("sess_frac_ordered_aid_to_all_aid"),
            # abs diff sess avg hour/weekday click/cart/order with current session hour/weekday
            np.abs(pl.col("sess_hour") - pl.col("sess_all_events_count"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sess_abs_diff_avg_hour_click"),
            np.abs(pl.col("sess_hour") - pl.col("sess_avg_hour_cart"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sess_abs_diff_avg_hour_cart"),
            np.abs(pl.col("sess_hour") - pl.col("sess_avg_hour_order"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sess_abs_diff_avg_hour_order"),
            np.abs(pl.col("sess_weekday") - pl.col("sess_avg_weekday_click"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sess_abs_diff_avg_weekday_click"),
            np.abs(pl.col("sess_weekday") - pl.col("sess_avg_weekday_cart"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sess_abs_diff_avg_weekday_cart"),
            np.abs(pl.col("sess_weekday") - pl.col("sess_avg_weekday_order"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sess_abs_diff_avg_weekday_order"),
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

        logging.info(f"start creating session features")
        df_output = gen_user_features(data=df)
        df_output = freemem(df_output)

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
