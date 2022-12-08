import click
import polars as pl
import numpy as np
from tqdm import tqdm
import gc
from src.utils.constants import (
    CFG,
    get_processed_training_train_candidates_dir,  # candidates dir
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
    get_processed_training_train_sess_features_dir,  # session features dir
    get_processed_training_test_sess_features_dir,
    get_processed_scoring_train_sess_features_dir,
    get_processed_scoring_test_sess_features_dir,
    get_processed_training_train_sess_item_features_dir,  # sessionXaid features dir
    get_processed_training_test_sess_item_features_dir,
    get_processed_scoring_train_sess_item_features_dir,
    get_processed_scoring_test_sess_item_features_dir,
    get_processed_training_train_item_features_dir,  # item features dir
    get_processed_training_test_item_features_dir,
    get_processed_scoring_train_item_features_dir,
    get_processed_scoring_test_item_features_dir,
    get_processed_training_train_item_hour_features_dir,  # itemXhour features dir
    get_processed_training_test_item_hour_features_dir,
    get_processed_scoring_train_item_hour_features_dir,
    get_processed_scoring_test_item_hour_features_dir,
    get_processed_training_train_item_weekday_features_dir,  # itemXweekday features dir
    get_processed_training_test_item_weekday_features_dir,
    get_processed_scoring_train_item_weekday_features_dir,
    get_processed_scoring_test_item_weekday_features_dir,
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_training_test_dataset_dir,
    get_processed_scoring_train_dataset_dir,
    get_processed_scoring_test_dataset_dir,
)

from src.utils.logger import get_logger

logging = get_logger()


def fcombine_features(mode: str, event: str, ix: int):
    candidate_path = ""
    session_fea_path = ""
    sessionXaid_fea_path = ""
    item_fea_path = ""
    itemXhour_fea_path = ""
    itemXweekday_fea_path = ""
    output_path = ""
    name = ""

    if mode == "training_train":
        candidate_path = get_processed_training_train_candidates_dir()
        session_fea_path = get_processed_training_train_sess_features_dir()
        sessionXaid_fea_path = get_processed_training_train_sess_item_features_dir()
        item_fea_path = get_processed_training_train_item_features_dir()
        itemXhour_fea_path = get_processed_training_train_item_hour_features_dir()
        itemXweekday_fea_path = get_processed_training_train_item_weekday_features_dir()
        output_path = get_processed_training_train_dataset_dir()
        name = "train"

    elif mode == "training_test":
        candidate_path = get_processed_training_test_candidates_dir()
        session_fea_path = get_processed_training_test_sess_features_dir()
        sessionXaid_fea_path = get_processed_training_test_sess_item_features_dir()
        item_fea_path = get_processed_training_test_item_features_dir()
        itemXhour_fea_path = get_processed_training_test_item_hour_features_dir()
        itemXweekday_fea_path = get_processed_training_test_item_weekday_features_dir()
        output_path = get_processed_training_test_dataset_dir()
        name = "test"

    elif mode == "scoring_train":
        candidate_path = get_processed_scoring_train_candidates_dir()
        session_fea_path = get_processed_scoring_train_sess_features_dir()
        sessionXaid_fea_path = get_processed_scoring_train_sess_item_features_dir()
        item_fea_path = get_processed_scoring_train_item_features_dir()
        itemXhour_fea_path = get_processed_scoring_train_item_hour_features_dir()
        itemXweekday_fea_path = get_processed_scoring_train_item_weekday_features_dir()
        output_path = get_processed_scoring_train_dataset_dir()
        name = "train"

    elif mode == "scoring_test":
        candidate_path = get_processed_scoring_test_candidates_dir()
        session_fea_path = get_processed_scoring_test_sess_features_dir()
        sessionXaid_fea_path = get_processed_scoring_test_sess_item_features_dir()
        item_fea_path = get_processed_scoring_test_item_features_dir()
        itemXhour_fea_path = get_processed_scoring_test_item_hour_features_dir()
        itemXweekday_fea_path = get_processed_scoring_test_item_weekday_features_dir()
        output_path = get_processed_scoring_test_dataset_dir()
        name = "test"

    logging.info(f"read candidate from: {candidate_path}")
    logging.info(f"will save chunks data to: {output_path}")

    c_path = f"{candidate_path}/{name}_{ix}_{event}_rows.parquet"
    sfea_path = f"{session_fea_path}/{name}_{ix}_session_feas.parquet"
    sfeaXaid_path = f"{sessionXaid_fea_path}/{name}_{ix}_session_item_feas.parquet"
    item_path = f"{item_fea_path}/{name}_{ix}_item_feas.parquet"
    itemXhour_path = f"{itemXhour_fea_path}/{name}_{ix}_item_hour_feas.parquet"
    itemXweekday_path = f"{itemXweekday_fea_path}/{name}_{ix}_item_weekday_feas.parquet"

    cand_df = pl.read_parquet(c_path)
    # make sure to cast session id & candidate_aid to int32
    cand_df = cand_df.with_columns(
        [
            pl.col("session").cast(pl.Int32).alias("session"),
            pl.col("candidate_aid").cast(pl.Int32).alias("candidate_aid"),
        ]
    )
    logging.info(f"read candidates with shape {cand_df.shape}")

    # read session features
    ses_agg = pl.read_parquet(sfea_path)
    logging.info(f"read session features with shape {ses_agg.shape}")
    cand_df = cand_df.join(ses_agg, how="left", on=["session"])
    logging.info(f"joined with session features! shape {cand_df.shape}")

    del ses_agg
    gc.collect()

    # read session-item features
    ses_aid_agg = pl.read_parquet(sfeaXaid_path)
    logging.info(f"read sessionXaid features with shape {ses_aid_agg.shape}")
    cand_df = cand_df.join(
        ses_aid_agg,
        how="left",
        left_on=["session", "candidate_aid"],
        right_on=["session", "aid"],
    )
    logging.info(f"joined with sessionXaid features! shape {cand_df.shape}")

    cand_df = cand_df.with_columns(
        [
            pl.col("sesXaid_type_weighted_log_recency_score")
            .fill_null(0)
            .alias("sesXaid_type_weighted_log_recency_score"),
            pl.col("sesXaid_mins_from_last_event")
            .fill_null(9999.0)
            .alias("sesXaid_mins_from_last_event"),
        ]
    )
    cand_df = cand_df.fill_null(-99)

    del ses_aid_agg
    gc.collect()

    # read item features
    item_agg = pl.read_parquet(item_path)
    logging.info(f"read item features with shape {item_agg.shape}")
    cand_df = cand_df.join(
        item_agg,
        how="left",
        left_on=["candidate_aid"],
        right_on=["aid"],
    )
    logging.info(f"joined with item features! shape {cand_df.shape}")

    del item_agg
    gc.collect()

    cand_df = cand_df.with_columns(
        [
            np.abs(pl.col("sess_hour") - pl.col("item_avg_hour_click"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sessXitem_abs_diff_avg_hour_click"),
            np.abs(pl.col("sess_hour") - pl.col("item_avg_hour_cart"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sessXitem_abs_diff_avg_hour_cart"),
            np.abs(pl.col("sess_hour") - pl.col("item_avg_hour_order"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sessXitem_abs_diff_avg_hour_order"),
            np.abs(pl.col("sess_weekday") - pl.col("item_avg_weekday_click"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sessXitem_abs_diff_avg_weekday_click"),
            np.abs(pl.col("sess_weekday") - pl.col("item_avg_weekday_cart"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sessXitem_abs_diff_avg_weekday_cart"),
            np.abs(pl.col("sess_weekday") - pl.col("item_avg_weekday_order"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sessXitem_abs_diff_avg_weekday_order"),
        ]
    )
    cand_df = cand_df.fill_null(-99)

    # read item-hour features
    item_hour_agg = pl.read_parquet(itemXhour_path)
    logging.info(f"read item-hour features with shape {item_hour_agg.shape}")
    cand_df = cand_df.join(
        item_hour_agg,
        how="left",
        left_on=["candidate_aid", "sess_hour"],
        right_on=["aid", "hour"],
    ).fill_null(-99)
    logging.info(f"joined with item-hour features! shape {cand_df.shape}")

    del item_hour_agg
    gc.collect()

    # read item-weekday features
    item_weekday_agg = pl.read_parquet(itemXweekday_path)
    logging.info(f"read item-weekday features with shape {item_weekday_agg.shape}")
    cand_df = cand_df.join(
        item_weekday_agg,
        how="left",
        left_on=["candidate_aid", "sess_weekday"],
        right_on=["aid", "weekday"],
    ).fill_null(-99)
    logging.info(f"joined with item-weekday features! shape {cand_df.shape}")

    del item_weekday_agg
    gc.collect()

    filepath = f"{output_path}/{name}_{ix}_{event}_combined.parquet"
    logging.info(f"save chunk to: {filepath}")
    cand_df.write_parquet(f"{filepath}")
    logging.info(f"output df shape {cand_df.shape}")

    del cand_df
    gc.collect()


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    name = "train"
    if mode == "training_train":
        name = "train"

    elif mode == "training_test":
        name = "test"

    elif mode == "scoring_train":
        name = "train"

    elif mode == "scoring_test":
        name = "test"

    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(n)):
        for event in ["clicks", "carts", "orders"]:
            logging.info(f"start combining features")
            fcombine_features(mode=mode, event=event, ix=ix)


if __name__ == "__main__":
    main()
