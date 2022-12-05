import click
import polars as pl
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
    output_path = ""
    name = ""

    if mode == "training_train":
        candidate_path = get_processed_training_train_candidates_dir()
        session_fea_path = get_processed_training_train_sess_features_dir()
        sessionXaid_fea_path = get_processed_training_train_sess_item_features_dir()
        output_path = get_processed_training_train_dataset_dir()
        name = "train"

    elif mode == "training_test":
        candidate_path = get_processed_training_test_candidates_dir()
        session_fea_path = get_processed_training_test_sess_features_dir()
        sessionXaid_fea_path = get_processed_training_test_sess_item_features_dir()
        output_path = get_processed_training_test_dataset_dir()
        name = "test"

    elif mode == "scoring_train":
        candidate_path = get_processed_scoring_train_candidates_dir()
        session_fea_path = get_processed_scoring_train_sess_features_dir()
        sessionXaid_fea_path = get_processed_scoring_train_sess_item_features_dir()
        output_path = get_processed_scoring_train_dataset_dir()
        name = "train"

    elif mode == "scoring_test":
        candidate_path = get_processed_scoring_test_candidates_dir()
        session_fea_path = get_processed_scoring_test_sess_features_dir()
        sessionXaid_fea_path = get_processed_scoring_test_sess_item_features_dir()
        output_path = get_processed_scoring_test_dataset_dir()
        name = "test"

    logging.info(f"read candidate from: {candidate_path}")
    logging.info(f"will save chunks data to: {output_path}")

    c_path = f"{candidate_path}/{name}_{ix}_{event}_rows.parquet"
    sfea_path = f"{session_fea_path}/{name}_{ix}_session_feas.parquet"
    sfeaXaid_path = f"{sessionXaid_fea_path}/{name}_{ix}_session_item_feas.parquet"

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

    # read session features
    ses_aid_agg = pl.read_parquet(sfeaXaid_path)
    logging.info(f"read sessionXaid features with shape {ses_aid_agg.shape}")
    cand_df = cand_df.join(
        ses_aid_agg,
        how="left",
        left_on=["session", "candidate_aid"],
        right_on=["session", "aid"],
    ).fill_null(-99)
    logging.info(f"joined with sessionXaid features! shape {cand_df.shape}")

    del ses_aid_agg
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
