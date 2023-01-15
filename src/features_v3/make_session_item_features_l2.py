import click
import polars as pl
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_train_sess_item_features_dir,  # sessXaid features l1
    get_processed_training_test_sess_item_features_dir,
    get_processed_scoring_train_sess_item_features_dir,
    get_processed_scoring_test_sess_item_features_dir,
    get_processed_training_train_sess_features_dir,  # session features
    get_processed_training_test_sess_features_dir,
    get_processed_scoring_train_sess_features_dir,
    get_processed_scoring_test_sess_features_dir,
    get_processed_training_train_candidates_dir,  # candidate rows
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
)
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def gen_session_item_features_l2(
    sess_fea_path: Path,
    candidate_path: Path,
    output_path: Path,
    name: str,
    mode: str,
    ix: int,
):
    """
    df input
    session | type | ts | aid
    123 | 0 | 12313 | AID1
    123 | 1 | 12314 | AID1
    123 | 2 | 12345 | AID1
    """

    filepath = f"{sess_fea_path}/{name}_{ix}_session_feas.parquet"
    sess_df = pl.read_parquet(filepath)
    # filter columns
    sess_df = sess_df.select(
        [
            "session",
            "sess_aid_dcount",
            "sess_last_type_in_session",
            "sess_hour",
            "sess_weekday",
        ]
    )

    def binning_aid_dcount(x):
        if x <= 4:
            return str(x)
        elif (x > 4) and (x <= 8):
            return "5_8"
        elif (x > 8) and (x <= 12):
            return "9_12"
        elif (x > 12) and (x <= 15):
            return "13_15"
        elif (x > 15) and (x <= 20):
            return "16_20"
        elif (x > 20) and (x <= 30):
            return "21_30"
        else:
            return ">30"

    sess_df = sess_df.with_columns(
        pl.col("sess_aid_dcount")
        .apply(lambda x: binning_aid_dcount(x))
        .alias("sess_binned_aid_dcount")
    )

    # drop cols
    sess_df = sess_df.drop(
        columns=[
            "sess_aid_dcount",
        ]
    )

    for event in ["clicks", "carts", "orders"]:

        if (mode == "training_train") & (event == "clicks") & (ix > 6):
            logging.info("click ix > 6 continue")
            continue

        # read cand_id
        filepath = f"{candidate_path}/{name}_{ix}_{event}_rows.parquet"
        cand_df = pl.read_parquet(filepath)
        # filter only session | candidate_aid | rank_covist
        cand_df = cand_df.select(["session", "candidate_aid", "rank_combined"])
        cand_df = freemem(cand_df)

        # left join
        cand_df = cand_df.join(
            sess_df,
            how="left",
            left_on=["session"],
            right_on=["session"],
        )

        # make combination categorical features
        cand_df = cand_df.with_columns(
            [
                (
                    pl.col("rank_combined")
                    + pl.lit("_")
                    + pl.col("sess_last_type_in_session")
                ).alias("combined_rank_combined_sess_last_type_in_session"),
                (pl.col("rank_combined") + pl.lit("_") + pl.col("sess_hour")).alias(
                    "combined_rank_combined_sess_hour"
                ),
                (pl.col("rank_combined") + pl.lit("_") + pl.col("sess_weekday")).alias(
                    "combined_rank_combined_sess_weekday"
                ),
                (
                    pl.col("rank_combined")
                    + pl.lit("_")
                    + pl.col("sess_weekday")
                    + pl.lit("_")
                    + pl.col("sess_hour")
                ).alias("combined_rank_combined_sess_weekday_hour"),
                (
                    pl.col("rank_combined")
                    + pl.lit("_")
                    + pl.col("sess_binned_aid_dcount")
                ).alias("combined_rank_combined_sess_binned_aid_dcount"),
            ]
        )

        cand_df = freemem(cand_df)

        # drop cols
        cand_df = cand_df.drop(
            columns=[
                "rank_combined",
                "sess_last_type_in_session",
                "sess_hour",
                "sess_weekday",
                "sess_binned_aid_dcount",
            ]
        )

        filepath = output_path / f"{name}_{ix}_session_item_feas_l2.parquet"
        logging.info(f"save chunk to: {filepath}")
        cand_df.write_parquet(f"{filepath}")
        logging.info(f"output df shape {cand_df.shape}")

        del cand_df
        gc.collect()


def make_session_item_features_l2(
    name: str, mode: str, sess_fea_path: Path, output_path: Path, candidate_path: Path
):

    if mode == "training_train":
        n = CFG.N_train
    elif mode == "training_test":
        n = CFG.N_local_test
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"start creating sessionXaid features")
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(n)):
        gen_session_item_features_l2(
            sess_fea_path=sess_fea_path,
            candidate_path=candidate_path,
            output_path=output_path,
            name=name,
            mode=mode,
            ix=ix,
        )


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    if mode == "training_train":
        sess_fea_path = get_processed_training_train_sess_features_dir()
        output_path = get_processed_training_train_sess_item_features_dir()
        candidate_path = get_processed_training_train_candidates_dir()
        logging.info(f"read input data from: {sess_fea_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_item_features_l2(
            name="train",
            mode=mode,
            sess_fea_path=sess_fea_path,
            output_path=output_path,
            candidate_path=candidate_path,
        )

    elif mode == "training_test":
        sess_fea_path = get_processed_training_test_sess_features_dir()
        output_path = get_processed_training_test_sess_item_features_dir()
        candidate_path = get_processed_training_test_candidates_dir()
        logging.info(f"read input data from: {sess_fea_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_item_features_l2(
            name="test",
            mode=mode,
            sess_fea_path=sess_fea_path,
            output_path=output_path,
            candidate_path=candidate_path,
        )

    elif mode == "scoring_train":
        sess_fea_path = get_processed_scoring_train_sess_features_dir()
        output_path = get_processed_scoring_train_sess_item_features_dir()
        candidate_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"read input data from: {sess_fea_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_item_features_l2(
            name="train",
            mode=mode,
            sess_fea_path=sess_fea_path,
            output_path=output_path,
            candidate_path=candidate_path,
        )

    elif mode == "scoring_test":
        sess_fea_path = get_processed_scoring_test_sess_features_dir()
        output_path = get_processed_scoring_test_sess_item_features_dir()
        candidate_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"read input data from: {sess_fea_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_item_features_l2(
            name="test",
            mode=mode,
            sess_fea_path=sess_fea_path,
            output_path=output_path,
            candidate_path=candidate_path,
        )


if __name__ == "__main__":
    main()
