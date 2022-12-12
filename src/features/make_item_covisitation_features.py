import click
import polars as pl
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_train_session_representation_items_dir,  # session representations dir
    get_processed_training_test_session_representation_items_dir,
    get_processed_scoring_train_session_representation_items_dir,
    get_processed_scoring_test_session_representation_items_dir,
    get_processed_training_train_item_covisitation_features_dir,  # output dir
    get_processed_training_test_item_covisitation_features_dir,
    get_processed_scoring_train_item_covisitation_features_dir,
    get_processed_scoring_test_item_covisitation_features_dir,
)
from src.utils.data import (
    get_top15_covisitation_buys_df,
    get_top15_covisitation_buy2buy_df,
    get_top20_covisitation_click_df,
)
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def gen_item_covisitation_features(
    name: str,
    ix: int,
    ses_representation_path: Path,
    output_path: Path,
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

    for event in ["clicks", "carts", "orders"]:
        logging.info(f"read session representation aids for event {event.upper()}")
        # read session representation
        filepath = f"{ses_representation_path}/{name}_{ix}_{event}_session_representation_items.parquet"
        cand_df = pl.read_parquet(filepath)

        # GET CLICK COVISITATION WEIGHT
        logging.info("join with click covisitation weight")
        cand_df = cand_df.join(
            top_20_clicks,
            how="left",
            left_on=["last_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select column
        cand_df = cand_df.select(
            [
                "session",
                "candidate_aid",
                "last_event_in_session_aid",
                "max_recency_event_in_session_aid",
                "max_weighted_recency_event_in_session_aid",
                "max_log_duration_event_in_session_aid",
                "max_weighted_log_duration_event_in_session_aid",
                "wgt",
            ]
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
        ]

        cand_df = cand_df.join(
            top_20_clicks,
            how="left",
            left_on=["max_recency_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
        ]

        cand_df = cand_df.join(
            top_20_clicks,
            how="left",
            left_on=["max_weighted_recency_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
        ]

        cand_df = cand_df.join(
            top_20_clicks,
            how="left",
            left_on=["max_log_duration_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
        ]

        cand_df = cand_df.join(
            top_20_clicks,
            how="left",
            left_on=["max_weighted_log_duration_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
        ]

        # BUYS COVISITATION
        logging.info("join with buys covisitation weight")
        cand_df = cand_df.join(
            top_15_buys,
            how="left",
            left_on=["last_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
        ]
        cand_df = cand_df.join(
            top_15_buys,
            how="left",
            left_on=["max_recency_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
        ]
        cand_df = cand_df.join(
            top_15_buys,
            how="left",
            left_on=["max_weighted_recency_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_with_max_weighted_recency_event_in_session_aid",
        ]

        cand_df = cand_df.join(
            top_15_buys,
            how="left",
            left_on=["max_log_duration_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_max_log_duration_event_in_session_aid",
        ]

        cand_df = cand_df.join(
            top_15_buys,
            how="left",
            left_on=["max_weighted_log_duration_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_max_log_duration_event_in_session_aid",
            "buys_weight_with_max_weighted_log_duration_event_in_session_aid",
        ]

        # BUY2BUY COVISITATION
        logging.info("join with buy2buy covisitation weight")
        cand_df = cand_df.join(
            top_15_buy2buy,
            how="left",
            left_on=["last_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_max_log_duration_event_in_session_aid",
            "buys_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buy2buy_weight_last_event_in_session_aid",
        ]
        cand_df = cand_df.join(
            top_15_buy2buy,
            how="left",
            left_on=["max_recency_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_max_log_duration_event_in_session_aid",
            "buys_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buy2buy_weight_last_event_in_session_aid",
            "buy2buy_weight_max_recency_event_in_session_aid",
        ]
        cand_df = cand_df.join(
            top_15_buy2buy,
            how="left",
            left_on=["max_weighted_recency_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_max_log_duration_event_in_session_aid",
            "buys_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buy2buy_weight_last_event_in_session_aid",
            "buy2buy_weight_max_recency_event_in_session_aid",
            "buy2buy_weight_max_weighted_recency_event_in_session_aid",
        ]

        cand_df = cand_df.join(
            top_15_buy2buy,
            how="left",
            left_on=["max_log_duration_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_max_log_duration_event_in_session_aid",
            "buys_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buy2buy_weight_last_event_in_session_aid",
            "buy2buy_weight_max_recency_event_in_session_aid",
            "buy2buy_weight_max_weighted_recency_event_in_session_aid",
            "buy2buy_weight_max_log_duration_event_in_session_aid",
        ]

        cand_df = cand_df.join(
            top_15_buy2buy,
            how="left",
            left_on=["max_weighted_log_duration_event_in_session_aid", "candidate_aid"],
            right_on=["aid_x", "aid_y"],
        )
        # select rename columns
        cand_df.columns = [
            "session",
            "candidate_aid",
            "last_event_in_session_aid",
            "max_recency_event_in_session_aid",
            "max_weighted_recency_event_in_session_aid",
            "max_log_duration_event_in_session_aid",
            "max_weighted_log_duration_event_in_session_aid",
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "click_weight_with_max_log_duration_event_in_session_aid",
            "click_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_max_log_duration_event_in_session_aid",
            "buys_weight_with_max_weighted_log_duration_event_in_session_aid",
            "buy2buy_weight_last_event_in_session_aid",
            "buy2buy_weight_max_recency_event_in_session_aid",
            "buy2buy_weight_max_weighted_recency_event_in_session_aid",
            "buy2buy_weight_max_log_duration_event_in_session_aid",
            "buy2buy_weight_max_weighted_log_duration_event_in_session_aid",
        ]

        cand_df = cand_df.select(
            [
                "session",
                "candidate_aid",
                "click_weight_with_last_event_in_session_aid",
                "click_weight_with_max_recency_event_in_session_aid",
                "click_weight_with_max_weighted_recency_event_in_session_aid",
                "click_weight_with_max_log_duration_event_in_session_aid",
                "click_weight_with_max_weighted_log_duration_event_in_session_aid",
                "buys_weight_with_last_event_in_session_aid",
                "buys_weight_with_max_recency_event_in_session_aid",
                "buys_weight_with_max_weighted_recency_event_in_session_aid",
                "buys_weight_with_max_log_duration_event_in_session_aid",
                "buys_weight_with_max_weighted_log_duration_event_in_session_aid",
                "buy2buy_weight_last_event_in_session_aid",
                "buy2buy_weight_max_recency_event_in_session_aid",
                "buy2buy_weight_max_weighted_recency_event_in_session_aid",
                "buy2buy_weight_max_log_duration_event_in_session_aid",
                "buy2buy_weight_max_weighted_log_duration_event_in_session_aid",
            ]
        )

        # save item covisitation features
        filepath = output_path / f"{name}_{ix}_{event}_item_covisitation_feas.parquet"
        logging.info(f"save chunk to: {filepath}")
        cand_df = freemem(cand_df)
        cand_df.write_parquet(f"{filepath}")
        logging.info(f"output df shape {cand_df.shape}")

        del cand_df
        gc.collect()


def make_item_covisitation_features(
    mode: str,
    name: str,
    ses_representation_path: Path,
    output_path: Path,
    istart: int,
    iend: int,
):

    if name == "train":
        n = CFG.N_train
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
    for ix in tqdm(range(istart, iend)):
        logging.info(f"start creating item covisitation features")
        gen_item_covisitation_features(
            name=name,
            ix=ix,
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            top_15_buys=top_15_buys,
            top_15_buy2buy=top_15_buy2buy,
            top_20_clicks=top_20_clicks,
        )


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
@click.option(
    "--istart",
    default=0,
    help="index start",
)
@click.option(
    "--iend",
    default=10,
    help="index end",
)
def main(mode: str, istart: int, iend: int):
    if mode == "training_train":
        ses_representation_path = (
            get_processed_training_train_session_representation_items_dir()
        )
        output_path = get_processed_training_train_item_covisitation_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_item_covisitation_features(
            mode=mode,
            name="train",
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            istart=istart,
            iend=iend,
        )

    elif mode == "training_test":
        ses_representation_path = (
            get_processed_training_test_session_representation_items_dir()
        )
        output_path = get_processed_training_test_item_covisitation_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_item_covisitation_features(
            mode=mode,
            name="test",
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            istart=istart,
            iend=iend,
        )

    elif mode == "scoring_train":
        ses_representation_path = (
            get_processed_scoring_train_session_representation_items_dir()
        )
        output_path = get_processed_scoring_train_item_covisitation_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_item_covisitation_features(
            mode=mode,
            name="train",
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            istart=istart,
            iend=iend,
        )

    elif mode == "scoring_test":
        ses_representation_path = (
            get_processed_scoring_test_session_representation_items_dir()
        )
        output_path = get_processed_scoring_test_item_covisitation_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_item_covisitation_features(
            mode=mode,
            name="test",
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            istart=istart,
            iend=iend,
        )


if __name__ == "__main__":
    main()
