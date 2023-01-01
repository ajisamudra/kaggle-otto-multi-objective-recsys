import click
import polars as pl
import pandas as pd
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
    get_top15_covisitation_buys_dict,
    get_top15_covisitation_buy2buy_dict,
    get_top20_covisitation_click_dict,
)
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def look_into_dict(
    candidate_list: list, top_20_clicks: dict, top_15_buys: dict, top_15_buy2buy: dict
):
    # FIND CLICK COVISIT WEIGHT
    click_wgts = []
    for candidate in candidate_list:
        click_wgts.append(top_20_clicks.get(candidate, None))

    # FIND CARTORDER COVISIT WEIGHT
    cart_order_wgts = []
    for candidate in candidate_list:
        cart_order_wgts.append(top_15_buys.get(candidate, None))

    # FIND BUY2BUY COVISIT WEIGHT
    buy2buy_wgts = []
    for candidate in candidate_list:
        buy2buy_wgts.append(top_15_buy2buy.get(candidate, None))

    return click_wgts, cart_order_wgts, buy2buy_wgts


def gen_item_covisitation_features(
    name: str,
    ix: int,
    ses_representation_path: Path,
    output_path: Path,
    top_15_buys: dict,
    top_15_buy2buy: dict,
    top_20_clicks: dict,
):
    """
    session representation aids
    session | candidate_aid | last_event_in_session_aid | last_event_in_session_aid
    123 | AID1 | AID1 | AID4
    123 | AID2 | AID1 | AID4
    123 | AID3 | AID1 | AID4
    """

    for event in ["clicks", "carts", "orders"]:
        logging.info(f"read session representation aids for event {event.upper()}")
        # read session representation
        filepath = f"{ses_representation_path}/{name}_{ix}_{event}_session_representation_items.parquet"
        cand_df = pd.read_parquet(filepath)
        # cand_df_pd = cand_df.to_pandas()

        # comparing candidate_aid with last_event_in_session_aid
        logging.info("compare last_event with candidate")
        cand_df["pair_list"] = (
            cand_df["last_event_in_session_aid"].astype(str)
            + "_"
            + cand_df["candidate_aid"].astype(str)
        )
        candidate_list = list(cand_df["pair_list"].values)
        (
            click_weights_last_event,
            buys_weights_last_event,
            buy2buy_weights_last_event,
        ) = look_into_dict(
            candidate_list=candidate_list,
            top_20_clicks=top_20_clicks,
            top_15_buy2buy=top_15_buy2buy,
            top_15_buys=top_15_buys,
        )

        # comparing candidate_aid with last_event_in_session_aid
        logging.info("compare max_recency_event_in_session_aid with candidate")
        cand_df["pair_list"] = (
            cand_df["max_recency_event_in_session_aid"].astype(str)
            + "_"
            + cand_df["candidate_aid"].astype(str)
        )
        candidate_list = list(cand_df["pair_list"].values)
        (
            click_weights_max_recency_event,
            buys_weights_max_recency_event,
            buy2buy_weights_max_recency_event,
        ) = look_into_dict(
            candidate_list=candidate_list,
            top_20_clicks=top_20_clicks,
            top_15_buy2buy=top_15_buy2buy,
            top_15_buys=top_15_buys,
        )

        logging.info("compare max_weighted_recency_event_in_session_aid with candidate")
        cand_df["pair_list"] = (
            cand_df["max_weighted_recency_event_in_session_aid"].astype(str)
            + "_"
            + cand_df["candidate_aid"].astype(str)
        )
        candidate_list = list(cand_df["pair_list"].values)
        (
            click_weights_max_weighted_recency_event,
            buys_weights_max_weighted_recency_event,
            buy2buy_weights_max_weighted_recency_event,
        ) = look_into_dict(
            candidate_list=candidate_list,
            top_20_clicks=top_20_clicks,
            top_15_buy2buy=top_15_buy2buy,
            top_15_buys=top_15_buys,
        )

        logging.info("compare max_log_duration_event_in_session_aid with candidate")
        cand_df["pair_list"] = (
            cand_df["max_log_duration_event_in_session_aid"].astype(str)
            + "_"
            + cand_df["candidate_aid"].astype(str)
        )
        candidate_list = list(cand_df["pair_list"].values)
        (
            click_weights_max_duration_event,
            buys_weights_max_duration_event,
            buy2buy_weights_max_duration_event,
        ) = look_into_dict(
            candidate_list=candidate_list,
            top_20_clicks=top_20_clicks,
            top_15_buy2buy=top_15_buy2buy,
            top_15_buys=top_15_buys,
        )

        logging.info(
            "compare max_weighted_log_duration_event_in_session_aid with candidate"
        )
        cand_df["pair_list"] = (
            cand_df["max_weighted_log_duration_event_in_session_aid"].astype(str)
            + "_"
            + cand_df["candidate_aid"].astype(str)
        )
        candidate_list = list(cand_df["pair_list"].values)
        (
            click_weights_max_weighted_duration_event,
            buys_weights_max_weighted_duration_event,
            buy2buy_weights_max_weighted_duration_event,
        ) = look_into_dict(
            candidate_list=candidate_list,
            top_20_clicks=top_20_clicks,
            top_15_buy2buy=top_15_buy2buy,
            top_15_buys=top_15_buys,
        )

        # add to cand_df
        cand_df = cand_df[["session", "candidate_aid"]]
        # click weights
        cand_df[
            "click_weight_with_last_event_in_session_aid"
        ] = click_weights_last_event
        cand_df[
            "click_weight_with_max_recency_event_in_session_aid"
        ] = click_weights_max_recency_event
        cand_df[
            "click_weight_with_max_weighted_recency_event_in_session_aid"
        ] = click_weights_max_weighted_recency_event
        cand_df[
            "click_weight_with_max_log_duration_event_in_session_aid"
        ] = click_weights_max_duration_event
        cand_df[
            "click_weight_with_max_weighted_log_duration_event_in_session_aid"
        ] = click_weights_max_weighted_duration_event

        # buys weights
        cand_df["buys_weight_with_last_event_in_session_aid"] = buys_weights_last_event
        cand_df[
            "buys_weight_with_max_recency_event_in_session_aid"
        ] = buys_weights_max_recency_event
        cand_df[
            "buys_weight_with_max_weighted_recency_event_in_session_aid"
        ] = buys_weights_max_weighted_recency_event
        cand_df[
            "buys_weight_with_max_log_duration_event_in_session_aid"
        ] = buys_weights_max_duration_event
        cand_df[
            "buys_weight_with_max_weighted_log_duration_event_in_session_aid"
        ] = buys_weights_max_weighted_duration_event

        # buy2buy weights
        cand_df[
            "buy2buy_weight_with_last_event_in_session_aid"
        ] = buy2buy_weights_last_event
        cand_df[
            "buy2buy_weight_with_max_recency_event_in_session_aid"
        ] = buy2buy_weights_max_recency_event
        cand_df[
            "buy2buy_weight_with_max_weighted_recency_event_in_session_aid"
        ] = buy2buy_weights_max_weighted_recency_event
        cand_df[
            "buy2buy_weight_with_max_log_duration_event_in_session_aid"
        ] = buy2buy_weights_max_duration_event
        cand_df[
            "buy2buy_weight_with_max_weighted_log_duration_event_in_session_aid"
        ] = buy2buy_weights_max_weighted_duration_event

        # save item covisitation features
        filepath = output_path / f"{name}_{ix}_{event}_item_covisitation_feas.parquet"
        logging.info(f"save chunk to: {filepath}")
        cand_df = pl.from_pandas(cand_df)
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
        top_15_buys = get_top15_covisitation_buys_dict()
        logging.info("read local covisitation buy2buy")
        top_15_buy2buy = get_top15_covisitation_buy2buy_dict()
        logging.info("read local covisitation click")
        top_20_clicks = get_top20_covisitation_click_dict()
    else:
        logging.info("read scoring covisitation buys")
        top_15_buys = get_top15_covisitation_buys_dict(mode="scoring")
        logging.info("read scoring covisitation buy2buy")
        top_15_buy2buy = get_top15_covisitation_buy2buy_dict(mode="scoring")
        logging.info("read scoring covisitation click")
        top_20_clicks = get_top20_covisitation_click_dict(mode="scoring")

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
