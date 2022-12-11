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
    get_processed_training_train_candidates_dir,  # candidates dir
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
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
from src.features.preprocess_events import preprocess_events
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def gen_item_covisitation_features(
    data: pl.DataFrame,
    name: str,
    ix: int,
    candidate_path: Path,
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

    # START: event data preprocess
    data = preprocess_events(data)
    # END: event data preprocess

    # agg per session & aid
    data_agg = data.groupby(["session", "aid"]).agg(
        [
            pl.col("action_num_reverse_chrono")
            .min()
            .alias("min_action_num_reverse_chrono"),
            pl.col("log_recency_score").sum().alias("sum_log_recency_score"),
            pl.col("type_weighted_log_recency_score")
            .sum()
            .alias("sum_type_weighted_log_recency_score"),
        ]
    )
    data_agg = data_agg.sort(pl.col("session"))

    # get last item per session
    data_aids = data_agg.filter(
        pl.col("min_action_num_reverse_chrono")
        == pl.col("min_action_num_reverse_chrono").min().over("session")
    ).select(["session", pl.col("aid").alias("last_event_in_session_aid")])

    # aid with max_log_recency_score
    data_aids_recency = data_agg.filter(
        pl.col("sum_log_recency_score")
        == pl.col("sum_log_recency_score").max().over("session")
    ).select(["session", pl.col("aid").alias("max_recency_event_in_session_aid")])

    # aid with max_type_weighted_log_recency_score
    data_aids_weighted_recency = data_agg.filter(
        pl.col("sum_type_weighted_log_recency_score")
        == pl.col("sum_type_weighted_log_recency_score").max().over("session")
    ).select(
        ["session", pl.col("aid").alias("max_weighted_recency_event_in_session_aid")]
    )

    # join
    # now each session is represented by 3 aids
    # we can get covisitation score using 3 covisitation matrix
    data_aids = data_aids.join(
        data_aids_recency,
        how="left",
        left_on=["session"],
        right_on=["session"],
    )
    data_aids = data_aids.join(
        data_aids_weighted_recency,
        how="left",
        left_on=["session"],
        right_on=["session"],
    )
    logging.info("get 3 aids representing 1 session")

    for event in ["clicks", "carts", "orders"]:
        logging.info(f"read candidate for event {event.upper()}")
        # read candidate
        filepath = f"{candidate_path}/{name}_{ix}_{event}_rows.parquet"
        cand_df = pl.read_parquet(filepath)

        cand_df = cand_df.with_columns(
            [
                pl.col("session").cast(pl.Int32).alias("session"),
                pl.col("candidate_aid").cast(pl.Int32).alias("candidate_aid"),
            ]
        )

        # left join with data_aids
        logging.info("join with session-3aids")
        cand_df = cand_df.join(
            data_aids,
            how="left",
            left_on=["session"],
            right_on=["session"],
        )

        # get click_covisitation weight
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
            "click_weight_with_last_event_in_session_aid",
        ]

        # get click_covisitation weight
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
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
        ]

        # get click_covisitation weight
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
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
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
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
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
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
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
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_withmax_weighted_recency_event_in_session_aid",
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
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_withmax_weighted_recency_event_in_session_aid",
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
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_withmax_weighted_recency_event_in_session_aid",
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
            "click_weight_with_last_event_in_session_aid",
            "click_weight_with_max_recency_event_in_session_aid",
            "click_weight_with_max_weighted_recency_event_in_session_aid",
            "buys_weight_with_last_event_in_session_aid",
            "buys_weight_with_max_recency_event_in_session_aid",
            "buys_weight_withmax_weighted_recency_event_in_session_aid",
            "buy2buy_weight_last_event_in_session_aid",
            "buy2buy_weight_max_recency_event_in_session_aid",
            "buy2buy_weight_max_weighted_recency_event_in_session_aid",
        ]

        cand_df = cand_df.select(
            [
                "session",
                "candidate_aid",
                "click_weight_with_last_event_in_session_aid",
                "click_weight_with_max_recency_event_in_session_aid",
                "click_weight_with_max_weighted_recency_event_in_session_aid",
                "buys_weight_with_last_event_in_session_aid",
                "buys_weight_with_max_recency_event_in_session_aid",
                "buys_weight_withmax_weighted_recency_event_in_session_aid",
                "buy2buy_weight_last_event_in_session_aid",
                "buy2buy_weight_max_recency_event_in_session_aid",
                "buy2buy_weight_max_weighted_recency_event_in_session_aid",
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

    del data_aids, data_agg
    gc.collect()


def make_item_covisitation_features(
    name: str,
    input_path: Path,
    candidate_path: Path,
    output_path: Path,
):

    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test

    logging.info("read covisitation buys")
    top_15_buys = get_top15_covisitation_buys_df()
    logging.info("read covisitation buy2buy")
    top_15_buy2buy = get_top15_covisitation_buy2buy_df()
    logging.info("read covisitation click")
    top_20_clicks = get_top20_covisitation_click_df()

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(n)):
        # logging.info(f"chunk {ix}: read input")
        filepath = f"{input_path}/{name}_{ix}.parquet"
        df = pl.read_parquet(filepath)

        logging.info(f"start creating item covisitation features")
        gen_item_covisitation_features(
            data=df,
            name=name,
            ix=ix,
            candidate_path=candidate_path,
            output_path=output_path,
            top_15_buys=top_15_buys,
            top_15_buy2buy=top_15_buy2buy,
            top_20_clicks=top_20_clicks,
        )

        del df
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
        candidate_path = get_processed_training_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_item_covisitation_features(
            name="train",
            input_path=input_path,
            candidate_path=candidate_path,
            output_path=output_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_item_covisitation_features_dir()
        candidate_path = get_processed_training_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_item_covisitation_features(
            name="test",
            input_path=input_path,
            candidate_path=candidate_path,
            output_path=output_path,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_item_covisitation_features_dir()
        candidate_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_item_covisitation_features(
            name="train",
            input_path=input_path,
            candidate_path=candidate_path,
            output_path=output_path,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_item_covisitation_features_dir()
        candidate_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_item_covisitation_features(
            name="test",
            input_path=input_path,
            candidate_path=candidate_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
