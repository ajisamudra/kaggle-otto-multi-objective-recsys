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
    get_processed_training_train_session_representation_items_dir,  # output dir
    get_processed_training_test_session_representation_items_dir,
    get_processed_scoring_train_session_representation_items_dir,
    get_processed_scoring_test_session_representation_items_dir,
)
from src.features.preprocess_events import preprocess_events
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def gen_session_representation_items(
    data: pl.DataFrame,
    name: str,
    mode: str,
    ix: int,
    candidate_path: Path,
    output_path: Path,
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
            pl.col("duration_second").sum().alias("sum_duration_second"),
            # pl.col("type_weighted_log_duration_second")
            # .sum()
            # .alias("sum_type_weighted_log_duration_second"),
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

    # aid with log_duration_second
    data_aids_log_duration = data_agg.filter(
        pl.col("sum_duration_second")
        == pl.col("sum_duration_second").max().over("session")
    ).select(["session", pl.col("aid").alias("max_duration_event_in_session_aid")])
    # remove duplicate
    data_aids_log_duration = data_aids_log_duration.groupby("session").agg(
        [
            pl.col("max_duration_event_in_session_aid")
            .max()
            .alias("max_duration_event_in_session_aid")
        ]
    )

    # # aid with sum_type_weighted_log_duration_second
    # data_aids_wighted_log_duration = data_agg.filter(
    #     pl.col("sum_type_weighted_log_duration_second")
    #     == pl.col("sum_type_weighted_log_duration_second").max().over("session")
    # ).select(
    #     [
    #         "session",
    #         pl.col("aid").alias("max_weighted_log_duration_event_in_session_aid"),
    #     ]
    # )
    # # remove duplicate
    # data_aids_wighted_log_duration = data_aids_wighted_log_duration.groupby(
    #     "session"
    # ).agg(
    #     [
    #         pl.col("max_weighted_log_duration_event_in_session_aid")
    #         .max()
    #         .alias("max_weighted_log_duration_event_in_session_aid")
    #     ]
    # )
    # join
    # now each session is represented by 5 aids
    # we can get covisitation score using 5 covisitation matrix
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
    data_aids = data_aids.join(
        data_aids_log_duration,
        how="left",
        left_on=["session"],
        right_on=["session"],
    )
    # data_aids = data_aids.join(
    #     data_aids_wighted_log_duration,
    #     how="left",
    #     left_on=["session"],
    #     right_on=["session"],
    # )

    logging.info("get 4 aids representing 1 session")
    data_aids = freemem(data_aids)

    del (
        # data_aids_wighted_log_duration,
        data_aids_log_duration,
        data_aids_weighted_recency,
        data_aids_recency,
    )
    gc.collect()

    for event in ["clicks", "carts", "orders"]:
        logging.info(f"read candidate for event {event.upper()}")
        if (mode == "training_train") & (event == "clicks") & (ix > 6):
            logging.info("click ix > 6 continue")
            continue

        # read candidate
        filepath = f"{candidate_path}/{name}_{ix}_{event}_rows.parquet"
        cand_df = pl.read_parquet(filepath)
        cand_df = freemem(cand_df)

        # left join with data_aids
        logging.info("join with session-representation-aids")
        cand_df = cand_df.join(
            data_aids,
            how="left",
            left_on=["session"],
            right_on=["session"],
        )

        # save session representation items
        filepath = (
            output_path / f"{name}_{ix}_{event}_session_representation_items.parquet"
        )
        logging.info(f"save chunk to: {filepath}")
        cand_df = freemem(cand_df)
        cand_df.write_parquet(f"{filepath}")
        logging.info(f"output df shape {cand_df.shape}")

        del cand_df
        gc.collect()

    del data_aids, data_agg
    gc.collect()


def make_session_representation_items(
    name: str,
    mode: str,
    input_path: Path,
    candidate_path: Path,
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

        logging.info(f"start creating sesion representation items")
        gen_session_representation_items(
            data=df,
            name=name,
            mode=mode,
            ix=ix,
            candidate_path=candidate_path,
            output_path=output_path,
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
        output_path = get_processed_training_train_session_representation_items_dir()
        candidate_path = get_processed_training_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_representation_items(
            name="train",
            mode=mode,
            input_path=input_path,
            candidate_path=candidate_path,
            output_path=output_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_session_representation_items_dir()
        candidate_path = get_processed_training_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_representation_items(
            name="test",
            mode=mode,
            input_path=input_path,
            candidate_path=candidate_path,
            output_path=output_path,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_session_representation_items_dir()
        candidate_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_representation_items(
            name="train",
            mode=mode,
            input_path=input_path,
            candidate_path=candidate_path,
            output_path=output_path,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_session_representation_items_dir()
        candidate_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_session_representation_items(
            name="test",
            mode=mode,
            input_path=input_path,
            candidate_path=candidate_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
