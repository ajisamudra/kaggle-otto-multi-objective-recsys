import click
import pandas as pd
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_train_splitted_dir,  # session that will be suggested
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
    get_processed_training_train_tmp_candidates_dir,  # tmp popular items
    get_processed_training_test_tmp_candidates_dir,
    get_processed_scoring_train_tmp_candidates_dir,
    get_processed_scoring_test_tmp_candidates_dir,
    get_processed_training_train_candidates_dir,  # output dir
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
)
from src.utils.logger import get_logger

logging = get_logger()


def suggest_candidates(df: pd.DataFrame, event: str, name: str, tmp_cands_path: Path):

    # read candidate
    filepath = f"{tmp_cands_path}/{name}_popular_week_{event}.parquet"
    cand_df = pd.read_parquet(filepath)

    # get session last ts and convert it to date
    df = df.groupby("session")["ts"].max().reset_index()
    # add type for join key
    df["type"] = event
    # add event in session
    df["session"] = df["session"].astype(str) + "_" + event
    # left join with cand_df
    df = df.merge(cand_df, how="left", left_on="type", right_on="type")
    # drop unused columns
    df = df.drop(columns=["ts", "type"])

    return df


def generate_candidates_popular_week(
    name: str,
    mode: str,
    input_path: Path,
    tmp_cands_path: Path,
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
        logging.info(f"chunk {ix}: read input")
        filepath = f"{input_path}/{name}_{ix}.parquet"
        df = pd.read_parquet(filepath)
        # input df as follow
        # session | aid | ts | type
        # A     | 1234  | 1  | 0
        # A     | 123   | 2  | 0
        # A     | 1234  | 3  | 1

        for event in ["clicks", "carts", "orders"]:

            if (mode == "training_train") & (event == "clicks") & (ix > 6):
                logging.info("click ix > 6 continue")
                continue

            logging.info(f"suggesting candidate {event}")

            # suggest based on event, only just left join
            candidate_list_df = suggest_candidates(
                df=df, event=event, name=name, tmp_cands_path=tmp_cands_path
            )

            filepath = output_path / f"{name}_{ix}_{event}_popular_week_list.parquet"
            logging.info(f"save chunk {ix} to: {filepath}")
            candidate_list_df.to_parquet(f"{filepath}")

            del candidate_list_df
            gc.collect()


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):

    if mode == "training_train":
        input_path = get_processed_training_train_splitted_dir()
        tmp_cands_path = get_processed_training_train_tmp_candidates_dir()
        output_path = get_processed_training_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_popular_week(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            tmp_cands_path=tmp_cands_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        tmp_cands_path = get_processed_training_test_tmp_candidates_dir()
        output_path = get_processed_training_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_popular_week(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            tmp_cands_path=tmp_cands_path,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        tmp_cands_path = get_processed_scoring_train_tmp_candidates_dir()
        output_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_popular_week(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            tmp_cands_path=tmp_cands_path,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        tmp_cands_path = get_processed_scoring_test_tmp_candidates_dir()
        output_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_popular_week(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            tmp_cands_path=tmp_cands_path,
        )


if __name__ == "__main__":
    main()
