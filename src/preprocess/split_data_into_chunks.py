import click
import pandas as pd
import polars as pl
import numpy as np
from random import shuffle
from tqdm import tqdm
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_local_validation_dir,
    get_processed_full_data_dir,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
)
from src.utils.logger import get_logger

logging = get_logger()


def stratified_sample_session(data: pd.DataFrame, n_splits: int):
    # stratified sample based on following session type
    # session has click + cart + buy ("3_3")
    # session has click + cart / click + buy / cart + buy ("1_2", "2_2", "3_2")
    # session has click only / cart only / buy only ( "0_1", "2_1", "1_1")

    unique_session_type = data.groupby(["session", "type"])["aid"].count().reset_index()
    unique_session_type = (
        unique_session_type.groupby("session")
        .agg({"type": [("type_sum", "sum"), ("type_dcount", "nunique")]})
        .reset_index(["session"], col_fill=None)
    )
    unique_session_type.columns = unique_session_type.columns.get_level_values(1)
    unique_session_type["stratified_col"] = (
        unique_session_type["type_sum"].astype(str)
        + "_"
        + unique_session_type["type_dcount"].astype(str)
    )
    classes = ["3_3", "0_1", "1_2", "2_2", "2_1", "1_1", "3_2"]
    # Get each of the classes into their own list of samples
    class_split_list = {}
    for curr_class in classes:
        class_list = list(
            set(
                unique_session_type.iloc[
                    unique_session_type.groupby(["stratified_col"]).groups[curr_class]
                ]["session"].tolist()
            )
        )
        shuffle(class_list)
        class_split_list[curr_class] = np.array_split(
            class_list, n_splits
        )  # create a dict of split chunks

    stratified_sample_chunks = []
    for i in range(n_splits):
        class_chunks = []
        for curr_class in classes:
            class_chunks.extend(
                class_split_list[curr_class][i]
            )  # get split from current class
        stratified_sample_chunks.append(class_chunks)

    return stratified_sample_chunks


def split_data_into_chunks(data: pl.DataFrame, name: str, output_path: Path):
    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test
    # split
    df = data.to_pandas()
    unique_session = list(df["session"].values)
    logging.info(f"start split data into {n} chunks")
    for ix, chunk_sessions in tqdm(
        # enumerate(stratified_sample_session(data=data.to_pandas(), n_splits=n))
        enumerate(np.array_split(unique_session, n))
    ):
        logging.info(f"chunk {ix} have unique session {len(chunk_sessions)}")
        logging.info(
            f"assuming each sesssion will have 40 candidates: n_row {len(chunk_sessions)*40}"
        )
        subset_of_data = data.filter(pl.col("session").is_in(list(chunk_sessions)))
        logging.info(subset_of_data["type"].value_counts(sort=True))
        filepath = output_path / f"{name}_{ix}.parquet"
        logging.info(f"save chunk {ix} to: {filepath}")
        subset_of_data.write_parquet(f"{filepath}")


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    if mode == "training_train":
        input_path = get_processed_local_validation_dir()
        output_path = get_processed_training_train_splitted_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        data = pl.read_parquet(input_path / "train.parquet")
        split_data_into_chunks(data=data, name="train", output_path=output_path)

    elif mode == "training_test":
        input_path = get_processed_local_validation_dir()
        output_path = get_processed_training_test_splitted_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        data = pl.read_parquet(input_path / "test.parquet")
        split_data_into_chunks(data=data, name="test", output_path=output_path)

    elif mode == "scoring_train":
        input_path = get_processed_full_data_dir()
        output_path = get_processed_scoring_train_splitted_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        data = pl.read_parquet(input_path / "train.parquet")
        split_data_into_chunks(data=data, name="train", output_path=output_path)

    elif mode == "scoring_test":
        input_path = get_processed_full_data_dir()
        output_path = get_processed_scoring_test_splitted_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        data = pl.read_parquet(input_path / "test.parquet")
        split_data_into_chunks(data=data, name="test", output_path=output_path)


if __name__ == "__main__":
    main()
