import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    get_processed_local_validation_dir,
    get_processed_full_data_dir,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
)
from src.utils.logger import get_logger

logging = get_logger()


def split_data_into_chunks(data: pd.DataFrame, name: str, output_path: Path):
    # get sessions
    unique_sessions = data["session"].unique()
    n = 10
    # split
    logging.info(f"start split data into {n} chunks")
    for ix, chunk_sessions in tqdm(enumerate(np.array_split(unique_sessions, n))):
        logging.info(f"chunk {ix} have unique session {len(chunk_sessions)}")
        logging.info(
            f"assuming each sesssion will have 40 candidates: n_row {len(chunk_sessions)*40}"
        )
        subset_of_data = data[data.session.isin(chunk_sessions)]
        filepath = output_path / f"{name}_{ix}.parquet"
        logging.info(f"save chunk {ix} to: {filepath}")
        subset_of_data.to_parquet(f"{filepath}")


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
        data = pd.read_parquet(input_path / "train.parquet")
        split_data_into_chunks(data=data, name="train", output_path=output_path)

    elif mode == "training_test":
        input_path = get_processed_local_validation_dir()
        output_path = get_processed_training_test_splitted_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        data = pd.read_parquet(input_path / "test.parquet")
        split_data_into_chunks(data=data, name="test", output_path=output_path)

    elif mode == "scoring_train":
        input_path = get_processed_full_data_dir()
        output_path = get_processed_scoring_train_splitted_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        data = pd.read_parquet(input_path / "train.parquet")
        split_data_into_chunks(data=data, name="train", output_path=output_path)

    elif mode == "scoring_test":
        input_path = get_processed_full_data_dir()
        output_path = get_processed_scoring_test_splitted_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        data = pd.read_parquet(input_path / "test.parquet")
        split_data_into_chunks(data=data, name="test", output_path=output_path)


if __name__ == "__main__":
    main()
