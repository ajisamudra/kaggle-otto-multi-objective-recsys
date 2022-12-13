import click
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_test_candidates_dir,
    get_processed_training_train_candidates_dir,
    get_processed_local_validation_dir,
)
from src.metrics.submission_evaluation import measure_recall
from src.utils.logger import get_logger

logging = get_logger()


# read chunk parquet
# for each chunk -> suggest 40 candidates clicks, carts, buys
# save 40 candidates clickc, carts, buys in different files


def eval_candidate_list(
    name: str,
    input_path: Path,
    gt_path: str,
):
    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    df_pred = pl.DataFrame()
    for ix in tqdm(range(n)):
        for event in ["clicks", "carts", "orders"]:
            filepath = input_path / f"{name}_{ix}_{event}_list.parquet"
            df_chunk = pl.read_parquet(filepath)
            df_pred = pl.concat([df_pred, df_chunk])
    logging.info("convert to pandas")
    df_pred = df_pred.to_pandas()
    logging.info(df_pred.shape)
    df_pred.columns = ["session_type", "labels"]
    df_pred["labels"] = df_pred.labels.apply(lambda x: " ".join(map(str, x)))
    logging.info(df_pred.head())

    logging.info("start computing metrics")
    # COMPUTE METRIC
    # read ground truth
    df_truth = pd.read_parquet(gt_path)
    measure_recall(df_pred=df_pred, df_truth=df_truth, Ks=[20, 40, 80])


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test",
)
def main(mode: str):

    if mode == "training_train":
        input_path = get_processed_training_train_candidates_dir()
        gt_path = get_processed_local_validation_dir()
        gt_path = f"{gt_path}/train_labels.parquet"
        logging.info(f"read input data from: {input_path}")
        eval_candidate_list(
            name="train",
            input_path=input_path,
            gt_path=gt_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_candidates_dir()
        gt_path = get_processed_local_validation_dir()
        gt_path = f"{gt_path}/test_labels.parquet"
        logging.info(f"read input data from: {input_path}")
        eval_candidate_list(
            name="test",
            input_path=input_path,
            gt_path=gt_path,
        )


if __name__ == "__main__":
    main()
