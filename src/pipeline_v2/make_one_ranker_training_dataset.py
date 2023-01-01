import click
import polars as pl
import numpy as np
from tqdm import tqdm
import gc
from src.utils.constants import (
    CFG,
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_training_test_dataset_dir,
    get_processed_scoring_train_dataset_dir,
)
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def make_label_one_ranker(mode: str, istart: int, iend: int):
    candidate_path = ""
    name = ""

    if mode == "training_train":
        candidate_path = get_processed_training_train_dataset_dir()
        name = "train"

    elif mode == "training_test":
        candidate_path = get_processed_training_test_dataset_dir()
        name = "train"

    elif mode == "scoring_train":
        candidate_path = get_processed_scoring_train_dataset_dir()
        name = "train"

    else:
        raise NotImplementedError("mode not implemented for this task")

    logging.info(f"read final dataset from: {candidate_path}")
    # iterate over chunks
    logging.info(f"iterate from {istart} to {iend} chunks")
    for ix in tqdm(range(istart, iend)):
        df = pl.DataFrame()
        for event in ["clicks", "carts", "orders"]:
            c_path = f"{candidate_path}/{name}_{ix}_{event}_combined.parquet"
            df_chunk = pl.read_parquet(c_path)

            # change label so that click:1, cart:2, order:3
            if event == "carts":
                df_chunk = df_chunk.with_columns([(pl.col("label") * 2).alias("label")])
            elif event == "orders":
                df_chunk = df_chunk.with_columns([(pl.col("label") * 3).alias("label")])

            df = pl.concat([df, df_chunk])

        # aggregate to max
        df = df.groupby(["session", "candidate_aid"]).max()
        # save
        filepath = f"{candidate_path}/{name}_{ix}_one_ranker_combined.parquet"
        logging.info(f"save chunk to: {filepath}")
        df = freemem(df)
        df.write_parquet(f"{filepath}")
        logging.info(f"output df shape {df.shape}")

        del df
        gc.collect()


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train",
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

    make_label_one_ranker(mode=mode, istart=istart, iend=iend)


if __name__ == "__main__":
    main()
