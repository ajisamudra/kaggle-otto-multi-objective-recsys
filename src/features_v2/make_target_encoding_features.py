import click
import polars as pl
from tqdm import tqdm
import numpy as np
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_training_test_dataset_dir,
    get_processed_scoring_train_dataset_dir,
    get_processed_scoring_test_dataset_dir,
    get_processed_training_train_target_encoding_dir,  # output features
    get_processed_training_test_target_encoding_dir,
    get_processed_scoring_train_target_encoding_dir,
    get_processed_scoring_test_target_encoding_dir,
)
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def gen_target_encoding(data: pl.DataFrame, column: str):
    """
    df input
    session | cat_col | label
    123 | 1 | 1
    123 | 2 | 1
    123 | 1 | 0
    """

    # weight smoothing (at least there's 1000 obs then trust the target encoding)
    w_smoothing = 1000

    data_agg = data.groupby(column).agg(
        [
            pl.col(TARGET).mean().alias(f"target_encoding_{column}_mean"),
        ]
    )

    global_mean = {
        column: [-1],
        f"target_encoding_{column}_mean": [data[TARGET].mean()],
        f"{column}_count": [data[TARGET].shape[0]],
    }
    global_mean = pl.DataFrame(global_mean)
    data_agg = data_agg.with_columns([pl.col(f"{column}_count").cast(pl.Int32)])
    data_agg = freemem(data_agg)
    global_mean = freemem(global_mean)

    data_agg = pl.concat([data_agg, global_mean])

    return data_agg


def iterate_target_encoding(
    data: pl.DataFrame, name: str, output_path: Path, idx: int, event: str
):
    cat_columns = ["rank_covisit", "rank_combined", "retrieval_covisit"]
    for cat_col in cat_columns:
        logging.info(f"target encodig for column: {cat_col.upper()}")
        # calculate target encoding for that columns
        df_output = gen_target_encoding(data=data, column=cat_col)
        df_output = freemem(df_output)

        filepath = ""
        if name == "test":
            filepath = output_path / f"{name}_{event}_{cat_col}_target_encoding.parquet"
        elif name == "train":
            filepath = (
                output_path / f"{name}_{idx}_{event}_{cat_col}_target_encoding.parquet"
            )
        logging.info(f"save chunk to: {filepath}")
        df_output.write_parquet(f"{filepath}")
        logging.info(f"output df shape {df_output.shape}")

        del df_output
        gc.collect()


def make_target_encoding_features(
    name: str,
    input_path: Path,
    output_path: Path,
):

    for event in ["clicks", "carts", "orders"]:
        logging.info(f"target encodig for event: {event.upper()}")

        if name == "train":
            # store each fold target encoding
            logging.info(f"{name} mode iterate over {CFG.N_train}")
            if event in ["orders", "carts"]:
                # create target encoding for each fold
                # use all other folds without that fold data
                for i in tqdm(range(CFG.N_train)):
                    train_df = pl.DataFrame()
                    for j in range(CFG.N_train):
                        if j == i:
                            continue
                        filepath = f"{input_path}/train_{i}_{event}_combined.parquet"
                        df_chunk = pl.read_parquet(filepath)
                        train_df = pl.concat([train_df, df_chunk])

                    iterate_target_encoding(
                        data=train_df,
                        name=name,
                        output_path=output_path,
                        idx=i,
                        event=event,
                    )

            else:
                # create target encoding for each fold
                # use only 5 other folds without that fold data
                for i in tqdm(range(CFG.N_train)):
                    train_df = pl.DataFrame()
                    start_idx = (i + 1) % CFG.N_train
                    end_idx = (i + 6) % CFG.N_train
                    if end_idx < start_idx:
                        for j in range(start_idx, CFG.N_train):
                            filepath = (
                                f"{input_path}/train_{i}_{event}_combined.parquet"
                            )
                            df_chunk = pl.read_parquet(filepath)
                            train_df = pl.concat([train_df, df_chunk])
                        for j in range(0, end_idx):
                            filepath = (
                                f"{input_path}/train_{i}_{event}_combined.parquet"
                            )
                            df_chunk = pl.read_parquet(filepath)
                            train_df = pl.concat([train_df, df_chunk])
                    else:
                        for j in range(start_idx, end_idx):
                            filepath = (
                                f"{input_path}/train_{i}_{event}_combined.parquet"
                            )
                            df_chunk = pl.read_parquet(filepath)
                            train_df = pl.concat([train_df, df_chunk])

                    iterate_target_encoding(
                        data=train_df,
                        name=name,
                        output_path=output_path,
                        idx=i,
                        event=event,
                    )

        elif name == "test":
            # iterate over all training data and store target encoding
            train_df = pl.DataFrame()
            logging.info(f"{name} mode only create 1 file")
            if event in ["orders", "carts"]:
                for i in range(CFG.N_train):
                    filepath = f"{input_path}/train_{i}_{event}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    train_df = pl.concat([train_df, df_chunk])
            else:
                for i in range(int(CFG.N_train / 5)):
                    filepath = f"{input_path}/train_{i}_{event}_combined.parquet"
                    df_chunk = pl.read_parquet(filepath)
                    train_df = pl.concat([train_df, df_chunk])

            iterate_target_encoding(
                data=train_df, name=name, output_path=output_path, idx=0, event=event
            )


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test",
)
def main(mode: str):
    if mode == "training_train":
        input_path = get_processed_training_train_dataset_dir()
        output_path = get_processed_training_train_target_encoding_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_target_encoding_features(
            name="train",
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_train_dataset_dir()
        output_path = get_processed_training_test_target_encoding_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_target_encoding_features(
            name="test",
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
