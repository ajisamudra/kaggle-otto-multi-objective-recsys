import click
import polars as pl
import numpy as np
from tqdm import tqdm
import gc
from src.utils.constants import (
    CFG,
    get_processed_training_train_target_encoding_dir,  # target encoding features
    get_processed_training_test_target_encoding_dir,
    get_processed_scoring_train_target_encoding_dir,
    get_processed_scoring_test_target_encoding_dir,
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_training_test_dataset_dir,
    get_processed_scoring_train_dataset_dir,
    get_processed_scoring_test_dataset_dir,
)
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def fcombine_features(mode: str, event: str, ix: int):
    te_features_path = ""
    output_path = ""
    name = ""

    if mode == "training_train":
        te_features_path = get_processed_training_train_target_encoding_dir()
        output_path = get_processed_training_train_dataset_dir()
        name = "train"

    elif mode == "training_test":
        te_features_path = get_processed_training_test_target_encoding_dir()
        output_path = get_processed_training_test_dataset_dir()
        name = "test"

    elif mode == "scoring_test":
        te_features_path = get_processed_scoring_test_target_encoding_dir()
        output_path = get_processed_scoring_test_dataset_dir()
        name = "test"

    logging.info(f"read training data from: {output_path}")
    logging.info(f"will save chunks data to: {output_path}")

    filepath = f"{output_path}/{name}_{ix}_{event}_combined.parquet"

    cand_df = pl.read_parquet(filepath)
    logging.info(f"read candidates with shape {cand_df.shape}")

    cat_columns = ["rank_covisit", "rank_combined", "retrieval_covisit"]
    for cat_col in cat_columns:
        logging.info(f"left join with target encodig column: {cat_col.upper()}")

        features_path = ""
        if name == "test":
            features_path = (
                f"{te_features_path}/{name}_{event}_{cat_col}_target_encoding.parquet"
            )
        elif name == "train":
            features_path = f"{te_features_path}/{name}_{ix}_{event}_{cat_col}_target_encoding.parquet"

        # read target encoding features
        te_fea_agg = pl.read_parquet(features_path)
        # get global mean just in case there's missing keys
        global_mean = (
            te_fea_agg.filter(pl.col(cat_col) == -1)[f"target_encoding_{cat_col}_mean"]
            .to_pandas()
            .values[0]
        )
        logging.info(f"read te col: {cat_col} with shape {te_fea_agg.shape}")
        cand_df = cand_df.join(te_fea_agg, how="left", on=[cat_col]).fill_null(
            pl.lit(global_mean)
        )
        logging.info(f"joined with te col: {cat_col}! shape {cand_df.shape}")

        del te_fea_agg
        gc.collect()

    filepath = f"{output_path}/{name}_{ix}_{event}_combined.parquet"
    logging.info(f"save chunk to: {filepath}")
    cand_df = freemem(cand_df)
    cand_df.write_parquet(f"{filepath}")
    logging.info(f"output df shape {cand_df.shape}")

    del cand_df
    gc.collect()


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
        n = CFG.N_train
    elif mode == "training_test":
        n = CFG.N_local_test
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(istart, iend)):
        for event in ["clicks", "carts", "orders"]:
            logging.info(f"start combining features")
            fcombine_features(mode=mode, event=event, ix=ix)


if __name__ == "__main__":
    main()
