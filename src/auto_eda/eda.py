import click
import polars as pl
import pandas as pd
from tqdm import tqdm
import numpy as np
import gc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, List
from scipy.stats import pointbiserialr
from src.utils.constants import (
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_training_test_dataset_dir,
    get_processed_scoring_test_dataset_dir,
)
from src.utils.constants import (
    get_artifacts_eda_dir,
    check_directory,
)

from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def get_numeric_features(data: pd.DataFrame, target_variable: Union[str, None]) -> List:
    # filter only numeric features
    numeric_features = list(data.select_dtypes(include=["number"]).columns)
    # drop features with _id, cols_to_drop, and target variable
    numeric_features = [f for f in numeric_features if "_id" not in f]

    if target_variable is None:  # pragma: no cover
        return numeric_features

    # remove the target_variable
    numeric_features.remove(target_variable)

    return numeric_features


def biserial_correlation(data: pd.DataFrame, target_variable: str):

    numeric_features = get_numeric_features(data=data, target_variable=target_variable)

    df = pd.DataFrame()
    for feature in numeric_features:
        r, p = pointbiserialr(data[target_variable], data[feature])
        df_temp = pd.DataFrame({"feature": [feature], "R": [r], "p-value": [p]})
        df = df.append(df_temp, ignore_index=True)

    return df


def report_biserial_correlation(
    data: pd.DataFrame, target_variable: str, filepath: str
):
    # plot variable
    y = "R"
    x = "feature"

    # calculate biserial correlation
    df_corr = biserial_correlation(data=data, target_variable=target_variable)
    df_corr = df_corr.sort_values(by=y, ascending=False)
    n_feature = df_corr.shape[0]

    # visualize the correlation value
    filename_str = f"{filepath}/point_biserial.png"

    plt.figure(figsize=(7, int(n_feature * 0.3)))
    plt.title(f"Biserial Correlation Score")
    sns.barplot(x=y, y=x, data=df_corr, color="#348ABD")
    plt.savefig(filename_str, bbox_inches="tight")
    plt.close()
    logging.info(f"Save figures to: {filename_str}")

    # save the csv
    filename_str = f"{filepath}/point_biserial.csv"
    df_corr.to_csv(filename_str, index=False)
    logging.info(f"Save CSV to: {filename_str}")


def perform_eda(events: list, n: int, mode: str):
    input_path = ""
    val_input_path = ""
    scoring_input_path = ""
    week = "w2"
    if week == "w2":
        input_path = get_processed_training_train_dataset_dir()
        val_input_path = get_processed_training_test_dataset_dir()
        scoring_input_path = get_processed_scoring_test_dataset_dir()

    for EVENT in events:
        logging.info(f"perform EDA for event {EVENT.upper()}")
        artifact_path = get_artifacts_eda_dir(event=EVENT, week=week)
        if mode in ["all", "across_dataset"]:
            logging.info("read feature distribution across train/val/test")
            IX = 0
            df = pd.DataFrame()
            logging.info(f"read training data for chunk: {IX}")
            filepath = f"{input_path}/train_{IX}_{EVENT}_combined.parquet"
            train_df = pd.read_parquet(filepath)
            train_df.loc[:, "dataset"] = "train"
            logging.info(train_df.shape)

            logging.info(f"read validation data for chunk: {IX}")
            filepath = f"{val_input_path}/test_{IX}_{EVENT}_combined.parquet"
            val_df = pd.read_parquet(filepath)
            val_df.loc[:, "dataset"] = "validation"
            logging.info(val_df.shape)

            logging.info(f"read test data for chunk: {IX}")
            filepath = f"{scoring_input_path}/test_{IX}_{EVENT}_combined.parquet"
            test_df = pd.read_parquet(filepath)
            test_df.loc[:, "dataset"] = "test"
            logging.info(test_df.shape)

            logging.info("concat to all chunks")
            df = pd.concat([df, train_df, val_df, test_df], ignore_index=True)
            logging.info(df.shape)

            del train_df, val_df, test_df
            gc.collect()

            logging.info("start plot feature distribution across dataset")
            # create artifact dir
            filepath = artifact_path / "dist_across_dataset"
            check_directory(filepath)
            for feature in tqdm(df.select_dtypes(include=["number"]).columns):
                if feature == TARGET:
                    continue
                plt.figure(figsize=(6, 4))
                plt.title(f"{feature} distribution across dataset")
                sns.histplot(
                    x=df[feature],
                    hue=df["dataset"],
                    element="step",
                    stat="density",
                    common_norm=False,
                    bins=50,
                )
                plt.savefig(
                    f"{filepath}/{feature}_across_dataset.png", bbox_inches="tight"
                )
                plt.close()

        logging.info(f"reading feature from {n} chunks")
        df = pd.DataFrame()
        for IX in tqdm(range(n)):

            logging.info(f"read training data for chunk: {IX}")
            filepath = f"{input_path}/train_{IX}_{EVENT}_combined.parquet"
            train_df = pd.read_parquet(filepath)
            train_df.loc[:, "chunk"] = IX
            logging.info(train_df.shape)

            logging.info("concat to all chunks")
            df = pd.concat([df, train_df], ignore_index=True)
            logging.info(df.shape)

            del train_df
            gc.collect()

        if mode in ["all", "class_dist"]:
            # create artifact dir
            filepath = artifact_path / "dist_per_class"
            check_directory(filepath)
            logging.info("start plot feature distribution per class")
            for feature in tqdm(df.select_dtypes(include=["number"]).columns):
                if feature == TARGET:
                    continue
                plt.figure(figsize=(6, 4))
                plt.title(f"{feature} distribution per class (original)")
                sns.histplot(
                    x=df[feature],
                    hue=df[TARGET],
                    element="step",
                    stat="density",
                    common_norm=False,
                    bins=50,
                )
                plt.savefig(f"{filepath}/{feature}_original.png", bbox_inches="tight")
                plt.close()

                plt.figure(figsize=(6, 4))
                plt.title(f"{feature} distribution per class (log2(1+x))")
                sns.histplot(
                    x=np.log2(0.001 + df[feature]),
                    hue=df[TARGET],
                    element="step",
                    stat="density",
                    common_norm=False,
                    bins=50,
                )
                plt.savefig(f"{filepath}/{feature}_log2.png", bbox_inches="tight")
                plt.close()

        if mode in ["all", "biserial"]:
            logging.info("start calculating biserial correlaction")
            report_biserial_correlation(
                data=df, target_variable=TARGET, filepath=f"{artifact_path}"
            )

        if mode in ["all", "chunk_dist"]:
            logging.info("start plot feature distribution per class")
            # create artifact dir
            filepath = artifact_path / "dist_per_chunk"
            check_directory(filepath)
            for feature in tqdm(df.select_dtypes(include=["number"]).columns):
                if feature == TARGET:
                    continue
                plt.figure(figsize=(6, 4))
                plt.title(f"{feature} distribution per chunks")
                sns.histplot(
                    x=df[feature],
                    hue=df["chunk"],
                    element="step",
                    stat="density",
                    common_norm=False,
                    bins=50,
                )
                plt.savefig(f"{filepath}/{feature}_per_chunk.png", bbox_inches="tight")
                plt.close()

        logging.info(
            f"EDA artifacts for event {EVENT} could be found here {artifact_path}"
        )


@click.command()
@click.option(
    "--event",
    default="all",
    help="avaiable event: clicks/carts/orders/all",
)
@click.option(
    "--mode",
    default="all",
    help="avaiable event: across_dataset/class_dist/chunk_dist/biserial/all",
)
@click.option(
    "--n",
    default=5,
    help="number of chunk for training; between 1-10",
)
def main(event: str = "all", n: int = 1, mode: str = "all"):
    events = ["clicks", "carts", "orders"]
    if event != "all":
        events = [event]
    perform_eda(events=events, n=n, mode=mode)


if __name__ == "__main__":
    main()
