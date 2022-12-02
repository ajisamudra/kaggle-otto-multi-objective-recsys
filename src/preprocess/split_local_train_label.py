import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    get_local_validation_dir,
    get_processed_local_validation_dir,
    get_full_data_dir,
    get_processed_full_data_dir,
)
from src.utils.logger import get_logger

logging = get_logger()


def split_local_train_last_week(
    train: pd.DataFrame,
    output_path: Path,
    frac: float = 0.2,
):
    all_train_unique_session = len(train.session.unique())

    # last 7 days of data
    seven_days = 7 * 24 * 60 * 60
    train_cutoff = train.ts.max() - seven_days
    train = train[train.ts > train_cutoff]
    last_week_unique_session = len(train.session.unique())

    # sample frac of session_id
    # frac = 0.2
    logging.info(f"sample input data with frac: {frac}")
    lucky_sessions_train = train.drop_duplicates(["session"]).sample(frac=frac)[
        "session"
    ]
    subset_of_train = train[train.session.isin(lucky_sessions_train)]
    subset_last_week_unique_session = len(subset_of_train.session.unique())

    del train
    gc.collect()

    logging.info(f"train unique session: {all_train_unique_session}")
    logging.info(f"last week train unique session: {last_week_unique_session}")
    logging.info(
        f"subset last week train unique session: {subset_last_week_unique_session}"
    )

    train_features = []
    train_labels = []

    for grp in tqdm(subset_of_train.groupby("session")):
        # randomly select cutoff for label & features: sequence of aids
        if grp[1].shape[0] != 1:
            cutoff = np.random.randint(
                1, grp[1].shape[0]
            )  # we want at least a single item in our validation data for each session
            train_features.append(grp[1].iloc[:cutoff])
            train_labels.append(grp[1].iloc[cutoff:])

    del subset_of_train
    gc.collect()

    # save subset train & its label to small-local-validation
    logging.info("create df train feature")
    train_features = pd.concat(train_features).reset_index(drop=True)
    filepath = output_path / "train.parquet"
    logging.info(f"save df train feature to: {filepath}")
    train_features.to_parquet(filepath)

    del train_features
    gc.collect()

    logging.info("create df train ground truth")
    train_labels = pd.concat(train_labels).reset_index(drop=True)
    train_labels["type"] = train_labels["type"].map(
        {0: "clicks", 1: "carts", 2: "orders"}
    )
    train_labels = (
        train_labels.groupby(["session", "type"])["aid"].apply(list).reset_index()
    )
    train_labels.columns = ["session", "type", "ground_truth"]
    filepath = output_path / "train_labels.parquet"
    logging.info(f"save df train feature to: {filepath}")
    train_labels.to_parquet(filepath)


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training/scoring",
)
def main(mode: str):
    if mode == "training":
        # local-training: save label to small-local-validation
        input_path = get_local_validation_dir()
        output_path = get_processed_local_validation_dir()
        logging.info(f"read input data from: {input_path}")
        train = pd.read_parquet(input_path / "train.parquet")
        split_local_train_last_week(
            train=train,
            output_path=output_path,
            frac=0.2,
        )
    elif mode == "scoring":
        # scoring use all training data
        input_path = get_full_data_dir()
        output_path = get_processed_full_data_dir()
        logging.info(f"read input data from: {input_path}")
        train = pd.read_parquet(input_path / "train.parquet")
        split_local_train_last_week(
            train=train,
            output_path=output_path,
            frac=0.2,
        )


if __name__ == "__main__":
    main()
