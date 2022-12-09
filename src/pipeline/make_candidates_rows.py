import click
import pandas as pd
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_train_candidates_dir,
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
    get_processed_local_validation_dir,
    get_processed_full_data_dir,
)
from src.utils.logger import get_logger

logging = get_logger()


# read chunk parquet
# for each chunk -> pivot N candidates into rows
# so for each session, there's N-1 additional rows


def pivot_candidates_list_to_rows(
    cand_df: pd.DataFrame,
    is_train: bool,
    include_all_gt: bool,
    drop_zero_positive_sample: bool,
    ses2truth: dict = {},
):
    """
    cand_df
    session | labels
    123_clicks | [aid1, aid2, aid3]
    123_carts | [aid1, aid2, aid3]
    123_orders | [aid1, aid2, aid3]
    """
    # drop event in session
    # logging.info("create prediction session column")
    cand_df["session"] = cand_df["session"].apply(lambda x: int(x.split("_")[0]))

    # dict of session and labels
    # logging.info("create ses2candidates")
    ses2candidates = dict(zip(cand_df["session"], cand_df["labels"]))
    unique_sessions = list(cand_df["session"].values)

    sessions = []
    candidates = []
    labels = []

    logging.info(
        f"pivot in train_mode: {is_train} | include_all_gt: {include_all_gt} | drop_zero_positive_sample: {drop_zero_positive_sample}"
    )
    for session in tqdm(unique_sessions):
        truths = set()
        if is_train:
            # get truths for specific session
            truths = set(ses2truth.get(session, []))

            if drop_zero_positive_sample:
                # if there's no truth for specific event & session
                # drop from training data
                if len(truths) == 0:
                    continue

        # get candidates for specific session
        cands = set(ses2candidates[session])

        if include_all_gt:
            # add all truths if it's training data
            cands = cands | truths

        # check whether it's in truths
        label_ls = [1 if c in truths else 0 for c in cands]
        session_ls = [session for i in range(len(cands))]
        cands_ls = cands

        sessions.extend(session_ls)
        candidates.extend(cands_ls)
        labels.extend(label_ls)

    # save as df
    data = {
        "session": sessions,
        "candidate_aid": candidates,
        "label": labels,
    }

    return pd.DataFrame(data)


def pivot_candidates(
    name: str,
    is_train: bool,
    include_all_gt: bool,
    drop_zero_positive_sample: bool,
    input_path: Path,
    output_path: Path,
    df_truth: pd.DataFrame = pd.DataFrame(),
):
    """
    df_truth
    session | type | ground_truth
    123 | clicks | [AID1 AID4]
    123 | carts | [AID1]
    123 | orders | [AID1]
    """

    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(n)):
        for event in ["clicks", "carts", "orders"]:
            # logging.info(f"chunk {ix}: read input")
            filepath = f"{input_path}/{name}_{ix}_{event}_list.parquet"
            df = pd.read_parquet(filepath)
            # input df as follow
            # session | labels
            # A_clicks | [aid1, aid2]
            # A_carts | [aid1, aid2]
            # A_orders  | [aid1, aid2]
            ses2truth = {}
            if is_train:
                # logging.info("create ses2truth")
                label_session = df_truth.loc[df_truth["type"] == event][
                    "session"
                ].values
                label_truth = df_truth.loc[df_truth["type"] == event][
                    "ground_truth"
                ].values
                ses2truth = dict(zip(label_session, label_truth))

            # logging.info(f"start pivoting candidate")
            df_output = pivot_candidates_list_to_rows(
                cand_df=df,
                is_train=is_train,
                include_all_gt=include_all_gt,
                drop_zero_positive_sample=drop_zero_positive_sample,
                ses2truth=ses2truth,
            )

            filepath = output_path / f"{name}_{ix}_{event}_rows.parquet"
            logging.info(f"save chunk {ix}_{event} to: {filepath}")
            df_output.to_parquet(f"{filepath}")
            logging.info(f"output df shape {df_output.shape}")

            del df, df_output
            gc.collect()


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    if mode == "training_train":
        input_path = get_processed_training_train_candidates_dir()
        output_path = get_processed_training_train_candidates_dir()
        logging.info(f"will read input from & save output to: {input_path}")
        truth = get_processed_local_validation_dir()
        truth = f"{truth}/train_labels.parquet"
        logging.info(f"read ses2truth from: {truth}")
        df_truth = pd.read_parquet(truth)
        pivot_candidates(
            name="train",
            is_train=True,
            include_all_gt=False,
            drop_zero_positive_sample=True,
            input_path=input_path,
            output_path=output_path,
            df_truth=df_truth,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_candidates_dir()
        output_path = get_processed_training_test_candidates_dir()
        logging.info(f"will read input from & save output to: {input_path}")
        truth = get_processed_local_validation_dir()
        truth = f"{truth}/test_labels.parquet"
        logging.info(f"read ses2truth from: {truth}")
        df_truth = pd.read_parquet(truth)
        logging.info(f"df_truth shape: {df_truth.shape}")
        pivot_candidates(
            name="test",
            is_train=True,
            include_all_gt=False,
            drop_zero_positive_sample=True,
            input_path=input_path,
            output_path=output_path,
            df_truth=df_truth,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_candidates_dir()
        output_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"will read input from & save output to: {input_path}")
        truth = get_processed_full_data_dir()
        truth = f"{truth}/train_labels.parquet"
        logging.info(f"read ses2truth from: {truth}")
        df_truth = pd.read_parquet(truth)
        pivot_candidates(
            name="train",
            is_train=True,
            include_all_gt=False,
            drop_zero_positive_sample=True,
            input_path=input_path,
            output_path=output_path,
            df_truth=df_truth,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_candidates_dir()
        output_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"will read input from & save output to: {input_path}")
        pivot_candidates(
            name="test",
            is_train=False,
            include_all_gt=False,
            drop_zero_positive_sample=False,
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
