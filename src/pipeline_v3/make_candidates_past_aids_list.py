import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from collections import Counter
from src.utils.constants import (
    CFG,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
    get_processed_training_train_candidates_dir,
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
)
from src.utils.logger import get_logger

logging = get_logger()


def suggest_past_aids(
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
):
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    sessions = []
    candidates = []
    ranks_list = []
    for session, aids in tqdm(ses2aids.items()):
        # unique_aids = set(aids)
        unique_aids = list(dict.fromkeys(aids[::-1]))
        types = ses2types[session]

        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]
            candidate = [k for k, v in aids_temp.most_common(n_candidate)]

        else:
            candidate = list(unique_aids)[:n_candidate]

        # append to list result
        rank_list = [i for i in range(len(candidate))]
        sessions.append(session)
        candidates.append(candidate)
        ranks_list.append(rank_list)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"

    return result_series, ranks_list


def generate_candidates_past_aids(
    name: str,
    mode: str,
    input_path: Path,
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
        # logging.info("create ses2aids")
        ses2aids = df.groupby("session")["aid"].apply(list).to_dict()
        # logging.info("create ses2types")
        ses2types = df.groupby("session")["type"].apply(list).to_dict()

        logging.info("input type class proportion")
        logging.info(df["type"].value_counts(ascending=False))

        del df
        gc.collect()

        candidates_list = pd.Series()
        ranks_list = []

        logging.info(f"retrieve past candidates")

        candidates_list, ranks_list = suggest_past_aids(
            n_candidate=CFG.past_candidates,
            ses2aids=ses2aids,
            ses2types=ses2types,
        )

        for event in ["clicks", "carts", "orders"]:
            logging.info(f"suggesting candidate {event}")
            candidates_series_tmp = candidates_list.copy(deep=True)
            logging.info("create candidates df")
            candidate_list_df = pd.DataFrame(
                candidates_series_tmp.add_suffix(f"_{event}"), columns=["labels"]
            ).reset_index()
            candidate_list_df["ranks"] = ranks_list
            filepath = output_path / f"{name}_{ix}_{event}_past_aids_list.parquet"
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
        output_path = get_processed_training_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_past_aids(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_past_aids(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_past_aids(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_past_aids(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
