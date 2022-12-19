import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from annoy import AnnoyIndex
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
from src.utils.matrix_factorization import load_annoy_idx_matrix_fact_embedding
from src.utils.logger import get_logger

logging = get_logger()


def suggest_matrix_fact(
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
    embedding: AnnoyIndex,
):
    sessions = []
    candidates = []
    for session, aids in tqdm(ses2aids.items()):
        # unique_aids = set(aids)
        unique_aids = list(dict.fromkeys(aids[::-1]))
        types = ses2types[session]
        mf_candidate = embedding.get_nns_by_item(unique_aids[0], n=n_candidate)

        # append to list result
        sessions.append(session)
        candidates.append(mf_candidate)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"
    return result_series


def generate_candidates_matrix_fact(
    name: str, input_path: Path, output_path: Path, embedding: AnnoyIndex
):
    if name == "train":
        n = CFG.N_train
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

        # retrieve matrix factorization candidates
        candidates_series = suggest_matrix_fact(
            n_candidate=CFG.matrix_factorization_candidates,
            ses2aids=ses2aids,
            ses2types=ses2types,
            embedding=embedding,
        )

        for event in ["clicks", "carts", "orders"]:
            logging.info(f"suggesting candidate {event}")
            candidates_series_tmp = candidates_series.copy(deep=True)
            logging.info("create candidates df")
            candidate_list_df = pd.DataFrame(
                candidates_series_tmp.add_suffix(f"_{event}"), columns=["labels"]
            ).reset_index()

            filepath = output_path / f"{name}_{ix}_{event}_matrix_fact_list.parquet"
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

    if mode in ["training_train", "training_test"]:
        logging.info("read local matrix factorization index")
        embedding = load_annoy_idx_matrix_fact_embedding()
    else:
        logging.info("read scoring matrix factorization index")
        embedding = load_annoy_idx_matrix_fact_embedding(mode="scoring")

    if mode == "training_train":
        input_path = get_processed_training_train_splitted_dir()
        output_path = get_processed_training_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_matrix_fact(
            name="train",
            input_path=input_path,
            output_path=output_path,
            embedding=embedding,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_matrix_fact(
            name="test",
            input_path=input_path,
            output_path=output_path,
            embedding=embedding,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_matrix_fact(
            name="train",
            input_path=input_path,
            output_path=output_path,
            embedding=embedding,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_matrix_fact(
            name="test",
            input_path=input_path,
            output_path=output_path,
            embedding=embedding,
        )


if __name__ == "__main__":
    main()
