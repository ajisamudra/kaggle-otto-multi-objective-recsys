import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from annoy import AnnoyIndex
from src.utils.constants import (
    CFG,
    get_processed_training_train_candidates_dir,  # candidate output
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
    get_processed_training_train_query_representation_dir,  # query representation dir
    get_processed_training_test_query_representation_dir,
    get_processed_scoring_train_query_representation_dir,
    get_processed_scoring_test_query_representation_dir,
)
from src.utils.word2vec import load_annoy_idx_word2vec_embedding
from src.utils.logger import get_logger

logging = get_logger()


def suggest_clicks_word2vec(
    n_candidate: int,
    ses2aids: dict,
    embedding: AnnoyIndex,
):
    sessions = []
    candidates = []
    ranks_list = []
    for session, aids in tqdm(ses2aids.items()):
        # unique_aids = set(aids)
        unique_aids = list(dict.fromkeys(aids[::-1]))
        mf_candidate = embedding.get_nns_by_item(unique_aids[0], n=n_candidate + 1)
        # drop first result which is the query aid
        mf_candidate = mf_candidate[1:]

        # append to list result
        rank_list = [i for i in range(n_candidate)]
        sessions.append(session)
        candidates.append(mf_candidate)
        ranks_list.append(rank_list)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"
    return result_series, ranks_list


def generate_candidates_word2vec(
    name: str, mode: str, input_path: Path, output_path: Path, embedding: AnnoyIndex
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
        filepath = f"{input_path}/{name}_{ix}_query_representation.parquet"
        df = pd.read_parquet(filepath)
        # logging.info("create ses2aids")
        ses2aids = (
            df.groupby("session")["max_log_duration_event_in_session_aid"]
            .apply(list)
            .to_dict()
        )

        del df
        gc.collect()

        # retrieve candidates
        logging.info(f"retrieve candidate clicks")
        clicks_candidates_series, ranks_list = suggest_clicks_word2vec(
            n_candidate=CFG.word2vec_duration_candidates,
            ses2aids=ses2aids,
            embedding=embedding,
        )

        for event in ["clicks", "carts", "orders"]:

            if (mode == "training_train") & (event == "clicks") & (ix > 6):
                logging.info("click ix > 6 continue")
                continue

            logging.info(f"suggesting candidate {event}")
            candidates_series_tmp = clicks_candidates_series.copy(deep=True)
            logging.info("create candidates df")
            candidate_list_df = pd.DataFrame(
                candidates_series_tmp.add_suffix(f"_{event}"), columns=["labels"]
            ).reset_index()
            candidate_list_df["ranks"] = ranks_list

            filepath = (
                output_path / f"{name}_{ix}_{event}_word2vec_duration_list.parquet"
            )
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
        logging.info("read local word2vec index")
        embedding = load_annoy_idx_word2vec_embedding()
    else:
        logging.info("read scoring word2vec index")
        embedding = load_annoy_idx_word2vec_embedding(mode="scoring")

    if mode == "training_train":
        input_path = get_processed_training_train_query_representation_dir()
        output_path = get_processed_training_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_word2vec(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            embedding=embedding,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_query_representation_dir()
        output_path = get_processed_training_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_word2vec(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            embedding=embedding,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_query_representation_dir()
        output_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_word2vec(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            embedding=embedding,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_query_representation_dir()
        output_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_word2vec(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            embedding=embedding,
        )


if __name__ == "__main__":
    main()
