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
from src.utils.fasttext import load_annoy_idx_fasttext_skipgram_wdw20_embedding
from src.utils.logger import get_logger

logging = get_logger()


def suggest_clicks_fasttext(
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
    embedding: AnnoyIndex,
):
    sessions = []
    candidates = []
    ranks_list = []
    for session, aids in tqdm(ses2aids.items()):
        # unique_aids = set(aids)
        unique_aids = list(dict.fromkeys(aids[::-1]))
        types = ses2types[session]
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


def suggest_orders_fasttext(
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
    embedding: AnnoyIndex,
):
    sessions = []
    candidates = []
    for session, aids in tqdm(ses2aids.items()):
        # unique_aids = set(aids)
        # unique_aids = list(dict.fromkeys(aids[::-1]))
        types = ses2types[session]

        # get reverse aids and its types
        reversed_aids = aids[::-1]
        reversed_types = types[::-1]

        carted_aids = [
            aid for i, aid in enumerate(reversed_aids) if reversed_types[i] == 1
        ]

        # # last x aids
        # if len(carted_aids) == 0:
        #     last_x_aids = reversed_aids[:1]
        # elif len(carted_aids) < 3:
        #     last_x_aids = carted_aids[: len(carted_aids)]
        # else:
        #     last_x_aids = carted_aids[:3]
        # # get query vector from last three aids
        # query_vcts = []
        # for aid in last_x_aids:
        #     vct = []
        #     try:
        #         vct = embedding.get_item_vector(aid)
        #     except KeyError:
        #         continue
        #     query_vcts.append(vct)
        # query_vcts = np.array(query_vcts)
        # query_vcts = np.mean(query_vcts, axis=0)

        # mf_candidate = embedding.get_nns_by_vector(query_vcts, n=n_candidate)

        # get last cart event
        cart_idx = 0
        try:
            cart_idx = reversed_types.index(1)
        except ValueError:
            cart_idx = 0

        mf_candidate = embedding.get_nns_by_item(reversed_aids[cart_idx], n=n_candidate)

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
        logging.info(f"retrieve candidate clicks")
        clicks_candidates_series, ranks_list = suggest_clicks_fasttext(
            n_candidate=CFG.fasttext_candidates,
            ses2aids=ses2aids,
            ses2types=ses2types,
            embedding=embedding,
        )

        # logging.info(f"retrieve candidate orders")
        # orders_candidates_series = suggest_orders_fasttext(
        #     n_candidate=CFG.fasttext_candidates,
        #     ses2aids=ses2aids,
        #     ses2types=ses2types,
        #     embedding=embedding,
        # )

        for event in ["clicks", "carts", "orders"]:
            logging.info(f"suggesting candidate {event}")
            candidates_series_tmp = clicks_candidates_series.copy(deep=True)
            logging.info("create candidates df")
            candidate_list_df = pd.DataFrame(
                candidates_series_tmp.add_suffix(f"_{event}"), columns=["labels"]
            ).reset_index()
            candidate_list_df["ranks"] = ranks_list

            filepath = output_path / f"{name}_{ix}_{event}_fasttext_list.parquet"
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
        logging.info("read local fasttext index")
        embedding = load_annoy_idx_fasttext_skipgram_wdw20_embedding()
    else:
        logging.info("read scoring fasttext index")
        embedding = load_annoy_idx_fasttext_skipgram_wdw20_embedding(mode="scoring")

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
