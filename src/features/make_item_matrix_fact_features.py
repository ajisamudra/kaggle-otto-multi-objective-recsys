import click
import polars as pl
from tqdm import tqdm
import gc
from pathlib import Path
import numpy as np
from src.utils.constants import (
    CFG,
    get_processed_training_train_session_representation_items_dir,  # session representations dir
    get_processed_training_test_session_representation_items_dir,
    get_processed_scoring_train_session_representation_items_dir,
    get_processed_scoring_test_session_representation_items_dir,
    get_processed_training_train_matrix_fact_features_dir,  # output dir
    get_processed_training_test_matrix_fact_features_dir,
    get_processed_scoring_train_matrix_fact_features_dir,
    get_processed_scoring_test_matrix_fact_features_dir,
)
from src.utils.matrix_factorization import load_matrix_fact_embedding
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def vectorized_cosine_distance(vectors1: np.array, vectors2: np.array):
    """
    Reference: https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
    """
    cosine_distances = np.sum(vectors1 * vectors2, axis=1) / (
        np.linalg.norm(vectors1, axis=1) * np.linalg.norm(vectors2, axis=1)
    )
    return cosine_distances


def vectorized_euclidean_distance(vectors1: np.array, vectors2: np.array):
    return np.linalg.norm((vectors1 - vectors2), axis=1)


def gen_matrix_fact_features(
    name: str,
    ix: int,
    ses_representation_path: Path,
    output_path: Path,
    embedding: np.ndarray,
):
    """
    session representation aids
    session | candidate_aid | last_event_in_session_aid | last_event_in_session_aid
    123 | AID1 | AID1 | AID4
    123 | AID2 | AID1 | AID4
    123 | AID3 | AID1 | AID4
    """

    for event in ["clicks", "carts", "orders"]:
        logging.info(f"read session representation aids for event {event.upper()}")

        # read session representation
        filepath = f"{ses_representation_path}/{name}_{ix}_{event}_session_representation_items.parquet"
        cand_df = pl.read_parquet(filepath)

        # # dummy data
        # candidates = [95, 124, 67, 97] * 400000
        # ls_sources = [124, 124, 124, 124] * 400000
        # data = {
        #     "session": candidates,
        #     "candidate_aid": candidates,
        #     "last_event_in_session_aid": ls_sources,
        #     "max_recency_event_in_session_aid": ls_sources,
        #     "max_weighted_recency_event_in_session_aid": ls_sources,
        #     "max_log_duration_event_in_session_aid": ls_sources,
        #     "max_weighted_log_duration_event_in_session_aid": ls_sources,
        # }
        # cand_df = pl.DataFrame(data)

        # measure distances between 2 vectors
        # "candidate_aid" vs "last_event_in_session_aid"
        # "candidate_aid" vs "max_recency_event_in_session_aid",
        # "candidate_aid" vs "max_weighted_recency_event_in_session_aid",
        # "candidate_aid" vs "max_log_duration_event_in_session_aid",
        # "candidate_aid" vs "max_weighted_log_duration_event_in_session_aid"

        sessions = cand_df["session"].to_list()
        candidate_aids = cand_df["candidate_aid"].to_list()
        last_event_aids = cand_df["last_event_in_session_aid"].to_list()
        max_recency_aids = cand_df["max_recency_event_in_session_aid"].to_list()
        max_weighted_recency_aids = cand_df[
            "max_weighted_recency_event_in_session_aid"
        ].to_list()
        max_duration_aids = cand_df["max_log_duration_event_in_session_aid"].to_list()
        max_weighted_duration_aids = cand_df[
            "max_weighted_log_duration_event_in_session_aid"
        ].to_list()

        logging.info("calculating distances between candidate_aid & last_event_aid")
        vectors1 = []
        vectors2 = []
        for source, target in zip(candidate_aids, last_event_aids):
            vector1 = embedding[source]
            vector2 = embedding[target]
            vectors1.append(vector1)
            vectors2.append(vector2)
            # cosine_dist = distance.cosine(vector1, vector2)
            # jaccard_dist = distance.jaccard(vector1, vector2)
            # euclidean_dist = np.linalg.norm(vector1 - vector2)

        # convert list to array 2d
        nd_vectors1 = np.array(vectors1)
        nd_vectors2 = np.array(vectors2)

        # compute cosine similarity
        last_event_cosine_distances = vectorized_cosine_distance(
            nd_vectors1, nd_vectors2
        )
        last_event_euclidean_distances = vectorized_euclidean_distance(
            nd_vectors1, nd_vectors2
        )

        logging.info("calculating distances between candidate_aid & max_recency_aid")
        vectors2 = []
        for target in max_recency_aids:
            vector2 = embedding[target]
            vectors2.append(vector2)

        # convert list to array 2d
        nd_vectors2 = np.array(vectors2)
        # compute cosine similarity
        max_recency_cosine_distances = vectorized_cosine_distance(
            nd_vectors1, nd_vectors2
        )
        max_recency_euclidean_distances = vectorized_euclidean_distance(
            nd_vectors1, nd_vectors2
        )

        logging.info(
            "calculating distances between candidate_aid & max_weighted_recency_aids"
        )
        vectors2 = []
        for target in max_weighted_recency_aids:
            vector2 = embedding[target]
            vectors2.append(vector2)

        # convert list to array 2d
        nd_vectors2 = np.array(vectors2)
        # compute cosine similarity
        max_weighted_recency_cosine_distances = vectorized_cosine_distance(
            nd_vectors1, nd_vectors2
        )
        max_weighted_recency_euclidean_distances = vectorized_euclidean_distance(
            nd_vectors1, nd_vectors2
        )

        logging.info("calculating distances between candidate_aid & max_duration_aids")
        vectors2 = []
        for target in max_duration_aids:
            vector2 = embedding[target]
            vectors2.append(vector2)

        # convert list to array 2d
        nd_vectors2 = np.array(vectors2)
        # compute cosine similarity
        max_duration_cosine_distances = vectorized_cosine_distance(
            nd_vectors1, nd_vectors2
        )
        max_duration_euclidean_distances = vectorized_euclidean_distance(
            nd_vectors1, nd_vectors2
        )

        logging.info(
            "calculating distances between candidate_aid & max_weighted_duration_aids"
        )
        vectors2 = []
        for target in max_weighted_duration_aids:
            vector2 = embedding[target]
            vectors2.append(vector2)

        # convert list to array 2d
        nd_vectors2 = np.array(vectors2)
        # compute cosine similarity
        max_weighted_duration_cosine_distances = vectorized_cosine_distance(
            nd_vectors1, nd_vectors2
        )
        max_weighted_duration_euclidean_distances = vectorized_euclidean_distance(
            nd_vectors1, nd_vectors2
        )

        # save matrix factorization features
        output_data = {
            "session": sessions,
            "candidate_aid": candidate_aids,
            "matrix_fact_last_event_cosine_distance": last_event_cosine_distances,
            "matrix_fact_last_event_euclidean_distance": last_event_euclidean_distances,
            "matrix_fact_max_recency_cosine_distance": max_recency_cosine_distances,
            "matrix_fact_max_recency_euclidean_distance": max_recency_euclidean_distances,
            "matrix_fact_max_weighted_recency_cosine_distance": max_weighted_recency_cosine_distances,
            "matrix_fact_max_weighted_recency_euclidean_distance": max_weighted_recency_euclidean_distances,
            "matrix_fact_max_duration_cosine_distance": max_duration_cosine_distances,
            "matrix_fact_max_duration_euclidean_distance": max_duration_euclidean_distances,
            "matrix_fact_max_weighted_duration_cosine_distance": max_weighted_duration_cosine_distances,
            "matrix_fact_max_weighted_duration_euclidean_distance": max_weighted_duration_euclidean_distances,
        }

        output_df = pl.DataFrame(output_data)
        filepath = output_path / f"{name}_{ix}_{event}_matrix_fact_feas.parquet"
        logging.info(f"save chunk to: {filepath}")
        output_df = freemem(output_df)
        output_df.write_parquet(f"{filepath}")
        logging.info(f"output df shape {output_df.shape}")

        del cand_df, output_df
        gc.collect()


def make_matrix_fact_features(
    mode: str,
    name: str,
    ses_representation_path: Path,
    output_path: Path,
    istart: int,
    iend: int,
):

    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test

    if mode in ["training_train", "training_test"]:
        logging.info("read local matrix factorization embedding")
        embedding = load_matrix_fact_embedding()
    else:
        logging.info("read scoring matrix factorization embedding")
        embedding = load_matrix_fact_embedding(mode="scoring")

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(istart, iend)):
        logging.info(f"start creating item covisitation features")
        gen_matrix_fact_features(
            name=name,
            ix=ix,
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            embedding=embedding,
        )


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
        ses_representation_path = (
            get_processed_training_train_session_representation_items_dir()
        )
        output_path = get_processed_training_train_matrix_fact_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_matrix_fact_features(
            mode=mode,
            name="train",
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            istart=istart,
            iend=iend,
        )

    elif mode == "training_test":
        ses_representation_path = (
            get_processed_training_test_session_representation_items_dir()
        )
        output_path = get_processed_training_test_matrix_fact_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_matrix_fact_features(
            mode=mode,
            name="test",
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            istart=istart,
            iend=iend,
        )

    elif mode == "scoring_train":
        ses_representation_path = (
            get_processed_scoring_train_session_representation_items_dir()
        )
        output_path = get_processed_scoring_train_matrix_fact_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_matrix_fact_features(
            mode=mode,
            name="train",
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            istart=istart,
            iend=iend,
        )

    elif mode == "scoring_test":
        ses_representation_path = (
            get_processed_scoring_test_session_representation_items_dir()
        )
        output_path = get_processed_scoring_test_matrix_fact_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_matrix_fact_features(
            mode=mode,
            name="test",
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            istart=istart,
            iend=iend,
        )


if __name__ == "__main__":
    main()
