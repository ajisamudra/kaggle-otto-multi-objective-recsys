import click
import polars as pl
from tqdm import tqdm
import gc
from pathlib import Path
import numpy as np
from gensim.models import KeyedVectors
from src.utils.constants import (
    CFG,
    get_processed_training_train_session_representation_items_dir,  # session representations dir
    get_processed_training_test_session_representation_items_dir,
    get_processed_scoring_train_session_representation_items_dir,
    get_processed_scoring_test_session_representation_items_dir,
    get_processed_training_train_word2vec_features_dir,  # output dir
    get_processed_training_test_word2vec_features_dir,
    get_processed_scoring_train_word2vec_features_dir,
    get_processed_scoring_test_word2vec_features_dir,
)
from src.utils.word2vec import load_word2vec_embedding, load_word2vec_cbow_embedding
from src.utils.memory import freemem, round_float_3decimals
from src.utils.logger import get_logger

logging = get_logger()

VECT_SIZE = 32


def vectorized_cosine_distance(vectors1: np.ndarray, vectors2: np.ndarray):
    """
    Reference: https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
    """
    cosine_distances = np.sum(vectors1 * vectors2, axis=1) / (
        np.linalg.norm(vectors1, axis=1) * np.linalg.norm(vectors2, axis=1)
    )
    return cosine_distances


def vectorized_euclidean_distance(vectors1: np.ndarray, vectors2: np.ndarray):
    return np.linalg.norm((vectors1 - vectors2), axis=1)


def calculate_distance_metrics(
    embedding: KeyedVectors,
    candidate_aids: list,
    last_event_aids: list,
    max_recency_aids: list,
    max_weighted_recency_aids: list,
    max_duration_aids: list,
    max_weighted_duration_aids: list,
):
    vectors1 = []
    vectors2 = []
    for source, target in zip(candidate_aids, last_event_aids):
        # vector1 = embedding.get_vector(source)
        # vector2 = embedding.get_vector(target)
        try:
            vector1 = embedding.get_vector(source)
        except KeyError:
            vector1 = [0 for _ in range(VECT_SIZE)]
        try:
            vector2 = embedding.get_vector(target)
        except KeyError:
            vector2 = [0 for _ in range(VECT_SIZE)]
        vectors1.append(vector1)
        vectors2.append(vector2)

    # convert list to array 2d
    nd_vectors1 = np.array(vectors1)
    nd_vectors2 = np.array(vectors2)

    # compute cosine similarity
    last_event_cosine_distances = vectorized_cosine_distance(nd_vectors1, nd_vectors2)
    last_event_euclidean_distances = vectorized_euclidean_distance(
        nd_vectors1, nd_vectors2
    )

    vectors2 = []
    for target in max_recency_aids:
        try:
            vector2 = embedding.get_vector(target)
        except KeyError:
            vector2 = [0 for _ in range(VECT_SIZE)]
        vectors2.append(vector2)

    # convert list to array 2d
    nd_vectors2 = np.array(vectors2)
    # compute cosine similarity
    max_recency_cosine_distances = vectorized_cosine_distance(nd_vectors1, nd_vectors2)
    max_recency_euclidean_distances = vectorized_euclidean_distance(
        nd_vectors1, nd_vectors2
    )

    vectors2 = []
    for target in max_weighted_recency_aids:
        try:
            vector2 = embedding.get_vector(target)
        except KeyError:
            vector2 = [0 for _ in range(VECT_SIZE)]
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

    vectors2 = []
    for target in max_duration_aids:
        try:
            vector2 = embedding.get_vector(target)
        except KeyError:
            vector2 = [0 for _ in range(VECT_SIZE)]
        vectors2.append(vector2)

    # convert list to array 2d
    nd_vectors2 = np.array(vectors2)
    # compute cosine similarity
    max_duration_cosine_distances = vectorized_cosine_distance(nd_vectors1, nd_vectors2)
    max_duration_euclidean_distances = vectorized_euclidean_distance(
        nd_vectors1, nd_vectors2
    )

    vectors2 = []
    for target in max_weighted_duration_aids:
        try:
            vector2 = embedding.get_vector(target)
        except KeyError:
            vector2 = [0 for _ in range(VECT_SIZE)]
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

    return (
        last_event_cosine_distances,
        last_event_euclidean_distances,
        max_recency_cosine_distances,
        max_recency_euclidean_distances,
        max_weighted_recency_cosine_distances,
        max_weighted_recency_euclidean_distances,
        max_duration_cosine_distances,
        max_duration_euclidean_distances,
        max_weighted_duration_cosine_distances,
        max_weighted_duration_euclidean_distances,
    )


def gen_word2vec_features(
    name: str,
    ix: int,
    ses_representation_path: Path,
    output_path: Path,
    word2vec_embedding: KeyedVectors,
    word2vec_cbow_embedding: KeyedVectors,
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

        logging.info("calculating distances in embedding word2vec skipgram")
        (
            word2vec_skipgram_last_event_cosine_distances,
            word2vec_skipgram_last_event_euclidean_distances,
            word2vec_skipgram_max_recency_cosine_distances,
            word2vec_skipgram_max_recency_euclidean_distances,
            word2vec_skipgram_max_weighted_recency_cosine_distances,
            word2vec_skipgram_max_weighted_recency_euclidean_distances,
            word2vec_skipgram_max_duration_cosine_distances,
            word2vec_skipgram_max_duration_euclidean_distances,
            word2vec_skipgram_max_weighted_duration_cosine_distances,
            word2vec_skipgram_max_weighted_duration_euclidean_distances,
        ) = calculate_distance_metrics(
            embedding=word2vec_embedding,
            candidate_aids=candidate_aids,
            last_event_aids=last_event_aids,
            max_recency_aids=max_recency_aids,
            max_weighted_recency_aids=max_weighted_recency_aids,
            max_duration_aids=max_duration_aids,
            max_weighted_duration_aids=max_weighted_duration_aids,
        )

        logging.info("calculating distances in embedding word2vec cbow")
        (
            word2vec_cbow_last_event_cosine_distances,
            word2vec_cbow_last_event_euclidean_distances,
            word2vec_cbow_max_recency_cosine_distances,
            word2vec_cbow_max_recency_euclidean_distances,
            word2vec_cbow_max_weighted_recency_cosine_distances,
            word2vec_cbow_max_weighted_recency_euclidean_distances,
            word2vec_cbow_max_duration_cosine_distances,
            word2vec_cbow_max_duration_euclidean_distances,
            word2vec_cbow_max_weighted_duration_cosine_distances,
            word2vec_cbow_max_weighted_duration_euclidean_distances,
        ) = calculate_distance_metrics(
            embedding=word2vec_cbow_embedding,
            candidate_aids=candidate_aids,
            last_event_aids=last_event_aids,
            max_recency_aids=max_recency_aids,
            max_weighted_recency_aids=max_weighted_recency_aids,
            max_duration_aids=max_duration_aids,
            max_weighted_duration_aids=max_weighted_duration_aids,
        )

        # save matrix factorization features
        output_data = {
            "session": sessions,
            "candidate_aid": candidate_aids,
            "word2vec_skipgram_last_event_cosine_distance": word2vec_skipgram_last_event_cosine_distances,
            "word2vec_skipgram_last_event_euclidean_distance": word2vec_skipgram_last_event_euclidean_distances,
            "word2vec_skipgram_max_recency_cosine_distance": word2vec_skipgram_max_recency_cosine_distances,
            "word2vec_skipgram_max_recency_euclidean_distance": word2vec_skipgram_max_recency_euclidean_distances,
            "word2vec_skipgram_max_weighted_recency_cosine_distance": word2vec_skipgram_max_weighted_recency_cosine_distances,
            "word2vec_skipgram_max_weighted_recency_euclidean_distance": word2vec_skipgram_max_weighted_recency_euclidean_distances,
            "word2vec_skipgram_max_duration_cosine_distance": word2vec_skipgram_max_duration_cosine_distances,
            "word2vec_skipgram_max_duration_euclidean_distance": word2vec_skipgram_max_duration_euclidean_distances,
            "word2vec_skipgram_max_weighted_duration_cosine_distance": word2vec_skipgram_max_weighted_duration_cosine_distances,
            "word2vec_skipgram_max_weighted_duration_euclidean_distance": word2vec_skipgram_max_weighted_duration_euclidean_distances,
            "word2vec_cbow_last_event_cosine_distance": word2vec_cbow_last_event_cosine_distances,
            "word2vec_cbow_last_event_euclidean_distance": word2vec_cbow_last_event_euclidean_distances,
            "word2vec_cbow_max_recency_cosine_distance": word2vec_cbow_max_recency_cosine_distances,
            "word2vec_cbow_max_recency_euclidean_distance": word2vec_cbow_max_recency_euclidean_distances,
            "word2vec_cbow_max_weighted_recency_cosine_distance": word2vec_cbow_max_weighted_recency_cosine_distances,
            "word2vec_cbow_max_weighted_recency_euclidean_distance": word2vec_cbow_max_weighted_recency_euclidean_distances,
            "word2vec_cbow_max_duration_cosine_distance": word2vec_cbow_max_duration_cosine_distances,
            "word2vec_cbow_max_duration_euclidean_distance": word2vec_cbow_max_duration_euclidean_distances,
            "word2vec_cbow_max_weighted_duration_cosine_distance": word2vec_cbow_max_weighted_duration_cosine_distances,
            "word2vec_cbow_max_weighted_duration_euclidean_distance": word2vec_cbow_max_weighted_duration_euclidean_distances,
        }

        output_df = pl.DataFrame(output_data)
        filepath = output_path / f"{name}_{ix}_{event}_word2vec_feas.parquet"
        logging.info(f"save chunk to: {filepath}")
        output_df = freemem(output_df)
        output_df = round_float_3decimals(output_df)
        output_df.write_parquet(f"{filepath}")
        logging.info(f"output df shape {output_df.shape}")

        del cand_df, output_df
        gc.collect()


def make_word2vec_features(
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
        logging.info("read local word2vec embedding")
        word2vec_embedding = load_word2vec_embedding()
        word2vec_cbow_embedding = load_word2vec_cbow_embedding()

    else:
        logging.info("read scoring word2vec embedding")
        word2vec_embedding = load_word2vec_embedding(mode="scoring")
        word2vec_cbow_embedding = load_word2vec_cbow_embedding(mode="scoring")

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(istart, iend)):
        logging.info(f"start creating word2vec features")
        gen_word2vec_features(
            name=name,
            ix=ix,
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            word2vec_embedding=word2vec_embedding,
            word2vec_cbow_embedding=word2vec_cbow_embedding,
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
        output_path = get_processed_training_train_word2vec_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_word2vec_features(
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
        output_path = get_processed_training_test_word2vec_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_word2vec_features(
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
        output_path = get_processed_scoring_train_word2vec_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_word2vec_features(
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
        output_path = get_processed_scoring_test_word2vec_features_dir()
        logging.info(f"read input data from: {ses_representation_path}")
        logging.info(f"will save chunks data to: {output_path}")
        make_word2vec_features(
            mode=mode,
            name="test",
            ses_representation_path=ses_representation_path,
            output_path=output_path,
            istart=istart,
            iend=iend,
        )


if __name__ == "__main__":
    main()
