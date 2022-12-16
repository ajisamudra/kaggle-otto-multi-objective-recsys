import numpy as np
from src.utils.constants import get_scoring_matrix_fact_dir, get_local_matrix_fact_dir
from annoy import AnnoyIndex


def load_matrix_fact_embedding(mode: str = "local"):

    if mode == "local":
        matrix_path = get_local_matrix_fact_dir()
        filepath = f"{matrix_path}/matrix_factorization_embeddings_local.npz"
    else:
        matrix_path = get_scoring_matrix_fact_dir()
        filepath = f"{matrix_path}/matrix_factorization_embeddings_scoring.npz"

    # load np array
    embedding_dict = np.load(filepath)
    embedding = embedding_dict["arr_0"]
    return embedding


def load_annoy_idx_matrix_fact_embedding(mode: str = "local"):

    if mode == "local":
        matrix_path = get_local_matrix_fact_dir()
        filepath = f"{matrix_path}/matrix_factorization_embeddings_local.npz"
    else:
        matrix_path = get_scoring_matrix_fact_dir()
        filepath = f"{matrix_path}/matrix_factorization_embeddings_scoring.npz"

    # load np array
    embedding_dict = np.load(filepath)
    embedding = embedding_dict["arr_0"]

    # create annoy index for search nn
    index = AnnoyIndex(32, "euclidean")
    for i in range(embedding.shape[0]):
        index.add_item(i, embedding[i, :])
    index.build(150)

    return index


def load_matrix_fact_order_cart_embedding(mode: str = "local"):

    if mode == "local":
        matrix_path = get_local_matrix_fact_dir()
        filepath = f"{matrix_path}/matrix_factorization_cart_order_embeddings_local.npz"
    else:
        matrix_path = get_scoring_matrix_fact_dir()
        filepath = (
            f"{matrix_path}/matrix_factorization_cart_order_embeddings_scoring.npz"
        )

    # load np array
    embedding_dict = np.load(filepath)
    embedding = embedding_dict["arr_0"]
    return embedding


def load_matrix_fact_buy2buy_embedding(mode: str = "local"):

    if mode == "local":
        matrix_path = get_local_matrix_fact_dir()
        filepath = f"{matrix_path}/matrix_factorization_buy2buy_embeddings_local.npz"
    else:
        matrix_path = get_scoring_matrix_fact_dir()
        filepath = f"{matrix_path}/matrix_factorization_buy2buy_embeddings_scoring.npz"

    # load np array
    embedding_dict = np.load(filepath)
    embedding = embedding_dict["arr_0"]
    return embedding
