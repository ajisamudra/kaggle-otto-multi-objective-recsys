import numpy as np
from src.utils.constants import get_scoring_matrix_fact_dir, get_local_matrix_fact_dir


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
