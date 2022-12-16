import numpy as np
from src.utils.constants import get_scoring_word2vec_dir, get_local_word2vec_dir
from annoy import AnnoyIndex
from gensim.models import KeyedVectors


VECTOR = 32
NTREE = 15
WINDOW = 20
NEGATIVE = 5


def load_word2vec_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = (
            f"{emd_path}/word2vec_scoring_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
        )

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")
    return kvectors


def load_annoy_idx_word2vec_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = (
            f"{emd_path}/word2vec_scoring_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
        )

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(VECTOR, "euclidean")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index
