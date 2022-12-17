import numpy as np
from src.utils.constants import get_scoring_fasttext_dir, get_local_fasttext_dir
from annoy import AnnoyIndex
from gensim.models import KeyedVectors


VECTOR = 32
NTREE = 15
WINDOW = 5
NEGATIVE = 1


def load_fasttext_skipgram_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_fasttext_dir()
        filepath = f"{emd_path}/fasttext_local_skipgram_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
    else:
        emd_path = get_scoring_fasttext_dir()
        filepath = f"{emd_path}/fasttext_scoring_skipgram_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")
    return kvectors


def load_fasttext_cbow_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_fasttext_dir()
        filepath = (
            f"{emd_path}/fasttext_local_cbow_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
        )
    else:
        emd_path = get_scoring_fasttext_dir()
        filepath = (
            f"{emd_path}/fasttext_scoring_cbow_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
        )

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")
    return kvectors


def load_annoy_idx_fasttext_skipgram_wdw5_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_fasttext_dir()
        filepath = f"{emd_path}/fasttext_local_skipgram_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
    else:
        emd_path = get_scoring_fasttext_dir()
        filepath = f"{emd_path}/fasttext_scoring_skipgram_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"

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


def load_annoy_idx_fasttext_skipgram_wdw20_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_fasttext_dir()
        filepath = f"{emd_path}/fasttext_local_skipgram_vec{VECTOR}_wdw20_neg5.kv"
    else:
        emd_path = get_scoring_fasttext_dir()
        filepath = f"{emd_path}/fasttext_scoring_skipgram_vec{VECTOR}_wdw20_neg5.kv"

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
