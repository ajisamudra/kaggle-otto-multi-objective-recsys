import numpy as np
from src.utils.constants import get_scoring_word2vec_dir, get_local_word2vec_dir
from annoy import AnnoyIndex
from gensim.models import KeyedVectors

#### BEST PERFORMING

VECTOR = 32
NTREE = 15
WINDOW = 50
NEGATIVE = 5


def load_word2vec_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = (
            f"{emd_path}/word2vec_local_skipgram_vec32_wdw{WINDOW}_neg{NEGATIVE}.kv"
        )
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = (
            f"{emd_path}/word2vec_scoring_skipgram_vec32_wdw{WINDOW}_neg{NEGATIVE}.kv"
        )

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")
    return kvectors


def load_word2vec_cbow_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_cbow_vec32_wdw{WINDOW}_neg{NEGATIVE}.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = (
            f"{emd_path}/word2vec_scoring_cbow_vec32_wdw{WINDOW}_neg{NEGATIVE}.kv"
        )

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")
    return kvectors


def load_annoy_idx_word2vec_vect64_wdw50_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_skipgram_vec64_wdw50_neg5.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_skipgram_vec64_wdw50_neg5.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(64, "angular")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index


def load_annoy_idx_word2vec_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_skipgram_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_skipgram_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"

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


def load_annoy_idx_word2vec_cbow_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = (
            f"{emd_path}/word2vec_local_cbow_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
        )
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = (
            f"{emd_path}/word2vec_scoring_cbow_vec{VECTOR}_wdw{WINDOW}_neg{NEGATIVE}.kv"
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


###### EXPERIMENTs


def load_annoy_idx_word2vec_wdw30_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_skipgram_vec{VECTOR}_wdw30_neg5.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_skipgram_vec{VECTOR}_wdw30_neg5.kv"

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


def load_annoy_idx_word2vec_wdw50_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_skipgram_vec{VECTOR}_wdw50_neg5.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_skipgram_vec{VECTOR}_wdw50_neg5.kv"

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


def load_annoy_idx_word2vec_wdw70_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_skipgram_vec{VECTOR}_wdw70_neg10.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_skipgram_vec{VECTOR}_wdw70_neg10.kv"

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


def load_annoy_idx_word2vec_dropn1_vect64_wdw50_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_skipgram_dropn1_vec64_wdw50_neg5.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_skipgram_dropn1_vec64_wdw50_neg5.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(64, "angular")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index


def load_annoy_idx_word2vec_vect64_wdw50_neg30_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_skipgram_vec64_wdw50_neg30.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_skipgram_vec64_wdw50_neg30.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(64, "angular")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index


def load_annoy_idx_word2vec_vect32_wdw50_neg5_real_session_embedding(
    mode: str = "local",
):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_clicks_skipgram_vec32_wdw50_neg5.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_clicks_skipgram_vec32_wdw50_neg5.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(32, "euclidean")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index


def load_annoy_idx_word2vec_vect32_wdw40_neg5_real_session_embedding(
    mode: str = "local",
):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_clicks_skipgram_vec32_wdw40_neg5.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_clicks_skipgram_vec32_wdw40_neg5.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(32, "angular")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index


def load_annoy_idx_word2vec_vect32_wdw30_neg10_real_session_embedding(
    mode: str = "local",
):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_clicks_skipgram_vec32_wdw30_neg10.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_clicks_skipgram_vec32_wdw30_neg10.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(32, "angular")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index


def load_annoy_idx_word2vec_cart_vect32_wdw45_neg5_real_session_embedding(
    mode: str = "local",
):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_carts_skipgram_vec32_wdw40_neg5.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_carts_skipgram_vec32_wdw40_neg5.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(32, "angular")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index


def load_annoy_idx_word2vec_buy_vect32_wdw15_neg7_embedding(
    mode: str = "local",
):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_buy2buy_skipgram_vec32_wdw15_neg7.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_buy2buy_skipgram_vec32_wdw15_neg7.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(32, "angular")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index


def load_annoy_idx_word2vec_vect32_wdw3_embedding(
    mode: str = "local",
):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_clicks_skipgram_vec32_wdw3.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_skipgram_clicks_vec32_wdw3.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")

    # create annoy index for search nn
    aid2idx = {aid: i for i, aid in enumerate(kvectors.index_to_key)}
    index = AnnoyIndex(32, "angular")

    for aid, idx in aid2idx.items():
        index.add_item(aid, kvectors.vectors[idx])

    # build annoy index
    index.build(NTREE)

    return index


def load_word2vec_vect32_wdw3_embedding(mode: str = "local"):

    if mode == "local":
        emd_path = get_local_word2vec_dir()
        filepath = f"{emd_path}/word2vec_local_clicks_skipgram_vec32_wdw3.kv"
    else:
        emd_path = get_scoring_word2vec_dir()
        filepath = f"{emd_path}/word2vec_scoring_skipgram_clicks_vec32_wdw3.kv"

    # load keyed vectors
    kvectors = KeyedVectors.load(filepath, mmap="r")
    return kvectors
