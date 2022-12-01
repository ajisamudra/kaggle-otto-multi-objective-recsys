import pandas as pd
import gc
import numpy as np
import itertools
from collections import Counter
from src.utils.data import (
    get_top15_covisitation_buys,
    get_top15_covisitation_buy2buy,
    get_top20_covisitation_click,
)
from src.utils.constants import (
    get_small_local_validation_dir,
    check_directory,
    get_data_output_dir,
)
from src.utils.logger import get_logger

DATA_DIR = get_small_local_validation_dir()
logging = get_logger()

# logging.info("read train parquet")
# df_train = pd.read_parquet(DATA_DIR / "train.parquet")
logging.info("read val parquet")
df_val = pd.read_parquet(DATA_DIR / "test.parquet")


# candidate generation
# get top 20 clicks & buys in validation set
# TOP CLICKS AND ORDERS IN TEST
logging.info("top clicks in validation set")
top_clicks = df_val.loc[df_val["type"] == 0, "aid"].value_counts().index.values[:20]
logging.info("top orders in validation set")
top_orders = df_val.loc[df_val["type"] == 2, "aid"].value_counts().index.values[:20]
logging.info("top carts in validation set")
top_carts = df_val.loc[df_val["type"] == 1, "aid"].value_counts().index.values[:20]

logging.info("read covisitation buys")
top_15_buys = get_top15_covisitation_buys()
logging.info("read covisitation buy2buy")
top_15_buy2buy = get_top15_covisitation_buy2buy()
logging.info("read covisitation click")
top_20_clicks = get_top20_covisitation_click()


logging.info("Here are size of our 3 co-visitation matrices:")
logging.info(f"{len(top_20_clicks)}, {len(top_15_buy2buy)}, {len(top_15_buys)}")

type_weight_multipliers = {0: 1, 1: 6, 2: 3}


def suggest_clicks(
    df: pd.DataFrame, n_candidate: int, top_clicks: np.ndarray, covisit_click: dict
):
    """
    covisit_click is dict of aid as key and list of suggested aid as value
    """
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    # 0 : click, 1: cart, 2: buy
    # USE USER HISTORY AIDS AND TYPES
    aids = df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1]))
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids) >= 20:
        weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k, v in aids_temp.most_common(n_candidate)]
        return sorted_aids
    # USE "CLICKS" CO-VISITATION MATRIX
    aids2 = list(
        itertools.chain(
            *[covisit_click[aid] for aid in unique_aids if aid in covisit_click]
        )
    )
    # RERANK CANDIDATES
    top_aids2 = [
        aid2
        for aid2, cnt in Counter(aids2).most_common(n_candidate)
        if aid2 not in unique_aids
    ]
    result = unique_aids + top_aids2[: n_candidate - len(unique_aids)]
    # USE TOP20 TEST CLICKS
    return result + list(top_clicks)[: n_candidate - len(result)]


def suggest_carts(
    df: pd.DataFrame,
    n_candidate: int,
    top_carts: np.ndarray,
    covisit_click: dict,
    covisit_buys: dict,
):
    # User history aids and types
    aids = df.aid.tolist()
    types = df.type.tolist()

    # UNIQUE AIDS AND UNIQUE BUYS
    unique_aids = list(dict.fromkeys(aids[::-1]))
    df = df.loc[(df["type"] == 0) | (df["type"] == 1)]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))

    # Rerank candidates using weights
    if len(unique_aids) >= 20:
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()

        # Rerank based on repeat items and types of items
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]

        # Rerank candidates using"top_20_carts" co-visitation matrix
        aids2 = list(
            itertools.chain(
                *[covisit_buys[aid] for aid in unique_buys if aid in covisit_buys]
            )
        )
        for aid in aids2:
            aids_temp[aid] += 0.1
        sorted_aids = [k for k, v in aids_temp.most_common(n_candidate)]
        return sorted_aids

    # Use "cart order" and "clicks" co-visitation matrices
    aids1 = list(
        itertools.chain(
            *[covisit_click[aid] for aid in unique_aids if aid in covisit_click]
        )
    )
    aids2 = list(
        itertools.chain(
            *[covisit_buys[aid] for aid in unique_aids if aid in covisit_buys]
        )
    )

    # RERANK CANDIDATES
    top_aids2 = [
        aid2
        for aid2, cnt in Counter(aids1 + aids2).most_common(n_candidate)
        if aid2 not in unique_aids
    ]
    result = unique_aids + top_aids2[: n_candidate - len(unique_aids)]

    # USE TOP20 TEST ORDERS
    return result + list(top_carts)[: n_candidate - len(result)]


def suggest_buys(
    df: pd.DataFrame,
    n_candidate: int,
    top_orders: np.ndarray,
    covisit_buys: dict,
    covisit_buy2buy: dict,
):
    """
    covisit_click is dict of aid as key and list of suggested aid as value
    """
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    # 0 : click, 1: cart, 2: buy
    # USE USER HISTORY AIDS AND TYPES
    aids = df.aid.tolist()
    types = df.type.tolist()
    # UNIQUE AIDS AND UNIQUE BUYS
    unique_aids = list(dict.fromkeys(aids[::-1]))
    df = df.loc[(df["type"] == 1) | (df["type"] == 2)]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids) >= 20:
        weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
        aids_temp = Counter()
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid, w, t in zip(aids, weights, types):
            aids_temp[aid] += w * type_weight_multipliers[t]
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(
            itertools.chain(
                *[covisit_buy2buy[aid] for aid in unique_buys if aid in covisit_buy2buy]
            )
        )

        for aid in aids3:
            aids_temp[aid] += 0.1

        sorted_aids = [k for k, v in aids_temp.most_common(n_candidate)]
        return sorted_aids
    # USE "CART ORDER" CO-VISITATION MATRIX
    aids2 = list(
        itertools.chain(
            *[covisit_buys[aid] for aid in unique_aids if aid in covisit_buys]
        )
    )
    # USE "BUY2BUY" CO-VISITATION MATRIX
    aids3 = list(
        itertools.chain(
            *[covisit_buy2buy[aid] for aid in unique_buys if aid in covisit_buy2buy]
        )
    )
    # RERANK CANDIDATES
    top_aids2 = [
        aid2
        for aid2, cnt in Counter(aids2 + aids3).most_common(n_candidate)
        if aid2 not in unique_aids
    ]
    result = unique_aids + top_aids2[: n_candidate - len(unique_aids)]
    # USE TOP20 TEST ORDERS
    return result + list(top_orders)[: n_candidate - len(result)]


logging.info("start of suggesting clicks")
pred_df_clicks = (
    df_val.sort_values(["session", "ts"])
    .groupby(["session"])
    .apply(
        lambda x: suggest_clicks(
            df=x, n_candidate=50, top_clicks=top_clicks, covisit_click=top_20_clicks
        )
    )
)
logging.info("end of suggesting clicks")

logging.info("start of suggesting carts")
pred_df_carts = (
    df_val.sort_values(["session", "ts"])
    .groupby(["session"])
    .apply(
        lambda x: suggest_carts(
            df=x,
            n_candidate=50,
            top_carts=top_carts,
            covisit_click=top_20_clicks,
            covisit_buys=top_15_buys,
        )
    )
)
logging.info("end of suggesting carts")

logging.info("start of suggesting buys")
pred_df_buys = (
    df_val.sort_values(["session", "ts"])
    .groupby(["session"])
    .apply(
        lambda x: suggest_buys(
            df=x,
            n_candidate=50,
            top_orders=top_orders,
            covisit_buy2buy=top_15_buy2buy,
            covisit_buys=top_15_buys,
        )
    )
)
logging.info("end of suggesting buys")

del df_val
gc.collect()

logging.info("create predicition df for click")
clicks_pred_df = pd.DataFrame(
    pred_df_clicks.add_suffix("_clicks"), columns=["labels"]
).reset_index()
logging.info("create predicition df for cart")
carts_pred_df = pd.DataFrame(
    pred_df_carts.add_suffix("_carts"), columns=["labels"]
).reset_index()
logging.info("create predicition df for order")
orders_pred_df = pd.DataFrame(
    pred_df_buys.add_suffix("_orders"), columns=["labels"]
).reset_index()

del pred_df_clicks, pred_df_carts, pred_df_buys
gc.collect()


logging.info("concat all predicitions")
pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
pred_df.columns = ["session_type", "labels"]
pred_df["labels"] = pred_df.labels.apply(lambda x: " ".join(map(str, x)))
logging.info(pred_df.head())

OUTPUT_DIR = get_data_output_dir()
filepath = OUTPUT_DIR / "covisit_retrieval"
check_directory(filepath)
logging.info("save validation prediction")
pred_df.to_csv(filepath / "validation_preds.csv", index=False)


logging.info("start computing metrics")
# COMPUTE METRIC
test_labels = pd.read_parquet(DATA_DIR / "test_labels.parquet")
# K = 20
for K in [20, 30, 40, 50]:
    score = 0
    weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
    for t in ["clicks", "carts", "orders"]:
        sub = pred_df.loc[pred_df.session_type.str.contains(t)].copy()
        sub["session"] = sub.session_type.apply(lambda x: int(x.split("_")[0]))
        sub.labels = sub.labels.apply(lambda x: [int(i) for i in x.split(" ")[:K]])
        tmp_test_labels = test_labels.loc[test_labels["type"] == t]
        tmp_test_labels = tmp_test_labels.merge(sub, how="left", on=["session"])
        tmp_test_labels["hits"] = tmp_test_labels.apply(
            lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1
        )
        tmp_test_labels["gt_count"] = tmp_test_labels.ground_truth.str.len().clip(0, K)
        n_hits = tmp_test_labels["hits"].sum()
        n_gt = tmp_test_labels["gt_count"].sum()
        recall = tmp_test_labels["hits"].sum() / tmp_test_labels["gt_count"].sum()
        score += weights[t] * recall
        logging.info(f"{t} hits@{K} = {n_hits} / gt@{K} = {n_gt}")
        logging.info(f"{t} recall@{K} = {recall}")

    logging.info("=============")
    logging.info(f"Overall Recall@{K} = {score}")
    logging.info("=============")

# recall @20 0.5646320148830121
# recall @40

# recall @40 with n_cand 40
# [2022-12-01 01:49:47,313] {covisit_retrieval.py:201} INFO - save validation prediction
# [2022-12-01 01:50:15,764] {covisit_retrieval.py:205} INFO - start computing metrics
# [2022-12-01 01:51:37,307] {covisit_retrieval.py:225} INFO - clicks hits = 922638 / gt = 1755534
# [2022-12-01 01:51:37,311] {covisit_retrieval.py:226} INFO - clicks recall = 0.5255597442145808
# [2022-12-01 01:52:42,917] {covisit_retrieval.py:225} INFO - carts hits = 235973 / gt = 580104
# [2022-12-01 01:52:42,922] {covisit_retrieval.py:226} INFO - carts recall = 0.4067770606649842
# [2022-12-01 01:53:02,044] {covisit_retrieval.py:225} INFO - orders hits = 203269 / gt = 314009
# [2022-12-01 01:53:02,046] {covisit_retrieval.py:226} INFO - orders recall = 0.6473349489982771
# [2022-12-01 01:53:02,046] {covisit_retrieval.py:228} INFO - =============
# [2022-12-01 01:53:02,046] {covisit_retrieval.py:229} INFO - Overall Recall = 0.5629900620199196

# Recall@20 with n_cand 20
# [2022-12-01 07:31:45,508] {covisit_retrieval.py:201} INFO - save validation prediction
# [2022-12-01 07:32:08,564] {covisit_retrieval.py:205} INFO - start computing metrics
# [2022-12-01 07:33:24,129] {covisit_retrieval.py:225} INFO - clicks hits = 922638 / gt = 1755534
# [2022-12-01 07:33:24,133] {covisit_retrieval.py:226} INFO - clicks recall = 0.5255597442145808
# [2022-12-01 07:34:19,542] {covisit_retrieval.py:225} INFO - carts hits = 235973 / gt = 576482
# [2022-12-01 07:34:19,544] {covisit_retrieval.py:226} INFO - carts recall = 0.4093328152483512
# [2022-12-01 07:34:36,650] {covisit_retrieval.py:225} INFO - orders hits = 203269 / gt = 313303
# [2022-12-01 07:34:36,654] {covisit_retrieval.py:226} INFO - orders recall = 0.6487936598117477
# [2022-12-01 07:34:36,654] {covisit_retrieval.py:228} INFO - =============
# [2022-12-01 07:34:36,655] {covisit_retrieval.py:229} INFO - Overall Recall = 0.5646320148830121
# [2022-12-01 07:34:36,655] {covisit_retrieval.py:230} INFO - =============


## Add suggest Cart and see Recall@20/30/40/50
# [2022-12-01 10:00:36,791] {covisit_retrieval.py:287} INFO - start computing metrics
# [2022-12-01 10:01:24,617] {covisit_retrieval.py:308} INFO - clicks hits@20 = 922665 / gt@20 = 1755534
# [2022-12-01 10:01:24,623] {covisit_retrieval.py:309} INFO - clicks recall@20 = 0.5255751241502585
# [2022-12-01 10:05:02,422] {covisit_retrieval.py:308} INFO - carts hits@20 = 236116 / gt@20 = 576482
# [2022-12-01 10:05:02,426] {covisit_retrieval.py:309} INFO - carts recall@20 = 0.40958087156233847
# [2022-12-01 10:05:34,128] {covisit_retrieval.py:308} INFO - orders hits@20 = 203307 / gt@20 = 313303
# [2022-12-01 10:05:34,134] {covisit_retrieval.py:309} INFO - orders recall@20 = 0.6489149481492357
# [2022-12-01 10:05:34,138] {covisit_retrieval.py:311} INFO - =============
# [2022-12-01 10:05:34,139] {covisit_retrieval.py:312} INFO - Overall Recall@20 = 0.5647807427732687
# [2022-12-01 10:05:34,139] {covisit_retrieval.py:313} INFO - =============
# [2022-12-01 10:09:09,857] {covisit_retrieval.py:308} INFO - clicks hits@30 = 963441 / gt@30 = 1755534
# [2022-12-01 10:09:09,867] {covisit_retrieval.py:309} INFO - clicks recall@30 = 0.5488022447870562
# [2022-12-01 10:12:51,939] {covisit_retrieval.py:308} INFO - carts hits@30 = 249539 / gt@30 = 579204
# [2022-12-01 10:12:51,951] {covisit_retrieval.py:309} INFO - carts recall@30 = 0.4308309334880284
# [2022-12-01 10:13:40,078] {covisit_retrieval.py:308} INFO - orders hits@30 = 207521 / gt@30 = 313960
# [2022-12-01 10:13:40,081] {covisit_retrieval.py:309} INFO - orders recall@30 = 0.6609791056185501
# [2022-12-01 10:13:40,082] {covisit_retrieval.py:311} INFO - =============
# [2022-12-01 10:13:40,082] {covisit_retrieval.py:312} INFO - Overall Recall@30 = 0.5807169678962442
# [2022-12-01 10:13:40,082] {covisit_retrieval.py:313} INFO - =============
# [2022-12-01 10:19:07,326] {covisit_retrieval.py:308} INFO - clicks hits@40 = 981827 / gt@40 = 1755534
# [2022-12-01 10:19:07,337] {covisit_retrieval.py:309} INFO - clicks recall@40 = 0.5592754113563166
# [2022-12-01 10:20:09,807] {covisit_retrieval.py:308} INFO - carts hits@40 = 257465 / gt@40 = 580104
# [2022-12-01 10:20:09,811] {covisit_retrieval.py:309} INFO - carts recall@40 = 0.4438255898942259
# [2022-12-01 10:25:33,889] {covisit_retrieval.py:308} INFO - orders hits@40 = 209889 / gt@40 = 314009
# [2022-12-01 10:25:33,921] {covisit_retrieval.py:309} INFO - orders recall@40 = 0.6684171472792181
# [2022-12-01 10:25:33,922] {covisit_retrieval.py:311} INFO - =============
# [2022-12-01 10:25:33,922] {covisit_retrieval.py:312} INFO - Overall Recall@40 = 0.5901255064714303
# [2022-12-01 10:25:33,922] {covisit_retrieval.py:313} INFO - =============
# [2022-12-01 10:27:40,811] {covisit_retrieval.py:308} INFO - clicks hits@50 = 991030 / gt@50 = 1755534
# [2022-12-01 10:27:40,816] {covisit_retrieval.py:309} INFO - clicks recall@50 = 0.5645176909134201
# [2022-12-01 10:33:43,505] {covisit_retrieval.py:308} INFO - carts hits@50 = 262856 / gt@50 = 580492
# [2022-12-01 10:33:43,518] {covisit_retrieval.py:309} INFO - carts recall@50 = 0.45281588721291594
# [2022-12-01 10:35:27,219] {covisit_retrieval.py:308} INFO - orders hits@50 = 211363 / gt@50 = 314021
# [2022-12-01 10:35:27,224] {covisit_retrieval.py:309} INFO - orders recall@50 = 0.6730855579722376
# [2022-12-01 10:35:27,225] {covisit_retrieval.py:311} INFO - =============
# [2022-12-01 10:35:27,225] {covisit_retrieval.py:312} INFO - Overall Recall@50 = 0.5961478700385593
# [2022-12-01 10:35:27,225] {covisit_retrieval.py:313} INFO - =============
