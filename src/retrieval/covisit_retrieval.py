import pandas as pd
import gc
import numpy as np
import itertools
from tqdm import tqdm
from collections import Counter
from src.utils.data import (
    get_top15_covisitation_buys,
    get_top15_covisitation_buy2buy,
    get_top20_covisitation_click,
)
from src.utils.constants import (
    get_processed_local_validation_dir,
    check_directory,
    get_data_output_dir,
)
from src.metrics.submission_evaluation import measure_recall
from src.utils.logger import get_logger

DATA_DIR = get_processed_local_validation_dir()
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
logging.info("top carts in validation set")
top_carts = df_val.loc[df_val["type"] == 1, "aid"].value_counts().index.values[:20]
logging.info("top orders in validation set")
top_orders = df_val.loc[df_val["type"] == 2, "aid"].value_counts().index.values[:20]

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
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
    top_clicks: np.ndarray,
    covisit_click: dict,
):
    """
    covisit_click is dict of aid as key and list of suggested aid as value
    """
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    sessions = []
    candidates = []
    for session, aids in tqdm(ses2aids.items()):
        unique_aids = set(aids)
        types = ses2types[session]

        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]
            candidate = [k for k, v in aids_temp.most_common(n_candidate)]

        else:
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
            result = list(unique_aids) + top_aids2[: n_candidate - len(unique_aids)]

            # # USE TOP20 TEST CLICKS
            candidate = result + list(top_clicks)[: n_candidate - len(result)]

        # append to list result
        sessions.append(session)
        candidates.append(candidate)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"

    return result_series


def suggest_carts(
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
    top_carts: np.ndarray,
    covisit_click: dict,
    covisit_buys: dict,
):
    """
    covisit_click is dict of aid as key and list of suggested aid as value
    """
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    sessions = []
    candidates = []
    for session, aids in tqdm(ses2aids.items()):
        unique_buys = []
        unique_aids = set(aids)
        types = ses2types[session]
        for ix, aid in enumerate(aids):
            curr_type = types[ix]
            if (curr_type == 0) or (curr_type == 1):
                unique_buys.append(aid)

        unique_buys = set(unique_buys)

        # RERANK CANDIDATES USING WEIGHTS
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

            candidate = [k for k, v in aids_temp.most_common(n_candidate)]

        else:
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
            result = list(unique_aids) + top_aids2[: n_candidate - len(unique_aids)]

            # USE TOP20 TEST ORDERS
            candidate = result + list(top_carts)[: n_candidate - len(result)]

        # append to list result
        sessions.append(session)
        candidates.append(candidate)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"

    return result_series


def suggest_buys(
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
    top_orders: np.ndarray,
    covisit_buys: dict,
    covisit_buy2buy: dict,
):
    """
    covisit_click is dict of aid as key and list of suggested aid as value
    """
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    sessions = []
    candidates = []
    for session, aids in tqdm(ses2aids.items()):
        unique_buys = []
        unique_aids = set(aids)
        types = ses2types[session]
        for ix, aid in enumerate(aids):
            curr_type = types[ix]
            if (curr_type == 1) or (curr_type == 2):
                unique_buys.append(aid)

        unique_buys = set(unique_buys)

        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= 20:
            weights = np.logspace(0.5, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()

            # Rerank based on repeat items and types of items
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]

            # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
            aids3 = list(
                itertools.chain(
                    *[
                        covisit_buy2buy[aid]
                        for aid in unique_buys
                        if aid in covisit_buy2buy
                    ]
                )
            )

            for aid in aids3:
                aids_temp[aid] += 0.1  # type: ignore

            candidate = [k for k, v in aids_temp.most_common(n_candidate)]

        else:
            # USE "CART ORDER" CO-VISITATION MATRIX
            aids2 = list(
                itertools.chain(
                    *[covisit_buys[aid] for aid in unique_aids if aid in covisit_buys]
                )
            )
            # USE "BUY2BUY" CO-VISITATION MATRIX
            aids3 = list(
                itertools.chain(
                    *[
                        covisit_buy2buy[aid]
                        for aid in unique_buys
                        if aid in covisit_buy2buy
                    ]
                )
            )
            # RERANK CANDIDATES
            top_aids2 = [
                aid2
                for aid2, cnt in Counter(aids2 + aids3).most_common(n_candidate)
                if aid2 not in unique_aids
            ]
            result = list(unique_aids) + top_aids2[: n_candidate - len(unique_aids)]
            # # USE TOP20 TEST ORDERS
            candidate = result + list(top_orders)[: n_candidate - len(result)]

        # append to list result
        sessions.append(session)
        candidates.append(candidate)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"

    return result_series


logging.info("start of suggesting clicks")
logging.info("sort session by ts ascendingly")
df_val = df_val.sort_values(["session", "ts"])
# create dict of session_id: list of aid/ts/type
logging.info("create ses2aids")
ses2aids = df_val.groupby("session")["aid"].apply(list).to_dict()
logging.info("create ses2types")
ses2types = df_val.groupby("session")["type"].apply(list).to_dict()

logging.info("start of suggesting clicks")
pred_df_clicks = suggest_clicks(
    n_candidate=30,
    ses2aids=ses2aids,
    ses2types=ses2types,
    top_clicks=top_clicks,
    covisit_click=top_20_clicks,
)
logging.info("end of suggesting clicks")

logging.info("start of suggesting carts")
pred_df_carts = suggest_carts(
    n_candidate=30,
    ses2aids=ses2aids,
    ses2types=ses2types,
    top_carts=top_carts,
    covisit_click=top_20_clicks,
    covisit_buys=top_15_buys,
)

logging.info("end of suggesting carts")

logging.info("start of suggesting buys")
pred_df_buys = suggest_buys(
    n_candidate=30,
    ses2aids=ses2aids,
    ses2types=ses2types,
    top_orders=top_orders,
    covisit_buy2buy=top_15_buy2buy,
    covisit_buys=top_15_buys,
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
measure_recall(df_pred=pred_df, df_truth=test_labels, Ks=[20, 30])
# measure_recall(df_pred=pred_df, df_truth=test_labels, Ks=[20, 30, 40])

## Add suggest Cart and see Recall@20/30/40/50
# using co-visitation LB 565 & type_weight_multipliers = {0: 1, 1: 6, 2: 3}
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

# using co-visitation LB 576 & type_weight_multipliers = {0: 1, 1: 5, 2: 4}
# [2022-12-01 23:01:46,940] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5784456467376878
# [2022-12-01 23:01:46,941] {submission_evaluation.py:84} INFO - clicks hits@20 = 1015481 / gt@20 = 1755534
# [2022-12-01 23:01:46,987] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5784456467376878
# [2022-12-01 23:03:55,928] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.6165238205481992
# [2022-12-01 23:03:55,973] {submission_evaluation.py:84} INFO - carts hits@20 = 271309 / gt@20 = 576482
# [2022-12-01 23:03:55,973] {submission_evaluation.py:85} INFO - carts recall@20 = 0.47062874469627847
# [2022-12-01 23:04:15,843] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.8165440654879669
# [2022-12-01 23:04:15,845] {submission_evaluation.py:84} INFO - orders hits@20 = 228583 / gt@20 = 313303
# [2022-12-01 23:04:15,845] {submission_evaluation.py:85} INFO - orders recall@20 = 0.7295908433688794
# [2022-12-01 23:04:15,845] {submission_evaluation.py:87} INFO - =============
# [2022-12-01 23:04:15,845] {submission_evaluation.py:88} INFO - Overall Recall@20 = 0.636787694103979


# using co-visitation LB 565 & type_weight_multipliers = {0: 1, 1: 5, 2: 4}
# [2022-12-01 23:18:50,550] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5150580962829544
# [2022-12-01 23:18:50,555] {submission_evaluation.py:84} INFO - clicks hits@20 = 904202 / gt@20 = 1755534
# [2022-12-01 23:18:50,555] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5150580962829544
# [2022-12-01 23:20:47,708] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.5451942191722331
# [2022-12-01 23:20:47,712] {submission_evaluation.py:84} INFO - carts hits@20 = 233344 / gt@20 = 576482
# [2022-12-01 23:20:47,713] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4047723953219702
# [2022-12-01 23:21:06,483] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.7296748095481295
# [2022-12-01 23:21:06,488] {submission_evaluation.py:84} INFO - orders hits@20 = 202329 / gt@20 = 313303
# [2022-12-01 23:21:06,488] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6457933693580974
# [2022-12-01 23:21:06,488] {submission_evaluation.py:87} INFO - =============
# [2022-12-01 23:21:06,488] {submission_evaluation.py:88} INFO - Overall Recall@20 = 0.5604135498397449

# using co-visitation LB 575 & type_weight_multipliers = {0: 1, 1: 6, 2: 3}
# [2022-12-01 23:37:06,869] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5784439378559458
# [2022-12-01 23:37:06,873] {submission_evaluation.py:84} INFO - clicks hits@20 = 1015478 / gt@20 = 1755534
# [2022-12-01 23:37:06,873] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5784439378559458
# [2022-12-01 23:39:04,591] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.6084399937475479
# [2022-12-01 23:39:04,595] {submission_evaluation.py:84} INFO - carts hits@20 = 267314 / gt@20 = 576482
# [2022-12-01 23:39:04,595] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4636987798404807
# [2022-12-01 23:39:23,173] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.7777221391635168
# [2022-12-01 23:39:23,174] {submission_evaluation.py:84} INFO - orders hits@20 = 217243 / gt@20 = 313303
# [2022-12-01 23:39:23,174] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6933958500237789
# [2022-12-01 23:39:23,174] {submission_evaluation.py:87} INFO - =============
# [2022-12-01 23:39:23,174] {submission_evaluation.py:88} INFO - Overall Recall@20 = 0.6129915377520061

# 2022-12-01 -> decide using co-visitation LB 565 & type_weight_multipliers = {0: 1, 1: 6, 2: 3}
