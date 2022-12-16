import pandas as pd
import gc
import numpy as np
import itertools
from tqdm import tqdm
from collections import Counter
from annoy import AnnoyIndex
from src.utils.constants import (
    get_processed_local_validation_dir,
    check_directory,
    get_data_output_dir,
)
from src.utils.matrix_factorization import load_annoy_idx_matrix_fact_embedding
from src.metrics.submission_evaluation import measure_recall
from src.utils.logger import get_logger

DATA_DIR = get_processed_local_validation_dir()
logging = get_logger()

######### CANDIDATES GENERATION FUNCTION


def suggest_matrix_fact(
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
    matrix_fact_idx: AnnoyIndex,
):
    """
    covisit_click is dict of aid as key and list of suggested aid as value
    """
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    sessions = []
    candidates = []
    for session, aids in tqdm(ses2aids.items()):
        # unique_aids = set(aids)
        unique_aids = list(dict.fromkeys(aids[::-1]))
        types = ses2types[session]

        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]
            candidate = [k for k, v in aids_temp.most_common(20)]
            mf_candidate = matrix_fact_idx.get_nns_by_item(
                unique_aids[0], n=n_candidate - 20
            )
            candidate.extend(mf_candidate)

        else:
            candidate = list(unique_aids)
            mf_candidate = matrix_fact_idx.get_nns_by_item(
                unique_aids[0], n=n_candidate - (len(candidate))
            )
            candidate.extend(mf_candidate)

        # append to list result
        sessions.append(session)
        candidates.append(mf_candidate)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"

    return result_series


######### CANDIDATES GENERATION FUNCTION

if __name__ == "__main__":

    logging.info("read val parquet")
    df_val = pd.read_parquet(DATA_DIR / "test.parquet")

    # candidate generation
    logging.info("read matrix fact index")
    matrix_fact_idx = load_annoy_idx_matrix_fact_embedding()

    logging.info("Here are size of our 3 co-visitation matrices:")
    # logging.info(f"{len(top_20_clicks)}, {len(top_15_buy2buy)}, {len(top_15_buys)}")

    logging.info("start of suggesting clicks")
    logging.info("sort session by ts ascendingly")
    df_val = df_val.sort_values(["session", "ts"])
    # sample session id
    lucky_sessions = df_val.drop_duplicates(["session"]).sample(frac=0.1)["session"]
    df_val = df_val[df_val.session.isin(lucky_sessions)]
    # create dict of session_id: list of aid/ts/type
    logging.info("create ses2aids")
    ses2aids = df_val.groupby("session")["aid"].apply(list).to_dict()
    logging.info("create ses2types")
    ses2types = df_val.groupby("session")["type"].apply(list).to_dict()

    logging.info("start of suggesting mf")
    pred_df_mf = suggest_matrix_fact(
        n_candidate=40,
        ses2aids=ses2aids,
        ses2types=ses2types,
        matrix_fact_idx=matrix_fact_idx,
    )

    logging.info("end of suggesting mf")

    del df_val
    gc.collect()

    logging.info("create predicition df for click")
    clicks_pred_df = pd.DataFrame(
        pred_df_mf.add_suffix("_clicks"), columns=["labels"]
    ).reset_index()
    logging.info("create predicition df for cart")
    carts_pred_df = pd.DataFrame(
        pred_df_mf.add_suffix("_carts"), columns=["labels"]
    ).reset_index()
    logging.info("create predicition df for order")
    orders_pred_df = pd.DataFrame(
        pred_df_mf.add_suffix("_orders"), columns=["labels"]
    ).reset_index()

    # del pred_df_clicks, pred_df_carts, pred_df_buys
    # gc.collect()

    del pred_df_mf
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
    pred_df.to_csv(filepath / "validation_preds_mf.csv", index=False)

    logging.info("start computing metrics")
    # COMPUTE METRIC
    test_labels = pd.read_parquet(DATA_DIR / "test_labels.parquet")
    test_labels = test_labels[test_labels.session.isin(lucky_sessions)]
    measure_recall(df_pred=pred_df, df_truth=test_labels, Ks=[20, 40])

# covisit retrieval - ver 9
# [2022-12-13 09:11:28,984] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5260462058837938
# [2022-12-13 09:11:28,988] {submission_evaluation.py:84} INFO - clicks hits@20 = 923492 / gt@20 = 1755534
# [2022-12-13 09:11:28,989] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5260462058837938
# [2022-12-13 09:13:57,767] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.5510988330711586
# [2022-12-13 09:13:57,779] {submission_evaluation.py:84} INFO - carts hits@20 = 236289 / gt@20 = 576482
# [2022-12-13 09:13:57,779] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4098809676624769
# [2022-12-13 09:14:22,426] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.7329741134290875
# [2022-12-13 09:14:22,431] {submission_evaluation.py:84} INFO - orders hits@20 = 203307 / gt@20 = 313303
# [2022-12-13 09:14:22,431] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6489149481492357
# [2022-12-13 09:14:22,433] {submission_evaluation.py:91} INFO - =============
# [2022-12-13 09:14:22,433] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5649178797766639
# [2022-12-13 09:14:22,433] {submission_evaluation.py:93} INFO - =============
# [2022-12-13 09:15:11,586] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.5598233927682403
# [2022-12-13 09:15:11,594] {submission_evaluation.py:84} INFO - clicks hits@40 = 982789 / gt@40 = 1755534
# [2022-12-13 09:15:11,594] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.5598233927682403
# [2022-12-13 09:18:10,124] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.5858564962351568
# [2022-12-13 09:18:10,136] {submission_evaluation.py:84} INFO - carts hits@40 = 257569 / gt@40 = 576482
# [2022-12-13 09:18:10,137] {submission_evaluation.py:85} INFO - carts recall@40 = 0.4467945226390416
# [2022-12-13 09:18:44,544] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.7493155696957251
# [2022-12-13 09:18:44,548] {submission_evaluation.py:84} INFO - orders hits@40 = 209880 / gt@40 = 313303
# [2022-12-13 09:18:44,548] {submission_evaluation.py:85} INFO - orders recall@40 = 0.6698946387363032
# [2022-12-13 09:18:44,548] {submission_evaluation.py:91} INFO - =============
# [2022-12-13 09:18:44,548] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.5919574793103184
# [2022-12-13 09:18:44,548] {submission_evaluation.py:93} INFO - =============
# [2022-12-13 09:22:25,874] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@80 = 0.5709277063275334
# [2022-12-13 09:22:25,896] {submission_evaluation.py:84} INFO - clicks hits@80 = 1002283 / gt@80 = 1755534
# [2022-12-13 09:22:25,897] {submission_evaluation.py:85} INFO - clicks recall@80 = 0.5709277063275334
# [2022-12-13 09:23:22,731] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@80 = 0.6054709422835255
# [2022-12-13 09:23:22,756] {submission_evaluation.py:84} INFO - carts hits@80 = 271427 / gt@80 = 576482
# [2022-12-13 09:23:22,756] {submission_evaluation.py:85} INFO - carts recall@80 = 0.4708334345218064
# [2022-12-13 09:27:24,936] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@80 = 0.7570737726548159
# [2022-12-13 09:27:24,948] {submission_evaluation.py:84} INFO - orders hits@80 = 213474 / gt@80 = 313303
# [2022-12-13 09:27:24,948] {submission_evaluation.py:85} INFO - orders recall@80 = 0.6813659620239831
# [2022-12-13 09:27:24,950] {submission_evaluation.py:91} INFO - =============
# [2022-12-13 09:27:24,950] {submission_evaluation.py:92} INFO - Overall Recall@80 = 0.6071623782036851
# [2022-12-13 09:27:24,950] {submission_evaluation.py:93} INFO - =============
# [2022-12-13 09:28:49,331] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@100 = 0.572093163675554
# [2022-12-13 09:28:49,337] {submission_evaluation.py:84} INFO - clicks hits@100 = 1004329 / gt@100 = 1755534
# [2022-12-13 09:28:49,338] {submission_evaluation.py:85} INFO - clicks recall@100 = 0.572093163675554
# [2022-12-13 09:29:53,133] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@100 = 0.6089995011782157
# [2022-12-13 09:29:53,143] {submission_evaluation.py:84} INFO - carts hits@100 = 274249 / gt@100 = 576482
# [2022-12-13 09:29:53,143] {submission_evaluation.py:85} INFO - carts recall@100 = 0.47572864373909335
# [2022-12-13 09:33:55,940] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@100 = 0.7582175713389592
# [2022-12-13 09:33:55,955] {submission_evaluation.py:84} INFO - orders hits@100 = 214111 / gt@100 = 313303
# [2022-12-13 09:33:55,955] {submission_evaluation.py:85} INFO - orders recall@100 = 0.6833991375760845
# [2022-12-13 09:33:55,957] {submission_evaluation.py:91} INFO - =============
# [2022-12-13 09:33:55,958] {submission_evaluation.py:92} INFO - Overall Recall@100 = 0.6099673920349341
# [2022-12-13 09:33:55,958] {submission_evaluation.py:93} INFO - =============

# covisit + matrix factorial retrieval
# [2022-12-16 14:03:40,568] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.24171587692973515
# [2022-12-16 14:03:40,568] {submission_evaluation.py:84} INFO - clicks hits@20 = 42447 / gt@20 = 175607
# [2022-12-16 14:03:40,568] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.24171587692973515
# [2022-12-16 14:03:42,072] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.35351825529035746
# [2022-12-16 14:03:42,072] {submission_evaluation.py:84} INFO - carts hits@20 = 13384 / gt@20 = 57377
# [2022-12-16 14:03:42,072] {submission_evaluation.py:85} INFO - carts recall@20 = 0.2332641999407428
# [2022-12-16 14:03:43,256] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.37837175538479934
# [2022-12-16 14:03:43,257] {submission_evaluation.py:84} INFO - orders hits@20 = 7546 / gt@20 = 31603
# [2022-12-16 14:03:43,257] {submission_evaluation.py:85} INFO - orders recall@20 = 0.2387747998607727
# [2022-12-16 14:03:43,257] {submission_evaluation.py:91} INFO - =============
# [2022-12-16 14:03:43,257] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.23741572759165996
# [2022-12-16 14:03:43,257] {submission_evaluation.py:93} INFO - =============
# [2022-12-16 14:03:46,338] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.24877140432898462
# [2022-12-16 14:03:46,340] {submission_evaluation.py:84} INFO - clicks hits@40 = 43686 / gt@40 = 175607
# [2022-12-16 14:03:46,340] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.24877140432898462
# [2022-12-16 14:03:48,766] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.3572121060897301
# [2022-12-16 14:03:48,766] {submission_evaluation.py:84} INFO - carts hits@40 = 13549 / gt@40 = 57377
# [2022-12-16 14:03:48,766] {submission_evaluation.py:85} INFO - carts recall@40 = 0.23613991669135717
# [2022-12-16 14:03:51,239] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.38102553261552796
# [2022-12-16 14:03:51,239] {submission_evaluation.py:84} INFO - orders hits@40 = 7605 / gt@40 = 31603
# [2022-12-16 14:03:51,239] {submission_evaluation.py:85} INFO - orders recall@40 = 0.24064171122994651
# [2022-12-16 14:03:51,239] {submission_evaluation.py:91} INFO - =============
# [2022-12-16 14:03:51,239] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.24010414217827353
# [2022-12-16 14:03:51,239] {submission_evaluation.py:93} INFO - =============
