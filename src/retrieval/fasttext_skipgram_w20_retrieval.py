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
from src.utils.fasttext import load_annoy_idx_fasttext_skipgram_wdw20_embedding
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
    logging.info("read embedding index")
    matrix_fact_idx = load_annoy_idx_fasttext_skipgram_wdw20_embedding()

    logging.info("start of suggesting")
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

    logging.info("start of suggesting")
    pred_df_mf = suggest_matrix_fact(
        n_candidate=60,
        ses2aids=ses2aids,
        ses2types=ses2types,
        matrix_fact_idx=matrix_fact_idx,
    )

    logging.info("end of suggesting")

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

    logging.info("start computing metrics")
    # COMPUTE METRIC
    test_labels = pd.read_parquet(DATA_DIR / "test_labels.parquet")
    test_labels = test_labels[test_labels.session.isin(lucky_sessions)]
    measure_recall(df_pred=pred_df, df_truth=test_labels, Ks=[20, 40, 60])

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

# covisit + word2vec 20wdw 5negative retrieval
# [2022-12-16 14:06:09,775] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.3212786438901578
# [2022-12-16 14:06:09,775] {submission_evaluation.py:84} INFO - clicks hits@20 = 56404 / gt@20 = 175561
# [2022-12-16 14:06:09,776] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3212786438901578
# [2022-12-16 14:06:10,941] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4024365826031879
# [2022-12-16 14:06:10,941] {submission_evaluation.py:84} INFO - carts hits@20 = 15679 / gt@20 = 57138
# [2022-12-16 14:06:10,941] {submission_evaluation.py:85} INFO - carts recall@20 = 0.2744058244950821
# [2022-12-16 14:06:12,288] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.41961585205982016
# [2022-12-16 14:06:12,290] {submission_evaluation.py:84} INFO - orders hits@20 = 8659 / gt@20 = 31254
# [2022-12-16 14:06:12,290] {submission_evaluation.py:85} INFO - orders recall@20 = 0.27705253727522877
# [2022-12-16 14:06:12,290] {submission_evaluation.py:91} INFO - =============
# [2022-12-16 14:06:12,291] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.28068113410267764
# [2022-12-16 14:06:12,291] {submission_evaluation.py:93} INFO - =============
# [2022-12-16 14:06:14,691] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.347816428477851
# [2022-12-16 14:06:14,692] {submission_evaluation.py:84} INFO - clicks hits@40 = 61063 / gt@40 = 175561
# [2022-12-16 14:06:14,692] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.347816428477851
# [2022-12-16 14:06:16,735] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4197144273310295
# [2022-12-16 14:06:16,735] {submission_evaluation.py:84} INFO - carts hits@40 = 16518 / gt@40 = 57138
# [2022-12-16 14:06:16,735] {submission_evaluation.py:85} INFO - carts recall@40 = 0.2890895726136722
# [2022-12-16 14:06:18,640] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.4337384771203976
# [2022-12-16 14:06:18,640] {submission_evaluation.py:84} INFO - orders hits@40 = 9024 / gt@40 = 31254
# [2022-12-16 14:06:18,640] {submission_evaluation.py:85} INFO - orders recall@40 = 0.2887310424265694
# [2022-12-16 14:06:18,640] {submission_evaluation.py:91} INFO - =============
# [2022-12-16 14:06:18,640] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.2947471400878284
# [2022-12-16 14:06:18,640] {submission_evaluation.py:93} INFO - =============

# fasttext skipgram wdw 20 neg5 minn 3 maxn 6 dist euclidean
# 2022-12-17 13:08:07,103] {submission_evaluation.py:84} INFO - clicks hits@20 = 57116 / gt@20 = 175437
# [2022-12-17 13:08:07,103] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.32556416263388
# [2022-12-17 13:08:08,612] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.40693836452259297
# [2022-12-17 13:08:08,612] {submission_evaluation.py:84} INFO - carts hits@20 = 16117 / gt@20 = 57881
# [2022-12-17 13:08:08,612] {submission_evaluation.py:85} INFO - carts recall@20 = 0.2784506141911854
# [2022-12-17 13:08:10,303] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4265100867631828
# [2022-12-17 13:08:10,303] {submission_evaluation.py:84} INFO - orders hits@20 = 8855 / gt@20 = 31118
# [2022-12-17 13:08:10,303] {submission_evaluation.py:85} INFO - orders recall@20 = 0.2845619898451057
# [2022-12-17 13:08:10,303] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 13:08:10,303] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.28682879442780707
# [2022-12-17 13:08:10,303] {submission_evaluation.py:93} INFO - =============
# [2022-12-17 13:08:13,212] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.36578942868380104
# [2022-12-17 13:08:13,212] {submission_evaluation.py:84} INFO - clicks hits@40 = 64173 / gt@40 = 175437
# [2022-12-17 13:08:13,212] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.36578942868380104
# [2022-12-17 13:08:15,696] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4323622122215737
# [2022-12-17 13:08:15,697] {submission_evaluation.py:84} INFO - carts hits@40 = 17474 / gt@40 = 57881
# [2022-12-17 13:08:15,697] {submission_evaluation.py:85} INFO - carts recall@40 = 0.30189526787719634
# [2022-12-17 13:08:18,091] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.44942336772091285
# [2022-12-17 13:08:18,091] {submission_evaluation.py:84} INFO - orders hits@40 = 9473 / gt@40 = 31118
# [2022-12-17 13:08:18,091] {submission_evaluation.py:85} INFO - orders recall@40 = 0.30442187801272574
# [2022-12-17 13:08:18,091] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 13:08:18,091] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.30980065003917445
# [2022-12-17 13:08:18,091] {submission_evaluation.py:93} INFO - =============
# [2022-12-17 13:08:21,883] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.3865547176479306
# [2022-12-17 13:08:21,884] {submission_evaluation.py:84} INFO - clicks hits@60 = 67816 / gt@60 = 175437
# [2022-12-17 13:08:21,884] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.3865547176479306
# [2022-12-17 13:08:24,624] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.445396986812266
# [2022-12-17 13:08:24,624] {submission_evaluation.py:84} INFO - carts hits@60 = 18141 / gt@60 = 57881
# [2022-12-17 13:08:24,625] {submission_evaluation.py:85} INFO - carts recall@60 = 0.31341891121438814
# [2022-12-17 13:08:27,603] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.45935840741628176
# [2022-12-17 13:08:27,603] {submission_evaluation.py:84} INFO - orders hits@60 = 9737 / gt@60 = 31118
# [2022-12-17 13:08:27,603] {submission_evaluation.py:85} INFO - orders recall@60 = 0.31290571373481585
# [2022-12-17 13:08:27,603] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 13:08:27,603] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.32042457336999897
# [2022-12-17 13:08:27,603] {submission_evaluation.py:93} INFO - =============

# fasttext skipgram wdw 20 neg5 minn 3 maxn 6 dist angular
# [2022-12-20 10:35:17,184] {submission_evaluation.py:84} INFO - clicks hits@20 = 57549 / gt@20 = 175600
# [2022-12-20 10:35:17,184] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3277277904328018
# [2022-12-20 10:35:19,450] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.40470094606224977
# [2022-12-20 10:35:19,451] {submission_evaluation.py:84} INFO - carts hits@20 = 16087 / gt@20 = 58260
# [2022-12-20 10:35:19,451] {submission_evaluation.py:85} INFO - carts recall@20 = 0.2761242705115002
# [2022-12-20 10:35:23,536] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4226890151976901
# [2022-12-20 10:35:23,537] {submission_evaluation.py:84} INFO - orders hits@20 = 8785 / gt@20 = 31232
# [2022-12-20 10:35:23,537] {submission_evaluation.py:85} INFO - orders recall@20 = 0.28128201844262296
# [2022-12-20 10:35:23,537] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 10:35:23,537] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.284379271262304
# [2022-12-20 10:35:23,537] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 10:35:28,390] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.36702164009111615
# [2022-12-20 10:35:28,390] {submission_evaluation.py:84} INFO - clicks hits@40 = 64449 / gt@40 = 175600
# [2022-12-20 10:35:28,390] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.36702164009111615
# [2022-12-20 10:35:33,007] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4311184954946002
# [2022-12-20 10:35:34,950] {submission_evaluation.py:84} INFO - carts hits@40 = 17460 / gt@40 = 58260
# [2022-12-20 10:35:35,021] {submission_evaluation.py:85} INFO - carts recall@40 = 0.2996910401647786
# [2022-12-20 10:35:39,469] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.44446718430816307
# [2022-12-20 10:35:39,469] {submission_evaluation.py:84} INFO - orders hits@40 = 9394 / gt@40 = 31232
# [2022-12-20 10:35:39,470] {submission_evaluation.py:85} INFO - orders recall@40 = 0.30078125
# [2022-12-20 10:35:39,470] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 10:35:39,470] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3070782260585452
# [2022-12-20 10:35:39,470] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 10:35:46,121] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.3879271070615034
# [2022-12-20 10:35:46,122] {submission_evaluation.py:84} INFO - clicks hits@60 = 68120 / gt@60 = 175600
# [2022-12-20 10:35:46,123] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.3879271070615034
# [2022-12-20 10:35:49,946] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.4452489002681301
# [2022-12-20 10:35:49,947] {submission_evaluation.py:84} INFO - carts hits@60 = 18180 / gt@60 = 58260
# [2022-12-20 10:35:49,947] {submission_evaluation.py:85} INFO - carts recall@60 = 0.3120494335736354
# [2022-12-20 10:35:54,329] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.45545377820564803
# [2022-12-20 10:35:54,329] {submission_evaluation.py:84} INFO - orders hits@60 = 9675 / gt@60 = 31232
# [2022-12-20 10:35:54,330] {submission_evaluation.py:85} INFO - orders recall@60 = 0.30977843237704916
# [2022-12-20 10:35:54,330] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 10:35:54,330] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.31827460020447046
# [2022-12-20 10:35:54,330] {submission_evaluation.py:93} INFO - =============

# fasttext skipgram wdw 50 neg5 minn 3 maxn 6
# [2022-12-19 08:31:30,790] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.32436805314221584
# [2022-12-19 08:31:30,790] {submission_evaluation.py:84} INFO - clicks hits@20 = 56936 / gt@20 = 175529
# [2022-12-19 08:31:30,790] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.32436805314221584
# [2022-12-19 08:31:32,637] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4079584789193014
# [2022-12-19 08:31:32,637] {submission_evaluation.py:84} INFO - carts hits@20 = 16063 / gt@20 = 57532
# [2022-12-19 08:31:32,637] {submission_evaluation.py:85} INFO - carts recall@20 = 0.27920114023499965
# [2022-12-19 08:31:34,659] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4277506650367614
# [2022-12-19 08:31:34,659] {submission_evaluation.py:84} INFO - orders hits@20 = 8911 / gt@20 = 31446
# [2022-12-19 08:31:34,659] {submission_evaluation.py:85} INFO - orders recall@20 = 0.28337467404439354
# [2022-12-19 08:31:34,659] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 08:31:34,659] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.28622195181135757
# [2022-12-19 08:31:34,659] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 08:31:38,253] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.36249850452062055
# [2022-12-19 08:31:38,253] {submission_evaluation.py:84} INFO - clicks hits@40 = 63629 / gt@40 = 175529
# [2022-12-19 08:31:38,253] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.36249850452062055
# [2022-12-19 08:31:41,291] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.433122672503403
# [2022-12-19 08:31:41,292] {submission_evaluation.py:84} INFO - carts hits@40 = 17368 / gt@40 = 57532
# [2022-12-19 08:31:41,292] {submission_evaluation.py:85} INFO - carts recall@40 = 0.30188416881040114
# [2022-12-19 08:31:44,212] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.4484838715793624
# [2022-12-19 08:31:44,212] {submission_evaluation.py:84} INFO - orders hits@40 = 9482 / gt@40 = 31446
# [2022-12-19 08:31:44,212] {submission_evaluation.py:85} INFO - orders recall@40 = 0.30153278636392544
# [2022-12-19 08:31:44,212] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 08:31:44,212] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3077347729135377
# [2022-12-19 08:31:44,213] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 08:31:48,943] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.38369158372690554
# [2022-12-19 08:31:48,944] {submission_evaluation.py:84} INFO - clicks hits@60 = 67349 / gt@60 = 175529
# [2022-12-19 08:31:48,944] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.38369158372690554
# [2022-12-19 08:31:52,452] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.4460233852818283
# [2022-12-19 08:31:52,452] {submission_evaluation.py:84} INFO - carts hits@60 = 18014 / gt@60 = 57532
# [2022-12-19 08:31:52,452] {submission_evaluation.py:85} INFO - carts recall@60 = 0.3131127024960022
# [2022-12-19 08:31:56,184] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.45906691300924507
# [2022-12-19 08:31:56,184] {submission_evaluation.py:84} INFO - orders hits@60 = 9750 / gt@60 = 31446
# [2022-12-19 08:31:56,184] {submission_evaluation.py:85} INFO - orders recall@60 = 0.31005533295172677
# [2022-12-19 08:31:56,184] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 08:31:56,185] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.31833616889252725
# [2022-12-19 08:31:56,185] {submission_evaluation.py:93} INFO - =============

# fasttext skipgram wdw 20 neg5 minn 1 maxn 3
# [2022-12-18 21:13:27,457] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.31939221986736854
# [2022-12-18 21:13:27,458] {submission_evaluation.py:84} INFO - clicks hits@20 = 56061 / gt@20 = 175524
# [2022-12-18 21:13:27,458] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.31939221986736854
# [2022-12-18 21:13:29,400] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4037348918744369
# [2022-12-18 21:13:29,401] {submission_evaluation.py:84} INFO - carts hits@20 = 15924 / gt@20 = 57755
# [2022-12-18 21:13:29,401] {submission_evaluation.py:85} INFO - carts recall@20 = 0.2757163881914986
# [2022-12-18 21:13:31,708] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.41439732041256405
# [2022-12-18 21:13:31,708] {submission_evaluation.py:84} INFO - orders hits@20 = 8580 / gt@20 = 31527
# [2022-12-18 21:13:31,708] {submission_evaluation.py:85} INFO - orders recall@20 = 0.2721476829384337
# [2022-12-18 21:13:31,708] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:13:31,708] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.2779427482072466
# [2022-12-18 21:13:31,708] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 21:13:35,398] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.35225382283904194
# [2022-12-18 21:13:35,399] {submission_evaluation.py:84} INFO - clicks hits@40 = 61829 / gt@40 = 175524
# [2022-12-18 21:13:35,399] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.35225382283904194
# [2022-12-18 21:13:40,044] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4246422022469278
# [2022-12-18 21:13:40,047] {submission_evaluation.py:84} INFO - carts hits@40 = 17014 / gt@40 = 57755
# [2022-12-18 21:13:40,047] {submission_evaluation.py:85} INFO - carts recall@40 = 0.29458921305514674
# [2022-12-18 21:13:43,468] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.43279390427637043
# [2022-12-18 21:13:43,468] {submission_evaluation.py:84} INFO - orders hits@40 = 9046 / gt@40 = 31527
# [2022-12-18 21:13:43,468] {submission_evaluation.py:85} INFO - orders recall@40 = 0.28692866431947217
# [2022-12-18 21:13:43,468] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:13:43,469] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.2957593447921315
# [2022-12-18 21:13:43,469] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 21:13:48,739] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.3697500056972266
# [2022-12-18 21:13:48,739] {submission_evaluation.py:84} INFO - clicks hits@60 = 64900 / gt@60 = 175524
# [2022-12-18 21:13:48,740] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.3697500056972266
# [2022-12-18 21:13:52,712] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.4359775152412315
# [2022-12-18 21:13:52,712] {submission_evaluation.py:84} INFO - carts hits@60 = 17584 / gt@60 = 57755
# [2022-12-18 21:13:52,712] {submission_evaluation.py:85} INFO - carts recall@60 = 0.30445848844255907
# [2022-12-18 21:13:56,946] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.44131634725200014
# [2022-12-18 21:13:56,947] {submission_evaluation.py:84} INFO - orders hits@60 = 9254 / gt@60 = 31527
# [2022-12-18 21:13:56,947] {submission_evaluation.py:85} INFO - orders recall@60 = 0.2935261839058585
# [2022-12-18 21:13:56,947] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:13:56,947] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.30442825744600543
# [2022-12-18 21:13:56,947] {submission_evaluation.py:93} INFO - =============

# fasttext skipgram wdw 5 neg1 minn 2 maxn 3
# [2022-12-18 21:11:34,480] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.2893414019048053
# [2022-12-18 21:11:34,480] {submission_evaluation.py:84} INFO - clicks hits@20 = 50826 / gt@20 = 175661
# [2022-12-18 21:11:34,480] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.2893414019048053
# [2022-12-18 21:11:36,414] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.37776357027677576
# [2022-12-18 21:11:36,414] {submission_evaluation.py:84} INFO - carts hits@20 = 14714 / gt@20 = 58125
# [2022-12-18 21:11:36,414] {submission_evaluation.py:85} INFO - carts recall@20 = 0.2531440860215054
# [2022-12-18 21:11:38,961] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4007383796584877
# [2022-12-18 21:11:38,962] {submission_evaluation.py:84} INFO - orders hits@20 = 8239 / gt@20 = 31502
# [2022-12-18 21:11:38,963] {submission_evaluation.py:85} INFO - orders recall@20 = 0.26153894990794235
# [2022-12-18 21:11:38,963] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:11:38,963] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.26180073594169756
# [2022-12-18 21:11:38,964] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 21:11:43,264] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.30800803820996125
# [2022-12-18 21:11:43,265] {submission_evaluation.py:84} INFO - clicks hits@40 = 54105 / gt@40 = 175661
# [2022-12-18 21:11:43,265] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.30800803820996125
# [2022-12-18 21:11:46,891] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.3904493085043969
# [2022-12-18 21:11:46,892] {submission_evaluation.py:84} INFO - carts hits@40 = 15376 / gt@40 = 58125
# [2022-12-18 21:11:46,892] {submission_evaluation.py:85} INFO - carts recall@40 = 0.26453333333333334
# [2022-12-18 21:11:51,774] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.41209895570086685
# [2022-12-18 21:11:51,775] {submission_evaluation.py:84} INFO - orders hits@40 = 8530 / gt@40 = 31502
# [2022-12-18 21:11:51,775] {submission_evaluation.py:85} INFO - orders recall@40 = 0.27077645863754685
# [2022-12-18 21:11:51,775] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:11:51,775] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.2726266790035242
# [2022-12-18 21:11:51,775] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 21:11:57,533] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.31778823984834426
# [2022-12-18 21:11:57,533] {submission_evaluation.py:84} INFO - clicks hits@60 = 55823 / gt@60 = 175661
# [2022-12-18 21:11:57,534] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.31778823984834426
# [2022-12-18 21:12:01,621] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.3966019746041166
# [2022-12-18 21:12:01,622] {submission_evaluation.py:84} INFO - carts hits@60 = 15690 / gt@60 = 58125
# [2022-12-18 21:12:01,622] {submission_evaluation.py:85} INFO - carts recall@60 = 0.2699354838709677
# [2022-12-18 21:12:13,641] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.4164595128323885
# [2022-12-18 21:12:13,664] {submission_evaluation.py:84} INFO - orders hits@60 = 8647 / gt@60 = 31502
# [2022-12-18 21:12:13,665] {submission_evaluation.py:85} INFO - orders recall@60 = 0.27449050853914037
# [2022-12-18 21:12:13,665] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:12:13,665] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.27745377426960893
# [2022-12-18 21:12:13,665] {submission_evaluation.py:93} INFO - =============

# fasttext cbow wdw 20 neg 5 minn 2 maxn 3
# [2022-12-17 22:39:13,188] {submission_evaluation.py:84} INFO - clicks hits@20 = 39299 / gt@20 = 175598
# [2022-12-17 22:39:13,188] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.2238009544527842
# [2022-12-17 22:39:15,236] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.3386526856004255
# [2022-12-17 22:39:15,236] {submission_evaluation.py:84} INFO - carts hits@20 = 12692 / gt@20 = 57257
# [2022-12-17 22:39:15,236] {submission_evaluation.py:85} INFO - carts recall@20 = 0.22166721972859213
# [2022-12-17 22:39:17,768] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.3740537276447995
# [2022-12-17 22:39:17,768] {submission_evaluation.py:84} INFO - orders hits@20 = 7376 / gt@20 = 30529
# [2022-12-17 22:39:17,768] {submission_evaluation.py:85} INFO - orders recall@20 = 0.24160634151134985
# [2022-12-17 22:39:17,768] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 22:39:17,768] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.23384406627066595
# [2022-12-17 22:39:17,768] {submission_evaluation.py:93} INFO - =============
# [2022-12-17 22:39:21,756] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.2238806820123236
# [2022-12-17 22:39:21,756] {submission_evaluation.py:84} INFO - clicks hits@40 = 39313 / gt@40 = 175598
# [2022-12-17 22:39:21,756] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.2238806820123236
# [2022-12-17 22:39:24,941] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.3387820189712468
# [2022-12-17 22:39:24,942] {submission_evaluation.py:84} INFO - carts hits@40 = 12700 / gt@40 = 57257
# [2022-12-17 22:39:24,942] {submission_evaluation.py:85} INFO - carts recall@40 = 0.22180694063607942
# [2022-12-17 22:39:28,077] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.37405707235013097
# [2022-12-17 22:39:28,077] {submission_evaluation.py:84} INFO - orders hits@40 = 7377 / gt@40 = 30529
# [2022-12-17 22:39:28,078] {submission_evaluation.py:85} INFO - orders recall@40 = 0.2416390972517934
# [2022-12-17 22:39:28,078] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 22:39:28,078] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.23391360874313222
# [2022-12-17 22:39:28,078] {submission_evaluation.py:93} INFO - =============
# [2022-12-17 22:39:33,081] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.22389776648936777
# [2022-12-17 22:39:33,082] {submission_evaluation.py:84} INFO - clicks hits@60 = 39316 / gt@60 = 175598
# [2022-12-17 22:39:33,082] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.22389776648936777
# [2022-12-17 22:39:36,790] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.33883122188405923
# [2022-12-17 22:39:36,790] {submission_evaluation.py:84} INFO - carts hits@60 = 12702 / gt@60 = 57257
# [2022-12-17 22:39:36,790] {submission_evaluation.py:85} INFO - carts recall@60 = 0.22184187086295126
# [2022-12-17 22:39:40,573] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.37409051940344556
# [2022-12-17 22:39:40,573] {submission_evaluation.py:84} INFO - orders hits@60 = 7378 / gt@60 = 30529
# [2022-12-17 22:39:40,573] {submission_evaluation.py:85} INFO - orders recall@60 = 0.2416718529922369
# [2022-12-17 22:39:40,573] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 22:39:40,573] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.2339454497031643
# [2022-12-17 22:39:40,574] {submission_evaluation.py:93} INFO - =============

# fasttext cbow wdw 20 neg 5 minn 1 maxn 3
# [2022-12-18 21:35:07,205] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.22506809194406774
# [2022-12-18 21:35:07,205] {submission_evaluation.py:84} INFO - clicks hits@20 = 39499 / gt@20 = 175498
# [2022-12-18 21:35:07,206] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.22506809194406774
# [2022-12-18 21:35:09,128] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.3457895957158914
# [2022-12-18 21:35:09,130] {submission_evaluation.py:84} INFO - carts hits@20 = 13178 / gt@20 = 57897
# [2022-12-18 21:35:09,130] {submission_evaluation.py:85} INFO - carts recall@20 = 0.22761110247508506
# [2022-12-18 21:35:11,250] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.3739503136230539
# [2022-12-18 21:35:11,250] {submission_evaluation.py:84} INFO - orders hits@20 = 7575 / gt@20 = 31297
# [2022-12-18 21:35:11,250] {submission_evaluation.py:85} INFO - orders recall@20 = 0.24203597788925457
# [2022-12-18 21:35:11,250] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:35:11,250] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.23601172667048503
# [2022-12-18 21:35:11,251] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 21:35:15,060] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.22510797843850072
# [2022-12-18 21:35:15,061] {submission_evaluation.py:84} INFO - clicks hits@40 = 39506 / gt@40 = 175498
# [2022-12-18 21:35:15,061] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.22510797843850072
# [2022-12-18 21:35:18,148] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.3457895957158914
# [2022-12-18 21:35:18,148] {submission_evaluation.py:84} INFO - carts hits@40 = 13178 / gt@40 = 57897
# [2022-12-18 21:35:18,148] {submission_evaluation.py:85} INFO - carts recall@40 = 0.22761110247508506
# [2022-12-18 21:35:21,119] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.3739503136230539
# [2022-12-18 21:35:21,119] {submission_evaluation.py:84} INFO - orders hits@40 = 7575 / gt@40 = 31297
# [2022-12-18 21:35:21,119] {submission_evaluation.py:85} INFO - orders recall@40 = 0.24203597788925457
# [2022-12-18 21:35:21,119] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:35:21,119] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.23601571531992832
# [2022-12-18 21:35:21,119] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 21:35:26,183] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.22512507265040058
# [2022-12-18 21:35:26,184] {submission_evaluation.py:84} INFO - clicks hits@60 = 39509 / gt@60 = 175498
# [2022-12-18 21:35:26,184] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.22512507265040058
# [2022-12-18 21:35:29,805] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.3458219948735133
# [2022-12-18 21:35:29,805] {submission_evaluation.py:84} INFO - carts hits@60 = 13180 / gt@60 = 57897
# [2022-12-18 21:35:29,805] {submission_evaluation.py:85} INFO - carts recall@60 = 0.22764564657927008
# [2022-12-18 21:35:33,729] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.3739503136230539
# [2022-12-18 21:35:33,729] {submission_evaluation.py:84} INFO - orders hits@60 = 7575 / gt@60 = 31297
# [2022-12-18 21:35:33,729] {submission_evaluation.py:85} INFO - orders recall@60 = 0.24203597788925457
# [2022-12-18 21:35:33,730] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:35:33,730] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.2360277879723738
# [2022-12-18 21:35:33,730] {submission_evaluation.py:93} INFO - =============

# fasttext skipgram wdw 5 neg1 minn 3 maxn 6
# [2022-12-17 13:10:33,313] {submission_evaluation.py:84} INFO - clicks hits@20 = 55654 / gt@20 = 175601
# [2022-12-17 13:10:33,313] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3169344138131332
# [2022-12-17 13:10:34,638] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.3959206689992911
# [2022-12-17 13:10:34,638] {submission_evaluation.py:84} INFO - carts hits@20 = 15508 / gt@20 = 57535
# [2022-12-17 13:10:34,638] {submission_evaluation.py:85} INFO - carts recall@20 = 0.2695402798296689
# [2022-12-17 13:10:36,156] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4169724117536153
# [2022-12-17 13:10:36,156] {submission_evaluation.py:84} INFO - orders hits@20 = 8573 / gt@20 = 31045
# [2022-12-17 13:10:36,156] {submission_evaluation.py:85} INFO - orders recall@20 = 0.27614752778225155
# [2022-12-17 13:10:36,156] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 13:10:36,156] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.2782440419995649
# [2022-12-17 13:10:36,156] {submission_evaluation.py:93} INFO - =============
# [2022-12-17 13:10:38,760] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.35237840331205406
# [2022-12-17 13:10:38,761] {submission_evaluation.py:84} INFO - clicks hits@40 = 61878 / gt@40 = 175601
# [2022-12-17 13:10:38,761] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.35237840331205406
# [2022-12-17 13:10:41,008] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4180316547374117
# [2022-12-17 13:10:41,008] {submission_evaluation.py:84} INFO - carts hits@40 = 16644 / gt@40 = 57535
# [2022-12-17 13:10:41,008] {submission_evaluation.py:85} INFO - carts recall@40 = 0.2892847831754584
# [2022-12-17 13:10:43,166] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.43241560423019926
# [2022-12-17 13:10:43,166] {submission_evaluation.py:84} INFO - orders hits@40 = 9009 / gt@40 = 31045
# [2022-12-17 13:10:43,166] {submission_evaluation.py:85} INFO - orders recall@40 = 0.2901916572717024
# [2022-12-17 13:10:43,166] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 13:10:43,166] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.29613826964686435
# [2022-12-17 13:10:43,166] {submission_evaluation.py:93} INFO - =============
# [2022-12-17 13:10:46,544] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.3712222595543306
# [2022-12-17 13:10:46,545] {submission_evaluation.py:84} INFO - clicks hits@60 = 65187 / gt@60 = 175601
# [2022-12-17 13:10:46,545] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.3712222595543306
# [2022-12-17 13:10:49,054] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.42863753097485296
# [2022-12-17 13:10:49,054] {submission_evaluation.py:84} INFO - carts hits@60 = 17199 / gt@60 = 57535
# [2022-12-17 13:10:49,054] {submission_evaluation.py:85} INFO - carts recall@60 = 0.2989310854262623
# [2022-12-17 13:10:51,777] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.44054726021017504
# [2022-12-17 13:10:51,777] {submission_evaluation.py:84} INFO - orders hits@60 = 9223 / gt@60 = 31045
# [2022-12-17 13:10:51,777] {submission_evaluation.py:85} INFO - orders recall@60 = 0.2970848767917539
# [2022-12-17 13:10:51,777] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 13:10:51,777] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3050524776583641
# [2022-12-17 13:10:51,778] {submission_evaluation.py:93} INFO - =============

# fasttext cbow wdw 5 neg1 minn 3 maxn 6
# [2022-12-17 22:38:12,875] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.234519621571775
# [2022-12-17 22:38:12,875] {submission_evaluation.py:84} INFO - clicks hits@20 = 41199 / gt@20 = 175674
# [2022-12-17 22:38:12,875] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.234519621571775
# [2022-12-17 22:38:14,773] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.3464909710157333
# [2022-12-17 22:38:14,773] {submission_evaluation.py:84} INFO - carts hits@20 = 13183 / gt@20 = 57751
# [2022-12-17 22:38:14,773] {submission_evaluation.py:85} INFO - carts recall@20 = 0.22827310349604335
# [2022-12-17 22:38:16,897] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.3712826601313368
# [2022-12-17 22:38:16,898] {submission_evaluation.py:84} INFO - orders hits@20 = 7489 / gt@20 = 31580
# [2022-12-17 22:38:16,898] {submission_evaluation.py:85} INFO - orders recall@20 = 0.23714376187460418
# [2022-12-17 22:38:16,898] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 22:38:16,898] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.234220150330753
# [2022-12-17 22:38:16,898] {submission_evaluation.py:93} INFO - =============
# [2022-12-17 22:38:20,893] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.23504901123672256
# [2022-12-17 22:38:20,893] {submission_evaluation.py:84} INFO - clicks hits@40 = 41292 / gt@40 = 175674
# [2022-12-17 22:38:20,893] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.23504901123672256
# [2022-12-17 22:38:23,978] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.34676733042454266
# [2022-12-17 22:38:23,978] {submission_evaluation.py:84} INFO - carts hits@40 = 13197 / gt@40 = 57751
# [2022-12-17 22:38:23,978] {submission_evaluation.py:85} INFO - carts recall@40 = 0.2285155235407179
# [2022-12-17 22:38:27,027] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.3714310085192843
# [2022-12-17 22:38:27,027] {submission_evaluation.py:84} INFO - orders hits@40 = 7493 / gt@40 = 31580
# [2022-12-17 22:38:27,027] {submission_evaluation.py:85} INFO - orders recall@40 = 0.23727042431918935
# [2022-12-17 22:38:27,027] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 22:38:27,028] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.23442181277740126
# [2022-12-17 22:38:27,028] {submission_evaluation.py:93} INFO - =============
# [2022-12-17 22:38:31,864] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.23522547445837175
# [2022-12-17 22:38:31,865] {submission_evaluation.py:84} INFO - clicks hits@60 = 41323 / gt@60 = 175674
# [2022-12-17 22:38:31,865] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.23522547445837175
# [2022-12-17 22:38:35,469] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.34693161442915565
# [2022-12-17 22:38:35,470] {submission_evaluation.py:84} INFO - carts hits@60 = 13205 / gt@60 = 57751
# [2022-12-17 22:38:35,470] {submission_evaluation.py:85} INFO - carts recall@60 = 0.22865404928053193
# [2022-12-17 22:38:39,395] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.3716617726783138
# [2022-12-17 22:38:39,397] {submission_evaluation.py:84} INFO - orders hits@60 = 7497 / gt@60 = 31580
# [2022-12-17 22:38:39,397] {submission_evaluation.py:85} INFO - orders recall@60 = 0.23739708676377455
# [2022-12-17 22:38:39,397] {submission_evaluation.py:91} INFO - =============
# [2022-12-17 22:38:39,397] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.23455701428826148
# [2022-12-17 22:38:39,397] {submission_evaluation.py:93} INFO - =============
