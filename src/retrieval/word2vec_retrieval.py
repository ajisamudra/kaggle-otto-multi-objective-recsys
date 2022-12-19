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
from src.utils.word2vec import load_annoy_idx_word2vec_embedding
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
    matrix_fact_idx = load_annoy_idx_word2vec_embedding()

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
        n_candidate=70,
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
    measure_recall(df_pred=pred_df, df_truth=test_labels, Ks=[20, 40, 50, 60, 70])

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

# covisit + word2vec cbow 20wdw 5negative retrieval
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

# covisit + word2vec skipgram 30wdw 5negative retrieval
# [2022-12-18 01:12:18,000] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.3727990977083844
# [2022-12-18 01:12:18,001] {submission_evaluation.py:84} INFO - clicks hits@20 = 65446 / gt@20 = 175553
# [2022-12-18 01:12:18,001] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3727990977083844
# [2022-12-18 01:12:19,439] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.43476485442215823
# [2022-12-18 01:12:19,439] {submission_evaluation.py:84} INFO - carts hits@20 = 17546 / gt@20 = 57496
# [2022-12-18 01:12:19,439] {submission_evaluation.py:85} INFO - carts recall@20 = 0.3051690552386253
# [2022-12-18 01:12:20,951] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.45387482192377165
# [2022-12-18 01:12:20,951] {submission_evaluation.py:84} INFO - orders hits@20 = 9596 / gt@20 = 31058
# [2022-12-18 01:12:20,951] {submission_evaluation.py:85} INFO - orders recall@20 = 0.30897031360680016
# [2022-12-18 01:12:20,951] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 01:12:20,951] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.3142128145065061
# [2022-12-18 01:12:20,951] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 01:12:23,765] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.41811874476653776
# [2022-12-18 01:12:23,766] {submission_evaluation.py:84} INFO - clicks hits@40 = 73402 / gt@40 = 175553
# [2022-12-18 01:12:23,766] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.41811874476653776
# [2022-12-18 01:12:26,121] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.46488816628820456
# [2022-12-18 01:12:26,122] {submission_evaluation.py:84} INFO - carts hits@40 = 19123 / gt@40 = 57496
# [2022-12-18 01:12:26,122] {submission_evaluation.py:85} INFO - carts recall@40 = 0.3325970502295812
# [2022-12-18 01:12:28,356] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.4784588351989086
# [2022-12-18 01:12:28,356] {submission_evaluation.py:84} INFO - orders hits@40 = 10299 / gt@40 = 31058
# [2022-12-18 01:12:28,356] {submission_evaluation.py:85} INFO - orders recall@40 = 0.33160538347607704
# [2022-12-18 01:12:28,356] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 01:12:28,356] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3405542196311744
# [2022-12-18 01:12:28,356] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 50wdw 5negative retrieval
# [2022-12-19 08:44:23,159] {submission_evaluation.py:84} INFO - clicks hits@20 = 65209 / gt@20 = 175626
# [2022-12-19 08:44:23,159] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.37129468301959845
# [2022-12-19 08:44:24,612] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4406296815047327
# [2022-12-19 08:44:24,612] {submission_evaluation.py:84} INFO - carts hits@20 = 17562 / gt@20 = 57029
# [2022-12-19 08:44:24,612] {submission_evaluation.py:85} INFO - carts recall@20 = 0.30794858756071475
# [2022-12-19 08:44:26,288] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4535407594199703
# [2022-12-19 08:44:26,289] {submission_evaluation.py:84} INFO - orders hits@20 = 9676 / gt@20 = 31283
# [2022-12-19 08:44:26,289] {submission_evaluation.py:85} INFO - orders recall@20 = 0.30930537352555704
# [2022-12-19 08:44:26,289] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 08:44:26,289] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.31509726868550847
# [2022-12-19 08:44:26,289] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 08:44:29,901] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.41619691845170986
# [2022-12-19 08:44:29,902] {submission_evaluation.py:84} INFO - clicks hits@40 = 73095 / gt@40 = 175626
# [2022-12-19 08:44:29,902] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.41619691845170986
# [2022-12-19 08:44:32,792] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.47075796719541385
# [2022-12-19 08:44:32,793] {submission_evaluation.py:84} INFO - carts hits@40 = 19150 / gt@40 = 57029
# [2022-12-19 08:44:32,793] {submission_evaluation.py:85} INFO - carts recall@40 = 0.33579406968384506
# [2022-12-19 08:44:35,241] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.4785545526636026
# [2022-12-19 08:44:35,242] {submission_evaluation.py:84} INFO - orders hits@40 = 10376 / gt@40 = 31283
# [2022-12-19 08:44:35,242] {submission_evaluation.py:85} INFO - orders recall@40 = 0.33168174407825335
# [2022-12-19 08:44:35,242] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 08:44:35,242] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3413669591972765
# [2022-12-19 08:44:35,242] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 08:44:38,477] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@50 = 0.4301697926275153
# [2022-12-19 08:44:38,477] {submission_evaluation.py:84} INFO - clicks hits@50 = 75549 / gt@50 = 175626
# [2022-12-19 08:44:38,478] {submission_evaluation.py:85} INFO - clicks recall@50 = 0.4301697926275153
# [2022-12-19 08:44:40,964] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@50 = 0.4800639768833599
# [2022-12-19 08:44:40,964] {submission_evaluation.py:84} INFO - carts hits@50 = 19667 / gt@50 = 57029
# [2022-12-19 08:44:40,964] {submission_evaluation.py:85} INFO - carts recall@50 = 0.34485963281839066
# [2022-12-19 08:44:43,764] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@50 = 0.48693482338407756
# [2022-12-19 08:44:43,765] {submission_evaluation.py:84} INFO - orders hits@50 = 10614 / gt@50 = 31283
# [2022-12-19 08:44:43,765] {submission_evaluation.py:85} INFO - orders recall@50 = 0.33928971006617015
# [2022-12-19 08:44:43,765] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 08:44:43,765] {submission_evaluation.py:92} INFO - Overall Recall@50 = 0.3500486951479708
# [2022-12-19 08:44:43,765] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 08:44:47,462] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.4417170578388166
# [2022-12-19 08:44:47,463] {submission_evaluation.py:84} INFO - clicks hits@60 = 77577 / gt@60 = 175626
# [2022-12-19 08:44:47,463] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.4417170578388166
# [2022-12-19 08:44:50,634] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.48798064065048824
# [2022-12-19 08:44:50,634] {submission_evaluation.py:84} INFO - carts hits@60 = 20064 / gt@60 = 57029
# [2022-12-19 08:44:50,634] {submission_evaluation.py:85} INFO - carts recall@60 = 0.3518210033491732
# [2022-12-19 08:44:53,218] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.4942234940339052
# [2022-12-19 08:44:53,218] {submission_evaluation.py:84} INFO - orders hits@60 = 10810 / gt@60 = 31283
# [2022-12-19 08:44:53,218] {submission_evaluation.py:85} INFO - orders recall@60 = 0.3455550938209251
# [2022-12-19 08:44:53,218] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 08:44:53,218] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.35705106308118867
# [2022-12-19 08:44:53,218] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 08:44:57,102] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.4489597212257866
# [2022-12-19 08:44:57,103] {submission_evaluation.py:84} INFO - clicks hits@70 = 78849 / gt@70 = 175626
# [2022-12-19 08:44:57,103] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.4489597212257866
# [2022-12-19 08:45:00,615] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.49187134451901515
# [2022-12-19 08:45:00,615] {submission_evaluation.py:84} INFO - carts hits@70 = 20266 / gt@70 = 57029
# [2022-12-19 08:45:00,615] {submission_evaluation.py:85} INFO - carts recall@70 = 0.35536306089884095
# [2022-12-19 08:45:03,925] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.4974291610941177
# [2022-12-19 08:45:03,925] {submission_evaluation.py:84} INFO - orders hits@70 = 10896 / gt@70 = 31283
# [2022-12-19 08:45:03,925] {submission_evaluation.py:85} INFO - orders recall@70 = 0.3483041907745421
# [2022-12-19 08:45:03,925] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 08:45:03,925] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.3604874048569562

# covisit + word2vec skipgram 70wdw 10negative retrieval
# [2022-12-18 21:20:14,129] {submission_evaluation.py:84} INFO - clicks hits@20 = 64826 / gt@20 = 175474
# [2022-12-18 21:20:14,129] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3694336482897751
# [2022-12-18 21:20:16,319] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.43715250071531175
# [2022-12-18 21:20:16,319] {submission_evaluation.py:84} INFO - carts hits@20 = 17529 / gt@20 = 57817
# [2022-12-18 21:20:16,319] {submission_evaluation.py:85} INFO - carts recall@20 = 0.3031807253921857
# [2022-12-18 21:20:18,489] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.45241137454353847
# [2022-12-18 21:20:18,490] {submission_evaluation.py:84} INFO - orders hits@20 = 9527 / gt@20 = 31313
# [2022-12-18 21:20:18,490] {submission_evaluation.py:85} INFO - orders recall@20 = 0.30425063072845143
# [2022-12-18 21:20:18,490] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:20:18,490] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.3104479608837041
# [2022-12-18 21:20:18,490] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 21:20:22,278] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.4135142528237802
# [2022-12-18 21:20:22,278] {submission_evaluation.py:84} INFO - clicks hits@40 = 72561 / gt@40 = 175474
# [2022-12-18 21:20:22,278] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.4135142528237802
# [2022-12-18 21:20:26,073] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.46766813216115294
# [2022-12-18 21:20:26,073] {submission_evaluation.py:84} INFO - carts hits@40 = 19127 / gt@40 = 57817
# [2022-12-18 21:20:26,073] {submission_evaluation.py:85} INFO - carts recall@40 = 0.3308196551187367
# [2022-12-18 21:20:29,951] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.4796694568766406
# [2022-12-18 21:20:29,951] {submission_evaluation.py:84} INFO - orders hits@40 = 10309 / gt@40 = 31313
# [2022-12-18 21:20:29,951] {submission_evaluation.py:85} INFO - orders recall@40 = 0.32922428384377095
# [2022-12-18 21:20:29,951] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:20:29,951] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3381318921242616
# [2022-12-18 21:20:29,952] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 21:20:35,657] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.43639513546166386
# [2022-12-18 21:20:35,657] {submission_evaluation.py:84} INFO - clicks hits@60 = 76576 / gt@60 = 175474
# [2022-12-18 21:20:35,657] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.43639513546166386
# [2022-12-18 21:20:39,458] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.48194374485114283
# [2022-12-18 21:20:39,459] {submission_evaluation.py:84} INFO - carts hits@60 = 19866 / gt@60 = 57817
# [2022-12-18 21:20:39,459] {submission_evaluation.py:85} INFO - carts recall@60 = 0.34360136292094023
# [2022-12-18 21:20:43,615] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.49029327653769
# [2022-12-18 21:20:43,615] {submission_evaluation.py:84} INFO - orders hits@60 = 10599 / gt@60 = 31313
# [2022-12-18 21:20:43,615] {submission_evaluation.py:85} INFO - orders recall@60 = 0.33848561300418356
# [2022-12-18 21:20:43,615] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 21:20:43,615] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3498112902249586

# covisit + word2vec cbow 50wdw 5negative retrieval
# [2022-12-18 23:54:57,822] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.3086624399747082
# [2022-12-18 23:54:57,822] {submission_evaluation.py:84} INFO - clicks hits@20 = 54186 / gt@20 = 175551
# [2022-12-18 23:54:57,822] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3086624399747082
# [2022-12-18 23:54:59,672] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4001419499892357
# [2022-12-18 23:54:59,673] {submission_evaluation.py:84} INFO - carts hits@20 = 15813 / gt@20 = 57922
# [2022-12-18 23:54:59,673] {submission_evaluation.py:85} INFO - carts recall@20 = 0.2730050757915818
# [2022-12-18 23:55:01,704] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.41845948889699974
# [2022-12-18 23:55:01,704] {submission_evaluation.py:84} INFO - orders hits@20 = 8654 / gt@20 = 31093
# [2022-12-18 23:55:01,704] {submission_evaluation.py:85} INFO - orders recall@20 = 0.2783263113884154
# [2022-12-18 23:55:01,704] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 23:55:01,704] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.2797635535679946
# [2022-12-18 23:55:01,704] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 23:55:05,363] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.3361587231061059
# [2022-12-18 23:55:05,363] {submission_evaluation.py:84} INFO - clicks hits@40 = 59013 / gt@40 = 175551
# [2022-12-18 23:55:05,363] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.3361587231061059
# [2022-12-18 23:55:08,417] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.41856324486861696
# [2022-12-18 23:55:08,418] {submission_evaluation.py:84} INFO - carts hits@40 = 16805 / gt@40 = 57922
# [2022-12-18 23:55:08,418] {submission_evaluation.py:85} INFO - carts recall@40 = 0.2901315562307931
# [2022-12-18 23:55:11,387] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.4326072931277852
# [2022-12-18 23:55:11,388] {submission_evaluation.py:84} INFO - orders hits@40 = 9045 / gt@40 = 31093
# [2022-12-18 23:55:11,388] {submission_evaluation.py:85} INFO - orders recall@40 = 0.29090148908114366
# [2022-12-18 23:55:11,388] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 23:55:11,388] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.2951962326285347
# [2022-12-18 23:55:11,388] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 23:55:16,156] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.35098632306281363
# [2022-12-18 23:55:16,159] {submission_evaluation.py:84} INFO - clicks hits@60 = 61616 / gt@60 = 175551
# [2022-12-18 23:55:16,159] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.35098632306281363
# [2022-12-18 23:55:19,671] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.42718250487760584
# [2022-12-18 23:55:19,671] {submission_evaluation.py:84} INFO - carts hits@60 = 17262 / gt@60 = 57922
# [2022-12-18 23:55:19,672] {submission_evaluation.py:85} INFO - carts recall@60 = 0.2980214771589379
# [2022-12-18 23:55:23,440] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.44049761220879324
# [2022-12-18 23:55:23,440] {submission_evaluation.py:84} INFO - orders hits@60 = 9244 / gt@60 = 31093
# [2022-12-18 23:55:23,440] {submission_evaluation.py:85} INFO - orders recall@60 = 0.2973016434567266
# [2022-12-18 23:55:23,440] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 23:55:23,440] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3028860615279987
# [2022-12-18 23:55:23,441] {submission_evaluation.py:93} INFO - =============
