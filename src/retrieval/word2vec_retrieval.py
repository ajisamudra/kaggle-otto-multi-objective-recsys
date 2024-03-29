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
from src.utils.word2vec import (
    load_annoy_idx_word2vec_embedding,
    load_annoy_idx_word2vec_cart_vect32_wdw45_neg5_real_session_embedding,
    load_annoy_idx_word2vec_buy_vect32_wdw15_neg7_embedding,
)
from src.metrics.submission_evaluation import measure_recall
from src.utils.logger import get_logger

DATA_DIR = get_processed_local_validation_dir()
logging = get_logger()

######### CANDIDATES GENERATION FUNCTION


def suggest_word2vec(
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
    embedding: AnnoyIndex,
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
        # # last x aids
        # last_x_aids = unique_aids[-2:]
        # # get query vector from last three aids
        # query_vcts = []
        # for aid in last_x_aids:
        #     vct = []
        #     try:
        #         vct = embedding.get_item_vector(aid)
        #     except KeyError:
        #         continue
        #     query_vcts.append(vct)
        # query_vcts = np.array(query_vcts)
        # query_vcts = np.mean(query_vcts, axis=0)

        types = ses2types[session]

        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]
            candidate = [k for k, v in aids_temp.most_common(20)]
            # mf_candidate = embedding.get_nns_by_vector(query_vcts, n=n_candidate - 20)
            mf_candidate = embedding.get_nns_by_item(unique_aids[0], n=n_candidate - 20)
            candidate.extend(mf_candidate)

        else:
            candidate = list(unique_aids)
            # mf_candidate = embedding.get_nns_by_vector(
            #     query_vcts, n=n_candidate - (len(candidate))
            # )
            mf_candidate = embedding.get_nns_by_item(
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


def suggest_cart_order(
    n_candidate: int,
    ses2aids: dict,
    ses2types: dict,
    embedding: AnnoyIndex,
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

        # get reverse aids and its types
        reversed_aids = aids[::-1]
        reversed_types = types[::-1]

        carted_aids = [
            aid for i, aid in enumerate(reversed_aids) if reversed_types[i] == 1
        ]

        # last x aids
        if len(carted_aids) == 0:
            last_x_aids = reversed_aids[:1]
        elif len(carted_aids) < 3:
            last_x_aids = carted_aids[: len(carted_aids)]
        else:
            last_x_aids = carted_aids[:3]
        # get query vector from last three aids
        query_vcts = []
        for aid in last_x_aids:
            vct = []
            try:
                vct = embedding.get_item_vector(aid)
            except KeyError:
                continue
            query_vcts.append(vct)
        query_vcts = np.array(query_vcts)
        query_vcts = np.mean(query_vcts, axis=0)

        # get last cart event
        cart_idx = 0
        try:
            cart_idx = reversed_types.index(1)
        except ValueError:
            cart_idx = 0

        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * type_weight_multipliers[t]
            candidate = [k for k, v in aids_temp.most_common(20)]
            mf_candidate = embedding.get_nns_by_vector(query_vcts, n=n_candidate - 20)
            # mf_candidate = embedding.get_nns_by_item(
            #     reversed_aids[cart_idx], n=n_candidate - 20
            # )
            candidate.extend(mf_candidate)

        else:
            candidate = list(unique_aids)
            mf_candidate = embedding.get_nns_by_vector(
                query_vcts, n=n_candidate - (len(candidate))
            )
            # mf_candidate = embedding.get_nns_by_item(
            #     reversed_aids[cart_idx], n=n_candidate - (len(candidate))
            # )
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
    click_word2vec = load_annoy_idx_word2vec_embedding()
    # cart_word2vec = (
    #     load_annoy_idx_word2vec_cart_vect32_wdw45_neg5_real_session_embedding()
    # )
    # order_word2vec = load_annoy_idx_word2vec_buy_vect32_wdw15_neg7_embedding()

    logging.info("start of suggesting clicks")
    logging.info("sort session by ts ascendingly")
    df_val = df_val.sort_values(["session", "ts"])
    # sample session id
    lucky_sessions = df_val.drop_duplicates(["session"]).sample(
        frac=0.1, random_state=1234
    )["session"]
    df_val = df_val[df_val.session.isin(lucky_sessions)]
    # create dict of session_id: list of aid/ts/type
    logging.info("create ses2aids")
    ses2aids = df_val.groupby("session")["aid"].apply(list).to_dict()
    logging.info("create ses2types")
    ses2types = df_val.groupby("session")["type"].apply(list).to_dict()

    logging.info("start of suggesting click")
    pred_click_series = suggest_word2vec(
        n_candidate=70,
        ses2aids=ses2aids,
        ses2types=ses2types,
        embedding=click_word2vec,
    )
    logging.info("end of suggesting click")

    logging.info("start of suggesting order")
    pred_cart_series = suggest_cart_order(
        n_candidate=70,
        ses2aids=ses2aids,
        ses2types=ses2types,
        embedding=click_word2vec,
    )
    logging.info("end of suggesting order")

    # logging.info("start of suggesting cart")
    # pred_cart_series = suggest_word2vec(
    #     n_candidate=70,
    #     ses2aids=ses2aids,
    #     ses2types=ses2types,
    #     embedding=cart_word2vec,
    # )
    # logging.info("end of suggesting cart")

    # logging.info("start of suggesting order")
    # pred_order_series = suggest_word2vec(
    #     n_candidate=70,
    #     ses2aids=ses2aids,
    #     ses2types=ses2types,
    #     embedding=order_word2vec,
    # )
    # logging.info("end of suggesting order")

    del df_val
    gc.collect()

    logging.info("create predicition df for click")
    clicks_pred_df = pd.DataFrame(
        pred_click_series.add_suffix("_clicks"), columns=["labels"]
    ).reset_index()
    logging.info("create predicition df for cart")
    carts_pred_df = pd.DataFrame(
        pred_click_series.add_suffix("_carts"), columns=["labels"]
    ).reset_index()
    logging.info("create predicition df for order")
    orders_pred_df = pd.DataFrame(
        pred_cart_series.add_suffix("_orders"), columns=["labels"]
    ).reset_index()

    # del pred_df_clicks, pred_df_carts, pred_df_buys
    # gc.collect()

    # del pred_order_series, pred_cart_series, pred_click_series
    del pred_click_series, pred_cart_series
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
    measure_recall(df_pred=pred_df, df_truth=test_labels, Ks=[20, 40, 60, 70])

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

# covisit + word2vec skipgram 50wdw 5negative retrieval vec_size 32
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

# covisit + word2vec skipgram 50wdw 5negative retrieval vec_size 32 + suggest order by last 3 cart events
# [2022-12-28 00:39:37,036] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.37095671592475704
# [2022-12-28 00:39:37,036] {submission_evaluation.py:84} INFO - clicks hits@20 = 65117 / gt@20 = 175538
# [2022-12-28 00:39:37,036] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.37095671592475704
# [2022-12-28 00:39:40,133] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.43738227152836834
# [2022-12-28 00:39:40,134] {submission_evaluation.py:84} INFO - carts hits@20 = 17716 / gt@20 = 58593
# [2022-12-28 00:39:40,134] {submission_evaluation.py:85} INFO - carts recall@20 = 0.30235693683545817
# [2022-12-28 00:39:42,640] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.5154737807106566
# [2022-12-28 00:39:42,640] {submission_evaluation.py:84} INFO - orders hits@20 = 11636 / gt@20 = 31786
# [2022-12-28 00:39:42,640] {submission_evaluation.py:85} INFO - orders recall@20 = 0.3660731139495375
# [2022-12-28 00:39:42,640] {submission_evaluation.py:91} INFO - =============
# [2022-12-28 00:39:42,640] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.34744662101283563
# [2022-12-28 00:39:42,640] {submission_evaluation.py:93} INFO - =============
# [2022-12-28 00:39:46,898] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.4163941710626759
# [2022-12-28 00:39:46,898] {submission_evaluation.py:84} INFO - clicks hits@40 = 73093 / gt@40 = 175538
# [2022-12-28 00:39:46,898] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.4163941710626759
# [2022-12-28 00:39:50,661] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.46796373617411097
# [2022-12-28 00:39:50,661] {submission_evaluation.py:84} INFO - carts hits@40 = 19326 / gt@40 = 58593
# [2022-12-28 00:39:50,661] {submission_evaluation.py:85} INFO - carts recall@40 = 0.3298346218831601
# [2022-12-28 00:39:53,992] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.5413961079777829
# [2022-12-28 00:39:53,992] {submission_evaluation.py:84} INFO - orders hits@40 = 12424 / gt@40 = 31786
# [2022-12-28 00:39:53,992] {submission_evaluation.py:85} INFO - orders recall@40 = 0.3908639023469452
# [2022-12-28 00:39:53,993] {submission_evaluation.py:91} INFO - =============
# [2022-12-28 00:39:53,993] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3751081450793827
# [2022-12-28 00:39:53,993] {submission_evaluation.py:93} INFO - =============
# [2022-12-28 00:39:58,683] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.4424796910070754
# [2022-12-28 00:39:58,684] {submission_evaluation.py:84} INFO - clicks hits@60 = 77672 / gt@60 = 175538
# [2022-12-28 00:39:58,684] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.4424796910070754
# [2022-12-28 00:40:03,026] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.48590346977671417
# [2022-12-28 00:40:03,026] {submission_evaluation.py:84} INFO - carts hits@60 = 20307 / gt@60 = 58593
# [2022-12-28 00:40:03,026] {submission_evaluation.py:85} INFO - carts recall@60 = 0.34657723618862324
# [2022-12-28 00:40:07,255] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.5559563859484573
# [2022-12-28 00:40:07,255] {submission_evaluation.py:84} INFO - orders hits@60 = 12850 / gt@60 = 31786
# [2022-12-28 00:40:07,255] {submission_evaluation.py:85} INFO - orders recall@60 = 0.4042660290694016
# [2022-12-28 00:40:07,256] {submission_evaluation.py:91} INFO - =============
# [2022-12-28 00:40:07,256] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3907807573989355
# [2022-12-28 00:40:07,256] {submission_evaluation.py:93} INFO - =============
# [2022-12-28 00:40:12,156] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.449577869179323
# [2022-12-28 00:40:12,157] {submission_evaluation.py:84} INFO - clicks hits@70 = 78918 / gt@70 = 175538
# [2022-12-28 00:40:12,157] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.449577869179323
# [2022-12-28 00:40:16,943] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.49040816168809537
# [2022-12-28 00:40:16,944] {submission_evaluation.py:84} INFO - carts hits@70 = 20540 / gt@70 = 58593
# [2022-12-28 00:40:16,945] {submission_evaluation.py:85} INFO - carts recall@70 = 0.35055382042223476
# [2022-12-28 00:40:21,674] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.5587805506696963
# [2022-12-28 00:40:21,674] {submission_evaluation.py:84} INFO - orders hits@70 = 12928 / gt@70 = 31786
# [2022-12-28 00:40:21,674] {submission_evaluation.py:85} INFO - orders recall@70 = 0.4067199395960486
# [2022-12-28 00:40:21,674] {submission_evaluation.py:91} INFO - =============
# [2022-12-28 00:40:21,674] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.39415589680223184
# [2022-12-28 00:40:21,674] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 50wdw 5negative retrieval vec_size 32 + nn by last 3 aid vectors
# [2022-12-27 23:52:26,026] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.3522569988323413
# [2022-12-27 23:52:26,026] {submission_evaluation.py:84} INFO - clicks hits@20 = 61844 / gt@20 = 175565
# [2022-12-27 23:52:26,026] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3522569988323413
# [2022-12-27 23:52:28,024] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.34733578283027794
# [2022-12-27 23:52:28,025] {submission_evaluation.py:84} INFO - carts hits@20 = 13509 / gt@20 = 57323
# [2022-12-27 23:52:28,025] {submission_evaluation.py:85} INFO - carts recall@20 = 0.23566456745111036
# [2022-12-27 23:52:30,227] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.40654864274552605
# [2022-12-27 23:52:30,227] {submission_evaluation.py:84} INFO - orders hits@20 = 8641 / gt@20 = 31617
# [2022-12-27 23:52:30,227] {submission_evaluation.py:85} INFO - orders recall@20 = 0.27330233735015974
# [2022-12-27 23:52:30,227] {submission_evaluation.py:91} INFO - =============
# [2022-12-27 23:52:30,227] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.26990647252866307
# [2022-12-27 23:52:30,227] {submission_evaluation.py:93} INFO - =============
# [2022-12-27 23:52:34,033] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.3940876598410845
# [2022-12-27 23:52:34,033] {submission_evaluation.py:84} INFO - clicks hits@40 = 69188 / gt@40 = 175565
# [2022-12-27 23:52:34,033] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.3940876598410845
# [2022-12-27 23:52:37,256] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.3802258973924395
# [2022-12-27 23:52:37,257] {submission_evaluation.py:84} INFO - carts hits@40 = 15179 / gt@40 = 57323
# [2022-12-27 23:52:37,257] {submission_evaluation.py:85} INFO - carts recall@40 = 0.26479772517139716
# [2022-12-27 23:52:40,331] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.4438551779015861
# [2022-12-27 23:52:40,331] {submission_evaluation.py:84} INFO - orders hits@40 = 9668 / gt@40 = 31617
# [2022-12-27 23:52:40,331] {submission_evaluation.py:85} INFO - orders recall@40 = 0.3057848625739317
# [2022-12-27 23:52:40,331] {submission_evaluation.py:91} INFO - =============
# [2022-12-27 23:52:40,332] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.30231900107988663
# [2022-12-27 23:52:40,332] {submission_evaluation.py:93} INFO - =============
# [2022-12-27 23:52:45,252] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.41834648136018
# [2022-12-27 23:52:45,253] {submission_evaluation.py:84} INFO - clicks hits@60 = 73447 / gt@60 = 175565
# [2022-12-27 23:52:45,253] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.41834648136018
# [2022-12-27 23:52:48,987] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.3999644755796916
# [2022-12-27 23:52:48,987] {submission_evaluation.py:84} INFO - carts hits@60 = 16158 / gt@60 = 57323
# [2022-12-27 23:52:48,987] {submission_evaluation.py:85} INFO - carts recall@60 = 0.28187638469724197
# [2022-12-27 23:52:53,701] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.46307842274539146
# [2022-12-27 23:52:53,703] {submission_evaluation.py:84} INFO - orders hits@60 = 10200 / gt@60 = 31617
# [2022-12-27 23:52:53,703] {submission_evaluation.py:85} INFO - orders recall@60 = 0.3226112534396053
# [2022-12-27 23:52:53,703] {submission_evaluation.py:91} INFO - =============
# [2022-12-27 23:52:53,703] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3199643156089538
# [2022-12-27 23:52:53,704] {submission_evaluation.py:93} INFO - =============
# [2022-12-27 23:52:59,259] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.42540939253268023
# [2022-12-27 23:52:59,259] {submission_evaluation.py:84} INFO - clicks hits@70 = 74687 / gt@70 = 175565
# [2022-12-27 23:52:59,259] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.42540939253268023
# [2022-12-27 23:53:04,664] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.40438275064216544
# [2022-12-27 23:53:04,665] {submission_evaluation.py:84} INFO - carts hits@70 = 16364 / gt@70 = 57323
# [2022-12-27 23:53:04,666] {submission_evaluation.py:85} INFO - carts recall@70 = 0.28547005564956474
# [2022-12-27 23:53:08,665] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.467507543749534
# [2022-12-27 23:53:08,666] {submission_evaluation.py:84} INFO - orders hits@70 = 10299 / gt@70 = 31617
# [2022-12-27 23:53:08,666] {submission_evaluation.py:85} INFO - orders recall@70 = 0.325742480311225
# [2022-12-27 23:53:08,666] {submission_evaluation.py:91} INFO - =============
# [2022-12-27 23:53:08,666] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.32362744413487243
# [2022-12-27 23:53:08,666] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 50wdw 5negative retrieval vec_size 32 + real session cutoff 2hr
# 2022-12-22 21:12:32,295] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.3590799569370632
# [2022-12-22 21:12:32,296] {submission_evaluation.py:84} INFO - clicks hits@20 = 63039 / gt@20 = 175557
# [2022-12-22 21:12:32,296] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3590799569370632
# [2022-12-22 21:12:33,945] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.42781815081914243
# [2022-12-22 21:12:33,945] {submission_evaluation.py:84} INFO - carts hits@20 = 17248 / gt@20 = 57718
# [2022-12-22 21:12:33,945] {submission_evaluation.py:85} INFO - carts recall@20 = 0.29883225336983266
# [2022-12-22 21:12:36,071] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4444481086299281
# [2022-12-22 21:12:36,071] {submission_evaluation.py:84} INFO - orders hits@20 = 9367 / gt@20 = 31058
# [2022-12-22 21:12:36,071] {submission_evaluation.py:85} INFO - orders recall@20 = 0.30159701204198597
# [2022-12-22 21:12:36,071] {submission_evaluation.py:91} INFO - =============
# [2022-12-22 21:12:36,071] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.30651587892984766
# [2022-12-22 21:12:36,071] {submission_evaluation.py:93} INFO - =============
# [2022-12-22 21:12:40,081] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.4032194671815991
# [2022-12-22 21:12:40,082] {submission_evaluation.py:84} INFO - clicks hits@40 = 70788 / gt@40 = 175557
# [2022-12-22 21:12:40,082] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.4032194671815991
# [2022-12-22 21:12:42,962] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4587016968898009
# [2022-12-22 21:12:42,964] {submission_evaluation.py:84} INFO - carts hits@40 = 18870 / gt@40 = 57718
# [2022-12-22 21:12:42,964] {submission_evaluation.py:85} INFO - carts recall@40 = 0.32693440521154576
# [2022-12-22 21:12:45,806] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.47101571948111
# [2022-12-22 21:12:45,808] {submission_evaluation.py:84} INFO - orders hits@40 = 10104 / gt@40 = 31058
# [2022-12-22 21:12:45,808] {submission_evaluation.py:85} INFO - orders recall@40 = 0.32532680790778545
# [2022-12-22 21:12:45,808] {submission_evaluation.py:91} INFO - =============
# [2022-12-22 21:12:45,808] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3335983530262949
# [2022-12-22 21:12:45,808] {submission_evaluation.py:93} INFO - =============
# [2022-12-22 21:12:50,691] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.42956988328577045
# [2022-12-22 21:12:50,691] {submission_evaluation.py:84} INFO - clicks hits@60 = 75414 / gt@60 = 175557
# [2022-12-22 21:12:50,691] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.42956988328577045
# [2022-12-22 21:12:53,653] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.4765383502428261
# [2022-12-22 21:12:53,653] {submission_evaluation.py:84} INFO - carts hits@60 = 19806 / gt@60 = 57718
# [2022-12-22 21:12:53,653] {submission_evaluation.py:85} INFO - carts recall@60 = 0.3431511833396861
# [2022-12-22 21:12:56,750] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.4872490446905911
# [2022-12-22 21:12:56,750] {submission_evaluation.py:84} INFO - orders hits@60 = 10567 / gt@60 = 31058
# [2022-12-22 21:12:56,751] {submission_evaluation.py:85} INFO - orders recall@60 = 0.34023440015454953
# [2022-12-22 21:12:56,751] {submission_evaluation.py:91} INFO - =============
# [2022-12-22 21:12:56,751] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3500429834232126
# [2022-12-22 21:12:56,751] {submission_evaluation.py:93} INFO - =============
# [2022-12-22 21:13:00,898] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.4373394396122057
# [2022-12-22 21:13:00,898] {submission_evaluation.py:84} INFO - clicks hits@70 = 76778 / gt@70 = 175557
# [2022-12-22 21:13:00,898] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.4373394396122057
# [2022-12-22 21:13:04,587] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.4809390512572721
# [2022-12-22 21:13:04,587] {submission_evaluation.py:84} INFO - carts hits@70 = 20026 / gt@70 = 57718
# [2022-12-22 21:13:04,587] {submission_evaluation.py:85} INFO - carts recall@70 = 0.34696281922450534
# [2022-12-22 21:13:07,684] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.4908584256942733
# [2022-12-22 21:13:07,684] {submission_evaluation.py:84} INFO - orders hits@70 = 10653 / gt@70 = 31058
# [2022-12-22 21:13:07,684] {submission_evaluation.py:85} INFO - orders recall@70 = 0.3430034129692833
# [2022-12-22 21:13:07,684] {submission_evaluation.py:91} INFO - =============
# [2022-12-22 21:13:07,685] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.3536248375101421
# [2022-12-22 21:13:07,685] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 50wdw 5negative retrieval vec_size 32 + real session cutoff 1day
# 2022-12-23 10:10:12,842] {submission_evaluation.py:38} INFO - create prediction type column
# [2022-12-23 10:10:17,715] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.3654599717501253
# [2022-12-23 10:10:17,716] {submission_evaluation.py:84} INFO - clicks hits@20 = 64166 / gt@20 = 175576
# [2022-12-23 10:10:17,716] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3654599717501253
# [2022-12-23 10:10:22,446] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.43870464533107456
# [2022-12-23 10:10:22,449] {submission_evaluation.py:84} INFO - carts hits@20 = 17486 / gt@20 = 56950
# [2022-12-23 10:10:22,449] {submission_evaluation.py:85} INFO - carts recall@20 = 0.3070412642669008
# [2022-12-23 10:10:24,612] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4564872805394379
# [2022-12-23 10:10:24,612] {submission_evaluation.py:84} INFO - orders hits@20 = 9483 / gt@20 = 31064
# [2022-12-23 10:10:24,612] {submission_evaluation.py:85} INFO - orders recall@20 = 0.3052729848055627
# [2022-12-23 10:10:24,612] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 10:10:24,612] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.3118221673384204
# [2022-12-23 10:10:24,612] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 10:10:31,070] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.4088770674807491
# [2022-12-23 10:10:31,070] {submission_evaluation.py:84} INFO - clicks hits@40 = 71789 / gt@40 = 175576
# [2022-12-23 10:10:31,070] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.4088770674807491
# [2022-12-23 10:10:39,199] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4682975867655876
# [2022-12-23 10:10:39,203] {submission_evaluation.py:84} INFO - carts hits@40 = 19004 / gt@40 = 56950
# [2022-12-23 10:10:39,203] {submission_evaluation.py:85} INFO - carts recall@40 = 0.33369622475856014
# [2022-12-23 10:10:47,180] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.48069401007692597
# [2022-12-23 10:10:47,182] {submission_evaluation.py:84} INFO - orders hits@40 = 10175 / gt@40 = 31064
# [2022-12-23 10:10:47,182] {submission_evaluation.py:85} INFO - orders recall@40 = 0.32754957507082155
# [2022-12-23 10:10:47,182] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 10:10:47,182] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3375263192181359
# [2022-12-23 10:10:47,182] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 10:10:57,625] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.43478607554563264
# [2022-12-23 10:10:57,628] {submission_evaluation.py:84} INFO - clicks hits@60 = 76338 / gt@60 = 175576
# [2022-12-23 10:10:57,628] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.43478607554563264
# [2022-12-23 10:11:04,001] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.48672746776719655
# [2022-12-23 10:11:04,002] {submission_evaluation.py:84} INFO - carts hits@60 = 19958 / gt@60 = 56950
# [2022-12-23 10:11:04,002] {submission_evaluation.py:85} INFO - carts recall@60 = 0.35044776119402987
# [2022-12-23 10:11:13,549] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.4963828840943456
# [2022-12-23 10:11:13,550] {submission_evaluation.py:84} INFO - orders hits@60 = 10618 / gt@60 = 31064
# [2022-12-23 10:11:13,550] {submission_evaluation.py:85} INFO - orders recall@60 = 0.3418104558331187
# [2022-12-23 10:11:13,550] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 10:11:13,550] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.35369920941264343
# [2022-12-23 10:11:13,550] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 10:11:23,467] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.44214471226135693
# [2022-12-23 10:11:23,468] {submission_evaluation.py:84} INFO - clicks hits@70 = 77630 / gt@70 = 175576
# [2022-12-23 10:11:23,468] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.44214471226135693
# [2022-12-23 10:11:32,381] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.49080424927469324
# [2022-12-23 10:11:32,382] {submission_evaluation.py:84} INFO - carts hits@70 = 20169 / gt@70 = 56950
# [2022-12-23 10:11:32,382] {submission_evaluation.py:85} INFO - carts recall@70 = 0.35415276558384545
# [2022-12-23 10:11:40,007] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.49948088309864935
# [2022-12-23 10:11:40,010] {submission_evaluation.py:84} INFO - orders hits@70 = 10693 / gt@70 = 31064
# [2022-12-23 10:11:40,010] {submission_evaluation.py:85} INFO - orders recall@70 = 0.34422482616533606
# [2022-12-23 10:11:40,010] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 10:11:40,010] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.35699519660049095
# [2022-12-23 10:11:40,010] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 40wdw 5negative retrieval vec_size 32 + real session cutoff 1day
# [2022-12-23 14:35:11,320] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.364288602030532
# [2022-12-23 14:35:11,322] {submission_evaluation.py:84} INFO - clicks hits@20 = 63976 / gt@20 = 175619
# [2022-12-23 14:35:11,322] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.364288602030532
# [2022-12-23 14:35:14,890] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4340593158584558
# [2022-12-23 14:35:14,891] {submission_evaluation.py:84} INFO - carts hits@20 = 17133 / gt@20 = 56615
# [2022-12-23 14:35:14,892] {submission_evaluation.py:85} INFO - carts recall@20 = 0.3026229797756778
# [2022-12-23 14:35:20,745] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.45129010054750673
# [2022-12-23 14:35:20,747] {submission_evaluation.py:84} INFO - orders hits@20 = 9526 / gt@20 = 31453
# [2022-12-23 14:35:20,747] {submission_evaluation.py:85} INFO - orders recall@20 = 0.3028645916128827
# [2022-12-23 14:35:20,747] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 14:35:20,747] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.3089345091034862
# [2022-12-23 14:35:20,747] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 14:35:27,708] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.40782603249078975
# [2022-12-23 14:35:27,709] {submission_evaluation.py:84} INFO - clicks hits@40 = 71622 / gt@40 = 175619
# [2022-12-23 14:35:27,709] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.40782603249078975
# [2022-12-23 14:35:39,131] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4628330544688336
# [2022-12-23 14:35:39,147] {submission_evaluation.py:84} INFO - carts hits@40 = 18613 / gt@40 = 56615
# [2022-12-23 14:35:39,147] {submission_evaluation.py:85} INFO - carts recall@40 = 0.32876446171509316
# [2022-12-23 14:35:47,132] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.47610733522307896
# [2022-12-23 14:35:47,136] {submission_evaluation.py:84} INFO - orders hits@40 = 10236 / gt@40 = 31453
# [2022-12-23 14:35:47,136] {submission_evaluation.py:85} INFO - orders recall@40 = 0.32543795504403394
# [2022-12-23 14:35:47,137] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 14:35:47,137] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3346747147900273
# [2022-12-23 14:35:47,137] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 14:35:59,492] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.4331194232970237
# [2022-12-23 14:35:59,500] {submission_evaluation.py:84} INFO - clicks hits@60 = 76064 / gt@60 = 175619
# [2022-12-23 14:35:59,500] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.4331194232970237
# [2022-12-23 14:36:04,536] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.48114413451408694
# [2022-12-23 14:36:04,539] {submission_evaluation.py:84} INFO - carts hits@60 = 19562 / gt@60 = 56615
# [2022-12-23 14:36:04,539] {submission_evaluation.py:85} INFO - carts recall@60 = 0.34552680385056966
# [2022-12-23 14:36:15,379] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.49239259093653504
# [2022-12-23 14:36:15,383] {submission_evaluation.py:84} INFO - orders hits@60 = 10687 / gt@60 = 31453
# [2022-12-23 14:36:15,383] {submission_evaluation.py:85} INFO - orders recall@60 = 0.3397768098432582
# [2022-12-23 14:36:15,383] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 14:36:15,383] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.35083606939082823
# [2022-12-23 14:36:15,384] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 14:36:27,815] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.44061291773669137
# [2022-12-23 14:36:27,821] {submission_evaluation.py:84} INFO - clicks hits@70 = 77380 / gt@70 = 175619
# [2022-12-23 14:36:27,821] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.44061291773669137
# [2022-12-23 14:36:41,295] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.4854494052352648
# [2022-12-23 14:36:41,301] {submission_evaluation.py:84} INFO - carts hits@70 = 19766 / gt@70 = 56615
# [2022-12-23 14:36:41,301] {submission_evaluation.py:85} INFO - carts recall@70 = 0.34913008919897554
# [2022-12-23 14:36:47,937] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.4949924659738018
# [2022-12-23 14:36:47,941] {submission_evaluation.py:84} INFO - orders hits@70 = 10754 / gt@70 = 31453
# [2022-12-23 14:36:47,941] {submission_evaluation.py:85} INFO - orders recall@70 = 0.34190697230788797
# [2022-12-23 14:36:47,941] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 14:36:47,941] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.35394450191809457
# [2022-12-23 14:36:47,941] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 30 wdw 10 negative retrieval vec_size 32 + real session cutoff 1day


# covisit + word2vec skipgram 50wdw 5negative retrieval vec_size 64 - euclidean dist
# [2022-12-19 21:24:47,021] {submission_evaluation.py:84} INFO - clicks hits@20 = 67466 / gt@20 = 175541
# [2022-12-19 21:24:47,021] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.38433186549011344
# [2022-12-19 21:24:48,782] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4455490904963889
# [2022-12-19 21:24:48,782] {submission_evaluation.py:84} INFO - carts hits@20 = 17896 / gt@20 = 57134
# [2022-12-19 21:24:48,782] {submission_evaluation.py:85} INFO - carts recall@20 = 0.31322855042531594
# [2022-12-19 21:24:50,944] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4601631160435077
# [2022-12-19 21:24:50,944] {submission_evaluation.py:84} INFO - orders hits@20 = 9748 / gt@20 = 31150
# [2022-12-19 21:24:50,944] {submission_evaluation.py:85} INFO - orders recall@20 = 0.3129373996789727
# [2022-12-19 21:24:50,944] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:24:50,944] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.3201641914839898
# [2022-12-19 21:24:50,944] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 21:24:59,341] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.4295178904073692
# [2022-12-19 21:24:59,344] {submission_evaluation.py:84} INFO - clicks hits@40 = 75398 / gt@40 = 175541
# [2022-12-19 21:24:59,344] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.4295178904073692
# [2022-12-19 21:25:05,266] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4750405735448451
# [2022-12-19 21:25:05,267] {submission_evaluation.py:84} INFO - carts hits@40 = 19471 / gt@40 = 57134
# [2022-12-19 21:25:05,267] {submission_evaluation.py:85} INFO - carts recall@40 = 0.34079532327510764
# [2022-12-19 21:25:10,473] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.48591544458447933
# [2022-12-19 21:25:10,474] {submission_evaluation.py:84} INFO - orders hits@40 = 10489 / gt@40 = 31150
# [2022-12-19 21:25:10,474] {submission_evaluation.py:85} INFO - orders recall@40 = 0.3367255216693419
# [2022-12-19 21:25:10,474] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:25:10,475] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3472256990248743
# [2022-12-19 21:25:10,475] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 21:25:22,406] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@50 = 0.4440558046268393
# [2022-12-19 21:25:22,433] {submission_evaluation.py:84} INFO - clicks hits@50 = 77950 / gt@50 = 175541
# [2022-12-19 21:25:22,433] {submission_evaluation.py:85} INFO - clicks recall@50 = 0.4440558046268393
# [2022-12-19 21:25:26,620] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@50 = 0.4844134802547137
# [2022-12-19 21:25:26,620] {submission_evaluation.py:84} INFO - carts hits@50 = 20022 / gt@50 = 57134
# [2022-12-19 21:25:26,620] {submission_evaluation.py:85} INFO - carts recall@50 = 0.3504393180943046
# [2022-12-19 21:25:31,537] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@50 = 0.4943729189016543
# [2022-12-19 21:25:31,538] {submission_evaluation.py:84} INFO - orders hits@50 = 10743 / gt@50 = 31150
# [2022-12-19 21:25:31,538] {submission_evaluation.py:85} INFO - orders recall@50 = 0.3448796147672552
# [2022-12-19 21:25:31,538] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:25:31,539] {submission_evaluation.py:92} INFO - Overall Recall@50 = 0.3564651447513284
# [2022-12-19 21:25:31,539] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 21:25:37,995] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.45568271799750487
# [2022-12-19 21:25:37,997] {submission_evaluation.py:84} INFO - clicks hits@60 = 79991 / gt@60 = 175541
# [2022-12-19 21:25:37,997] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.45568271799750487
# [2022-12-19 21:25:44,476] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.4930900583138464
# [2022-12-19 21:25:44,477] {submission_evaluation.py:84} INFO - carts hits@60 = 20455 / gt@60 = 57134
# [2022-12-19 21:25:44,477] {submission_evaluation.py:85} INFO - carts recall@60 = 0.3580179927888823
# [2022-12-19 21:25:48,166] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.5021116226962318
# [2022-12-19 21:25:48,167] {submission_evaluation.py:84} INFO - orders hits@60 = 10946 / gt@60 = 31150
# [2022-12-19 21:25:48,167] {submission_evaluation.py:85} INFO - orders recall@60 = 0.3513964686998395
# [2022-12-19 21:25:48,167] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:25:48,167] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3638115508563189
# [2022-12-19 21:25:48,167] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 21:25:55,026] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.4627579881623097
# [2022-12-19 21:25:55,027] {submission_evaluation.py:84} INFO - clicks hits@70 = 81233 / gt@70 = 175541
# [2022-12-19 21:25:55,027] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.4627579881623097
# [2022-12-19 21:25:58,904] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.497543240436091
# [2022-12-19 21:25:58,904] {submission_evaluation.py:84} INFO - carts hits@70 = 20678 / gt@70 = 57134
# [2022-12-19 21:25:58,904] {submission_evaluation.py:85} INFO - carts recall@70 = 0.36192109777015435
# [2022-12-19 21:26:02,835] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.5059946501124905
# [2022-12-19 21:26:02,836] {submission_evaluation.py:84} INFO - orders hits@70 = 11038 / gt@70 = 31150
# [2022-12-19 21:26:02,836] {submission_evaluation.py:85} INFO - orders recall@70 = 0.35434991974317814
# [2022-12-19 21:26:02,836] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:26:02,836] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.36746207999318414
# [2022-12-19 21:26:02,836] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 50wdw 5negative retrieval vec_size 64 - angular dist
# [2022-12-19 21:29:33,990] {submission_evaluation.py:84} INFO - clicks hits@20 = 68073 / gt@20 = 175552
# [2022-12-19 21:29:33,990] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3877654484141451
# [2022-12-19 21:29:35,494] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4453916113176386
# [2022-12-19 21:29:35,495] {submission_evaluation.py:84} INFO - carts hits@20 = 18046 / gt@20 = 58036
# [2022-12-19 21:29:35,495] {submission_evaluation.py:85} INFO - carts recall@20 = 0.3109449307326487
# [2022-12-19 21:29:37,253] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.46521974198146826
# [2022-12-19 21:29:37,253] {submission_evaluation.py:84} INFO - orders hits@20 = 10032 / gt@20 = 31565
# [2022-12-19 21:29:37,253] {submission_evaluation.py:85} INFO - orders recall@20 = 0.31782037066370983
# [2022-12-19 21:29:37,253] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:29:37,254] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.322752246459435
# [2022-12-19 21:29:37,254] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 21:29:40,309] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.43322206525701784
# [2022-12-19 21:29:40,309] {submission_evaluation.py:84} INFO - clicks hits@40 = 76053 / gt@40 = 175552
# [2022-12-19 21:29:40,309] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.43322206525701784
# [2022-12-19 21:29:42,999] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4763409770091859
# [2022-12-19 21:29:43,002] {submission_evaluation.py:84} INFO - carts hits@40 = 19687 / gt@40 = 58036
# [2022-12-19 21:29:43,002] {submission_evaluation.py:85} INFO - carts recall@40 = 0.33922048383761805
# [2022-12-19 21:29:45,600] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.49217802636906055
# [2022-12-19 21:29:45,600] {submission_evaluation.py:84} INFO - orders hits@40 = 10820 / gt@40 = 31565
# [2022-12-19 21:29:45,600] {submission_evaluation.py:85} INFO - orders recall@40 = 0.3427847299223824
# [2022-12-19 21:29:45,600] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:29:45,600] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.35075918963041663
# [2022-12-19 21:29:45,600] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 21:29:49,146] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@50 = 0.44791286912139994
# [2022-12-19 21:29:49,147] {submission_evaluation.py:84} INFO - clicks hits@50 = 78632 / gt@50 = 175552
# [2022-12-19 21:29:49,147] {submission_evaluation.py:85} INFO - clicks recall@50 = 0.44791286912139994
# [2022-12-19 21:29:51,852] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@50 = 0.48798800077590127
# [2022-12-19 21:29:51,853] {submission_evaluation.py:84} INFO - carts hits@50 = 20304 / gt@50 = 58036
# [2022-12-19 21:29:51,853] {submission_evaluation.py:85} INFO - carts recall@50 = 0.34985181611413607
# [2022-12-19 21:29:55,049] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@50 = 0.5017848690689719
# [2022-12-19 21:29:55,050] {submission_evaluation.py:84} INFO - orders hits@50 = 11104 / gt@50 = 31565
# [2022-12-19 21:29:55,050] {submission_evaluation.py:85} INFO - orders recall@50 = 0.35178203706637096
# [2022-12-19 21:29:55,050] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:29:55,050] {submission_evaluation.py:92} INFO - Overall Recall@50 = 0.3608160539862034
# [2022-12-19 21:29:55,050] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 21:30:00,796] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.45986374407582936
# [2022-12-19 21:30:00,797] {submission_evaluation.py:84} INFO - clicks hits@60 = 80730 / gt@60 = 175552
# [2022-12-19 21:30:00,797] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.45986374407582936
# [2022-12-19 21:30:05,329] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.49637971165293515
# [2022-12-19 21:30:05,329] {submission_evaluation.py:84} INFO - carts hits@60 = 20756 / gt@60 = 58036
# [2022-12-19 21:30:05,329] {submission_evaluation.py:85} INFO - carts recall@60 = 0.3576400854641946
# [2022-12-19 21:30:08,406] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.5078536229639543
# [2022-12-19 21:30:08,407] {submission_evaluation.py:84} INFO - orders hits@60 = 11280 / gt@60 = 31565
# [2022-12-19 21:30:08,407] {submission_evaluation.py:85} INFO - orders recall@60 = 0.3573578330429273
# [2022-12-19 21:30:08,407] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:30:08,407] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3676930998725977
# [2022-12-19 21:30:08,407] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 21:30:16,388] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.46731452788917244
# [2022-12-19 21:30:16,388] {submission_evaluation.py:84} INFO - clicks hits@70 = 82038 / gt@70 = 175552
# [2022-12-19 21:30:16,388] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.46731452788917244
# [2022-12-19 21:30:21,546] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.5010715375973228
# [2022-12-19 21:30:21,547] {submission_evaluation.py:84} INFO - carts hits@70 = 20987 / gt@70 = 58036
# [2022-12-19 21:30:21,547] {submission_evaluation.py:85} INFO - carts recall@70 = 0.36162037356123783
# [2022-12-19 21:30:25,602] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.5113061224838147
# [2022-12-19 21:30:25,603] {submission_evaluation.py:84} INFO - orders hits@70 = 11371 / gt@70 = 31565
# [2022-12-19 21:30:25,603] {submission_evaluation.py:85} INFO - orders recall@70 = 0.36024077300807855
# [2022-12-19 21:30:25,603] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 21:30:25,603] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.37136202866213575
# [2022-12-19 21:30:25,603] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 50wdw 5negative retrieval vec_size 64 - angular dist + cart real_session & buy2buy embedding
# [2022-12-22 21:30:17,441] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.38872455987370413
# [2022-12-22 21:30:17,441] {submission_evaluation.py:84} INFO - clicks hits@20 = 68206 / gt@20 = 175461
# [2022-12-22 21:30:17,477] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.38872455987370413
# [2022-12-22 21:30:23,541] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.42925397080659655
# [2022-12-22 21:30:23,543] {submission_evaluation.py:84} INFO - carts hits@20 = 17230 / gt@20 = 58080
# [2022-12-22 21:30:23,544] {submission_evaluation.py:85} INFO - carts recall@20 = 0.29665977961432505
# [2022-12-22 21:30:25,532] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.4001580296917653
# [2022-12-22 21:30:25,532] {submission_evaluation.py:84} INFO - orders hits@20 = 8375 / gt@20 = 31077
# [2022-12-22 21:30:25,532] {submission_evaluation.py:85} INFO - orders recall@20 = 0.2694919071982495
# [2022-12-22 21:30:25,532] {submission_evaluation.py:91} INFO - =============
# [2022-12-22 21:30:25,532] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.2895655341906176
# [2022-12-22 21:30:25,532] {submission_evaluation.py:93} INFO - =============
# [2022-12-22 21:30:31,106] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.4347176865514274
# [2022-12-22 21:30:31,106] {submission_evaluation.py:84} INFO - clicks hits@40 = 76276 / gt@40 = 175461
# [2022-12-22 21:30:31,106] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.4347176865514274
# [2022-12-22 21:30:37,223] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.456908629411233
# [2022-12-22 21:30:37,223] {submission_evaluation.py:84} INFO - carts hits@40 = 18661 / gt@40 = 58080
# [2022-12-22 21:30:37,223] {submission_evaluation.py:85} INFO - carts recall@40 = 0.3212982093663912
# [2022-12-22 21:30:40,061] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.41399186724436277
# [2022-12-22 21:30:40,062] {submission_evaluation.py:84} INFO - orders hits@40 = 8776 / gt@40 = 31077
# [2022-12-22 21:30:40,062] {submission_evaluation.py:85} INFO - orders recall@40 = 0.28239534060559257
# [2022-12-22 21:30:40,062] {submission_evaluation.py:91} INFO - =============
# [2022-12-22 21:30:40,062] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.30929843582841565
# [2022-12-22 21:30:40,062] {submission_evaluation.py:93} INFO - =============
# [2022-12-22 21:30:49,401] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.461213603022894
# [2022-12-22 21:30:49,402] {submission_evaluation.py:84} INFO - clicks hits@60 = 80925 / gt@60 = 175461
# [2022-12-22 21:30:49,402] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.461213603022894
# [2022-12-22 21:30:53,256] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.4734347937005295
# [2022-12-22 21:30:53,256] {submission_evaluation.py:84} INFO - carts hits@60 = 19547 / gt@60 = 58080
# [2022-12-22 21:30:53,256] {submission_evaluation.py:85} INFO - carts recall@60 = 0.3365530303030303
# [2022-12-22 21:31:01,411] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.42179474114348986
# [2022-12-22 21:31:01,412] {submission_evaluation.py:84} INFO - orders hits@60 = 9004 / gt@60 = 31077
# [2022-12-22 21:31:01,413] {submission_evaluation.py:85} INFO - orders recall@60 = 0.28973195610901953
# [2022-12-22 21:31:01,413] {submission_evaluation.py:91} INFO - =============
# [2022-12-22 21:31:01,413] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3209264430586102
# [2022-12-22 21:31:01,413] {submission_evaluation.py:93} INFO - =============
# [2022-12-22 21:31:06,245] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.46801283476100103
# [2022-12-22 21:31:06,246] {submission_evaluation.py:84} INFO - clicks hits@70 = 82118 / gt@70 = 175461
# [2022-12-22 21:31:06,246] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.46801283476100103
# [2022-12-22 21:31:14,522] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.4773911992418204
# [2022-12-22 21:31:14,523] {submission_evaluation.py:84} INFO - carts hits@70 = 19744 / gt@70 = 58080
# [2022-12-22 21:31:14,523] {submission_evaluation.py:85} INFO - carts recall@70 = 0.3399449035812672
# [2022-12-22 21:31:18,788] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.4239303636964962
# [2022-12-22 21:31:18,788] {submission_evaluation.py:84} INFO - orders hits@70 = 9061 / gt@70 = 31077
# [2022-12-22 21:31:18,788] {submission_evaluation.py:85} INFO - orders recall@70 = 0.29156610998487625
# [2022-12-22 21:31:18,788] {submission_evaluation.py:91} INFO - =============
# [2022-12-22 21:31:18,789] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.32372442054140604
# [2022-12-22 21:31:18,789] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 50wdw 30 negative retrieval vec_size 64 - angular dist
# [2022-12-20 18:05:34,346] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.3873399545291373
# [2022-12-20 18:05:34,347] {submission_evaluation.py:84} INFO - clicks hits@20 = 67977 / gt@20 = 175497
# [2022-12-20 18:05:34,347] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3873399545291373
# [2022-12-20 18:05:36,183] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.4499505386293092
# [2022-12-20 18:05:36,183] {submission_evaluation.py:84} INFO - carts hits@20 = 18094 / gt@20 = 57626
# [2022-12-20 18:05:36,183] {submission_evaluation.py:85} INFO - carts recall@20 = 0.3139902127511887
# [2022-12-20 18:05:38,052] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.46002501898141784
# [2022-12-20 18:05:38,053] {submission_evaluation.py:84} INFO - orders hits@20 = 9792 / gt@20 = 30900
# [2022-12-20 18:05:38,053] {submission_evaluation.py:85} INFO - orders recall@20 = 0.3168932038834951
# [2022-12-20 18:05:38,053] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 18:05:38,053] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.32306698160836744
# [2022-12-20 18:05:38,053] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 18:05:41,262] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.43386496635270116
# [2022-12-20 18:05:41,263] {submission_evaluation.py:84} INFO - clicks hits@40 = 76142 / gt@40 = 175497
# [2022-12-20 18:05:41,263] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.43386496635270116
# [2022-12-20 18:05:44,073] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4826938851900966
# [2022-12-20 18:05:44,073] {submission_evaluation.py:84} INFO - carts hits@40 = 19851 / gt@40 = 57626
# [2022-12-20 18:05:44,073] {submission_evaluation.py:85} INFO - carts recall@40 = 0.3444799222573144
# [2022-12-20 18:05:46,745] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.4877095730893552
# [2022-12-20 18:05:46,745] {submission_evaluation.py:84} INFO - orders hits@40 = 10573 / gt@40 = 30900
# [2022-12-20 18:05:46,745] {submission_evaluation.py:85} INFO - orders recall@40 = 0.342168284789644
# [2022-12-20 18:05:46,745] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 18:05:46,745] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3520314441862509
# [2022-12-20 18:05:46,745] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 18:05:51,060] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.46005914631019335
# [2022-12-20 18:05:51,061] {submission_evaluation.py:84} INFO - clicks hits@60 = 80739 / gt@60 = 175497
# [2022-12-20 18:05:51,061] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.46005914631019335
# [2022-12-20 18:05:54,157] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.5015429483097902
# [2022-12-20 18:05:54,157] {submission_evaluation.py:84} INFO - carts hits@60 = 20862 / gt@60 = 57626
# [2022-12-20 18:05:54,158] {submission_evaluation.py:85} INFO - carts recall@60 = 0.3620240863499115
# [2022-12-20 18:05:59,365] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.5039239147869163
# [2022-12-20 18:05:59,368] {submission_evaluation.py:84} INFO - orders hits@60 = 11047 / gt@60 = 30900
# [2022-12-20 18:05:59,369] {submission_evaluation.py:85} INFO - orders recall@60 = 0.35750809061488675
# [2022-12-20 18:05:59,370] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 18:05:59,370] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.3691179949049248
# [2022-12-20 18:05:59,370] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 18:06:04,754] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.467369812589389
# [2022-12-20 18:06:04,754] {submission_evaluation.py:84} INFO - clicks hits@70 = 82022 / gt@70 = 175497
# [2022-12-20 18:06:04,754] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.467369812589389
# [2022-12-20 18:06:08,743] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.5057134048191754
# [2022-12-20 18:06:08,744] {submission_evaluation.py:84} INFO - carts hits@70 = 21067 / gt@70 = 57626
# [2022-12-20 18:06:08,744] {submission_evaluation.py:85} INFO - carts recall@70 = 0.3655815083469267
# [2022-12-20 18:06:12,134] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.5078077626059122
# [2022-12-20 18:06:12,134] {submission_evaluation.py:84} INFO - orders hits@70 = 11141 / gt@70 = 30900
# [2022-12-20 18:06:12,134] {submission_evaluation.py:85} INFO - orders recall@70 = 0.3605501618122977
# [2022-12-20 18:06:12,135] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 18:06:12,135] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.3727415308503955
# [2022-12-20 18:06:12,135] {submission_evaluation.py:93} INFO - =============

# covisit + word2vec skipgram 50wdw 5negative retrieval vec_size 64 - angular dist + dropn1
# [2022-12-20 10:17:57,281] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.3845066891537708
# [2022-12-20 10:17:57,282] {submission_evaluation.py:84} INFO - clicks hits@20 = 67484 / gt@20 = 175508
# [2022-12-20 10:17:57,282] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.3845066891537708
# [2022-12-20 10:17:59,763] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.44902454199442277
# [2022-12-20 10:17:59,763] {submission_evaluation.py:84} INFO - carts hits@20 = 18081 / gt@20 = 57374
# [2022-12-20 10:17:59,763] {submission_evaluation.py:85} INFO - carts recall@20 = 0.31514274758601457
# [2022-12-20 10:18:02,364] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.46090110864599476
# [2022-12-20 10:18:02,364] {submission_evaluation.py:84} INFO - orders hits@20 = 9811 / gt@20 = 31463
# [2022-12-20 10:18:02,364] {submission_evaluation.py:85} INFO - orders recall@20 = 0.3118265899628135
# [2022-12-20 10:18:02,364] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 10:18:02,364] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.32008944716886956
# [2022-12-20 10:18:02,364] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 10:18:06,908] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.43070971123823415
# [2022-12-20 10:18:06,908] {submission_evaluation.py:84} INFO - clicks hits@40 = 75593 / gt@40 = 175508
# [2022-12-20 10:18:06,909] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.43070971123823415
# [2022-12-20 10:18:11,042] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.4816679292977757
# [2022-12-20 10:18:11,043] {submission_evaluation.py:84} INFO - carts hits@40 = 19782 / gt@40 = 57374
# [2022-12-20 10:18:11,043] {submission_evaluation.py:85} INFO - carts recall@40 = 0.34479032314288705
# [2022-12-20 10:18:15,305] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.48742090170777624
# [2022-12-20 10:18:15,306] {submission_evaluation.py:84} INFO - orders hits@40 = 10567 / gt@40 = 31463
# [2022-12-20 10:18:15,306] {submission_evaluation.py:85} INFO - orders recall@40 = 0.3358548135905667
# [2022-12-20 10:18:15,306] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 10:18:15,306] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.3480209562210296
# [2022-12-20 10:18:15,306] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 10:18:21,180] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@50 = 0.4460366479020899
# [2022-12-20 10:18:21,181] {submission_evaluation.py:84} INFO - clicks hits@50 = 78283 / gt@50 = 175508
# [2022-12-20 10:18:21,182] {submission_evaluation.py:85} INFO - clicks recall@50 = 0.4460366479020899
# [2022-12-20 10:18:25,370] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@50 = 0.49107999987607526
# [2022-12-20 10:18:25,371] {submission_evaluation.py:84} INFO - carts hits@50 = 20311 / gt@50 = 57374
# [2022-12-20 10:18:25,371] {submission_evaluation.py:85} INFO - carts recall@50 = 0.35401052741659983
# [2022-12-20 10:18:28,858] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@50 = 0.49574447218455187
# [2022-12-20 10:18:28,859] {submission_evaluation.py:84} INFO - orders hits@50 = 10824 / gt@50 = 31463
# [2022-12-20 10:18:28,859] {submission_evaluation.py:85} INFO - orders recall@50 = 0.34402313828941933
# [2022-12-20 10:18:28,859] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 10:18:28,859] {submission_evaluation.py:92} INFO - Overall Recall@50 = 0.35722070598884054
# [2022-12-20 10:18:28,859] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 10:18:34,262] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.45776260911183536
# [2022-12-20 10:18:34,264] {submission_evaluation.py:84} INFO - clicks hits@60 = 80341 / gt@60 = 175508
# [2022-12-20 10:18:34,264] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.45776260911183536
# [2022-12-20 10:18:42,349] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.49917279575428103
# [2022-12-20 10:18:42,351] {submission_evaluation.py:84} INFO - carts hits@60 = 20741 / gt@60 = 57374
# [2022-12-20 10:18:42,351] {submission_evaluation.py:85} INFO - carts recall@60 = 0.3615052114198069
# [2022-12-20 10:18:47,326] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.50213128069458
# [2022-12-20 10:18:47,328] {submission_evaluation.py:84} INFO - orders hits@60 = 11006 / gt@60 = 31463
# [2022-12-20 10:18:47,328] {submission_evaluation.py:85} INFO - orders recall@60 = 0.3498077106442488
# [2022-12-20 10:18:47,328] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 10:18:47,328] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.36411245072367493
# [2022-12-20 10:18:47,328] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 10:18:59,894] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.46477083665701846
# [2022-12-20 10:18:59,896] {submission_evaluation.py:84} INFO - clicks hits@70 = 81571 / gt@70 = 175508
# [2022-12-20 10:18:59,896] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.46477083665701846
# [2022-12-20 10:19:08,905] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.503860872022231
# [2022-12-20 10:19:08,906] {submission_evaluation.py:84} INFO - carts hits@70 = 20971 / gt@70 = 57374
# [2022-12-20 10:19:08,906] {submission_evaluation.py:85} INFO - carts recall@70 = 0.3655139958866385
# [2022-12-20 10:19:14,282] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.5063673383324263
# [2022-12-20 10:19:14,283] {submission_evaluation.py:84} INFO - orders hits@70 = 11108 / gt@70 = 31463
# [2022-12-20 10:19:14,283] {submission_evaluation.py:85} INFO - orders recall@70 = 0.35304961383212025
# [2022-12-20 10:19:14,283] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 10:19:14,283] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.3679610507309655
# [2022-12-20 10:19:14,283] {submission_evaluation.py:93} INFO - =============

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
