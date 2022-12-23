import click
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_test_candidates_dir,
    get_processed_training_train_candidates_dir,
    get_processed_local_validation_dir,
)
from src.pipeline.make_candidates_rows import get_ses2candidates
from src.metrics.submission_evaluation import measure_recall
from src.utils.logger import get_logger

logging = get_logger()


def concat_candidates(
    unique_sessions: list,
    event: str,
    covisit_ses2candidates: dict,
    fasttext_ses2candidates: dict,
    word2vec_ses2candidates: dict,
    matrix_fact_ses2candidates: dict,
    popular_daily_ses2candidates: dict,
    popular_hour_ses2candidates: dict,
    popular_datehour_ses2candidates: dict,
):
    labels = []
    sessions = []
    for session in unique_sessions:
        # get candidates for specific session
        # covisitation candidates
        cands = list(covisit_ses2candidates[session])
        # word2vec candidates
        word2vec_cands = list(word2vec_ses2candidates[session])
        cands.extend(word2vec_cands)
        # fasttext candidates
        fasttext_cands = list(fasttext_ses2candidates[session])
        cands.extend(fasttext_cands)
        # matrix fact candidates
        matrix_fact_cands = list(matrix_fact_ses2candidates[session])
        cands.extend(matrix_fact_cands)
        # popular datehour
        popular_datehour_cands = list(popular_datehour_ses2candidates[session])
        cands.extend(popular_datehour_cands)
        # popular hour
        popular_hour_cands = list(popular_hour_ses2candidates[session])
        cands.extend(popular_hour_cands)
        # popular daily
        popular_daily_cands = list(popular_daily_ses2candidates[session])
        cands.extend(popular_daily_cands)
        # drop duplicate
        # cands = set(cands)
        # convert back to list
        # cands = list(cands)
        labels.append(cands)
        sessions.append(f"{session}_{event}")

    # save as df
    data = {
        "session": sessions,
        "labels": labels,
        "raw_session": unique_sessions,
    }

    # save as pl dataframe
    df = pl.from_dict(data)
    return df


def eval_candidate_list(
    name: str,
    input_path: Path,
    gt_path: str,
):
    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    df_pred = pl.DataFrame()
    for ix in tqdm(range(20)):
        for event in ["clicks", "carts", "orders"]:
            # candidate #1 covisitation
            filepath = f"{input_path}/{name}_{ix}_{event}_list.parquet"
            cand_df = pd.read_parquet(filepath)
            covisit_ses2candidates = get_ses2candidates(cand_df)
            unique_sessions = list(cand_df["session"].values)

            del cand_df
            gc.collect()

            # candidate #2 fasttext
            filepath = f"{input_path}/{name}_{ix}_{event}_fasttext_list.parquet"
            cand_df = pd.read_parquet(filepath)
            fasttext_ses2candidates = get_ses2candidates(cand_df)

            del cand_df
            gc.collect()

            # candidate #3 word2vec
            filepath = f"{input_path}/{name}_{ix}_{event}_word2vec_list.parquet"
            cand_df = pd.read_parquet(filepath)
            word2vec_ses2candidates = get_ses2candidates(cand_df)

            del cand_df
            gc.collect()

            # candidate #3 matrix factorization
            filepath = f"{input_path}/{name}_{ix}_{event}_matrix_fact_list.parquet"
            cand_df = pd.read_parquet(filepath)
            matrix_fact_ses2candidates = get_ses2candidates(cand_df)

            del cand_df
            gc.collect()

            # candidate #4 popular daily
            filepath = f"{input_path}/{name}_{ix}_{event}_popular_daily_list.parquet"
            cand_df = pd.read_parquet(filepath)
            popular_daily_ses2candidates = get_ses2candidates(cand_df)

            # candidate #5 popular hour
            filepath = f"{input_path}/{name}_{ix}_{event}_popular_hour_list.parquet"
            cand_df = pd.read_parquet(filepath)
            popular_hour_ses2candidates = get_ses2candidates(cand_df)

            # candidate #6 popular datehour
            filepath = f"{input_path}/{name}_{ix}_{event}_popular_datehour_list.parquet"
            cand_df = pd.read_parquet(filepath)
            popular_datehour_ses2candidates = get_ses2candidates(cand_df)

            del cand_df
            gc.collect()

            # concat candidates, output as df_chunk
            df_chunk = concat_candidates(
                unique_sessions=unique_sessions,
                event=event,
                covisit_ses2candidates=covisit_ses2candidates,
                fasttext_ses2candidates=fasttext_ses2candidates,
                word2vec_ses2candidates=word2vec_ses2candidates,
                matrix_fact_ses2candidates=matrix_fact_ses2candidates,
                popular_daily_ses2candidates=popular_daily_ses2candidates,
                popular_hour_ses2candidates=popular_hour_ses2candidates,
                popular_datehour_ses2candidates=popular_datehour_ses2candidates,
            )

            df_pred = pl.concat([df_pred, df_chunk])

    logging.info("convert to pandas")
    df_pred = df_pred.to_pandas()
    logging.info(df_pred.shape)
    df_pred.columns = ["session_type", "labels", "raw_session"]
    df_pred["labels"] = df_pred.labels.apply(lambda x: " ".join(map(str, x)))
    logging.info(df_pred.head())
    lucky_session = list(df_pred["raw_session"].unique())

    logging.info("start computing metrics")
    # COMPUTE METRIC
    # read ground truth
    df_truth = pd.read_parquet(gt_path)
    df_truth = df_truth[df_truth.session.isin(lucky_session)]
    measure_recall(
        df_pred=df_pred,
        df_truth=df_truth,
        Ks=[20, 40, 70, 100, 120, 130, 150, 170, 190],
    )


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test",
)
def main(mode: str):

    if mode == "training_train":
        input_path = get_processed_training_train_candidates_dir()
        gt_path = get_processed_local_validation_dir()
        gt_path = f"{gt_path}/train_labels.parquet"
        logging.info(f"read input data from: {input_path}")
        eval_candidate_list(
            name="train",
            input_path=input_path,
            gt_path=gt_path,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_candidates_dir()
        gt_path = get_processed_local_validation_dir()
        gt_path = f"{gt_path}/test_labels.parquet"
        logging.info(f"read input data from: {input_path}")
        eval_candidate_list(
            name="test",
            input_path=input_path,
            gt_path=gt_path,
        )


if __name__ == "__main__":
    main()

# strategy covisit 40 fasttext 40 word2vec 20 matrix fact 20
# [2022-12-18 20:44:48,787] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5266859884889332
# [2022-12-18 20:44:48,787] {submission_evaluation.py:84} INFO - clicks hits@20 = 231153 / gt@20 = 438882
# [2022-12-18 20:44:48,787] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5266859884889332
# [2022-12-18 20:44:55,953] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.551882775368036
# [2022-12-18 20:44:55,953] {submission_evaluation.py:84} INFO - carts hits@20 = 58994 / gt@20 = 143878
# [2022-12-18 20:44:55,953] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4100279403383422
# [2022-12-18 20:45:03,056] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.734083428731436
# [2022-12-18 20:45:03,056] {submission_evaluation.py:84} INFO - orders hits@20 = 51191 / gt@20 = 79036
# [2022-12-18 20:45:03,056] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6476921909003492
# [2022-12-18 20:45:03,056] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 20:45:03,056] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5642922954906056 (covisit)
# [2022-12-18 20:45:03,056] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 20:45:15,447] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.560690572864688
# [2022-12-18 20:45:15,448] {submission_evaluation.py:84} INFO - clicks hits@40 = 246077 / gt@40 = 438882
# [2022-12-18 20:45:15,448] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.560690572864688
# [2022-12-18 20:45:26,601] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.5872979466395353
# [2022-12-18 20:45:26,602] {submission_evaluation.py:84} INFO - carts hits@40 = 64383 / gt@40 = 143878
# [2022-12-18 20:45:26,602] {submission_evaluation.py:85} INFO - carts recall@40 = 0.4474832844493251
# [2022-12-18 20:45:37,160] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.751181009951017
# [2022-12-18 20:45:37,161] {submission_evaluation.py:84} INFO - orders hits@40 = 52899 / gt@40 = 79036
# [2022-12-18 20:45:37,161] {submission_evaluation.py:85} INFO - orders recall@40 = 0.6693025962852371
# [2022-12-18 20:45:37,161] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 20:45:37,161] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.5918956003924086 (covisit)
# [2022-12-18 20:45:37,161] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 20:45:52,269] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.5712720047757711
# [2022-12-18 20:45:52,271] {submission_evaluation.py:84} INFO - clicks hits@60 = 250721 / gt@60 = 438882
# [2022-12-18 20:45:52,272] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.5712720047757711
# [2022-12-18 20:46:10,733] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.5948863702614784
# [2022-12-18 20:46:10,734] {submission_evaluation.py:84} INFO - carts hits@60 = 65588 / gt@60 = 143878
# [2022-12-18 20:46:10,734] {submission_evaluation.py:85} INFO - carts recall@60 = 0.45585843561906614
# [2022-12-18 20:46:25,122] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.7563735832279317
# [2022-12-18 20:46:25,123] {submission_evaluation.py:84} INFO - orders hits@60 = 53322 / gt@60 = 79036
# [2022-12-18 20:46:25,123] {submission_evaluation.py:85} INFO - orders recall@60 = 0.6746545877827825
# [2022-12-18 20:46:25,124] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 20:46:25,124] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.5986774838329665 (fasttext)
# [2022-12-18 20:46:25,124] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 20:46:43,276] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.5764920867112344
# [2022-12-18 20:46:43,278] {submission_evaluation.py:84} INFO - clicks hits@70 = 253012 / gt@70 = 438882
# [2022-12-18 20:46:43,278] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.5764920867112344
# [2022-12-18 20:46:57,166] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.5983828527859771
# [2022-12-18 20:46:57,167] {submission_evaluation.py:84} INFO - carts hits@70 = 66125 / gt@70 = 143878
# [2022-12-18 20:46:57,168] {submission_evaluation.py:85} INFO - carts recall@70 = 0.45959076439761465
# [2022-12-18 20:47:12,628] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.7586864657867386
# [2022-12-18 20:47:12,629] {submission_evaluation.py:84} INFO - orders hits@70 = 53505 / gt@70 = 79036
# [2022-12-18 20:47:12,629] {submission_evaluation.py:85} INFO - orders recall@70 = 0.6769699883597348
# [2022-12-18 20:47:12,629] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 20:47:12,629] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.6017084310062487 (fasttext)
# [2022-12-18 20:47:12,629] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 20:47:31,059] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@80 = 0.5809146877748461
# [2022-12-18 20:47:31,061] {submission_evaluation.py:84} INFO - clicks hits@80 = 254953 / gt@80 = 438882
# [2022-12-18 20:47:31,061] {submission_evaluation.py:85} INFO - clicks recall@80 = 0.5809146877748461
# [2022-12-18 20:47:46,688] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@80 = 0.6011513992922577
# [2022-12-18 20:47:46,690] {submission_evaluation.py:84} INFO - carts hits@80 = 66536 / gt@80 = 143878
# [2022-12-18 20:47:46,690] {submission_evaluation.py:85} INFO - carts recall@80 = 0.4624473512281238
# [2022-12-18 20:48:03,231] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@80 = 0.7603058776713357
# [2022-12-18 20:48:03,232] {submission_evaluation.py:84} INFO - orders hits@80 = 53620 / gt@80 = 79036
# [2022-12-18 20:48:03,233] {submission_evaluation.py:85} INFO - orders recall@80 = 0.6784250215091857
# [2022-12-18 20:48:03,233] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 20:48:03,233] {submission_evaluation.py:92} INFO - Overall Recall@80 = 0.6038806870514332 (fasttext)
# [2022-12-18 20:48:03,233] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 20:48:27,484] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@100 = 0.5888940535269162
# [2022-12-18 20:48:27,487] {submission_evaluation.py:84} INFO - clicks hits@100 = 258455 / gt@100 = 438882
# [2022-12-18 20:48:27,487] {submission_evaluation.py:85} INFO - clicks recall@100 = 0.5888940535269162
# [2022-12-18 20:48:51,664] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@100 = 0.6067871692100175
# [2022-12-18 20:48:51,668] {submission_evaluation.py:84} INFO - carts hits@100 = 67429 / gt@100 = 143878
# [2022-12-18 20:48:51,668] {submission_evaluation.py:85} INFO - carts recall@100 = 0.46865399852652945
# [2022-12-18 20:49:16,975] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@100 = 0.7637101563600257
# [2022-12-18 20:49:16,977] {submission_evaluation.py:84} INFO - orders hits@100 = 53897 / gt@100 = 79036
# [2022-12-18 20:49:16,977] {submission_evaluation.py:85} INFO - orders recall@100 = 0.6819297535300369
# [2022-12-18 20:49:16,977] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 20:49:16,977] {submission_evaluation.py:92} INFO - Overall Recall@100 = 0.6086434570286725 (word2vec)
# [2022-12-18 20:49:16,977] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 20:49:49,215] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@120 = 0.5908148431696902
# [2022-12-18 20:49:49,219] {submission_evaluation.py:84} INFO - clicks hits@120 = 259298 / gt@120 = 438882
# [2022-12-18 20:49:49,219] {submission_evaluation.py:85} INFO - clicks recall@120 = 0.5908148431696902
# [2022-12-18 20:50:30,785] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@120 = 0.6079904941781838
# [2022-12-18 20:50:30,789] {submission_evaluation.py:84} INFO - carts hits@120 = 67598 / gt@120 = 143878
# [2022-12-18 20:50:30,789] {submission_evaluation.py:85} INFO - carts recall@120 = 0.46982860479016947
# [2022-12-18 20:50:59,249] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@120 = 0.7644343198969292
# [2022-12-18 20:50:59,251] {submission_evaluation.py:84} INFO - orders hits@120 = 53945 / gt@120 = 79036
# [2022-12-18 20:50:59,251] {submission_evaluation.py:85} INFO - orders recall@120 = 0.6825370717141556
# [2022-12-18 20:50:59,251] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 20:50:59,251] {submission_evaluation.py:92} INFO - Overall Recall@120 = 0.6095523087825132 (matrix factorization)
# [2022-12-18 20:50:59,251] {submission_evaluation.py:93} INFO - =============
# [2022-12-18 20:51:42,366] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@140 = 0.5908148431696902
# [2022-12-18 20:51:42,369] {submission_evaluation.py:84} INFO - clicks hits@140 = 259298 / gt@140 = 438882
# [2022-12-18 20:51:42,369] {submission_evaluation.py:85} INFO - clicks recall@140 = 0.5908148431696902
# [2022-12-18 20:52:17,362] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@140 = 0.6079904941781838
# [2022-12-18 20:52:17,363] {submission_evaluation.py:84} INFO - carts hits@140 = 67598 / gt@140 = 143878
# [2022-12-18 20:52:17,363] {submission_evaluation.py:85} INFO - carts recall@140 = 0.46982860479016947
# [2022-12-18 20:52:52,794] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@140 = 0.7644343198969292
# [2022-12-18 20:52:52,799] {submission_evaluation.py:84} INFO - orders hits@140 = 53945 / gt@140 = 79036
# [2022-12-18 20:52:52,800] {submission_evaluation.py:85} INFO - orders recall@140 = 0.6825370717141556
# [2022-12-18 20:52:52,800] {submission_evaluation.py:91} INFO - =============
# [2022-12-18 20:52:52,800] {submission_evaluation.py:92} INFO - Overall Recall@140 = 0.6095523087825132
# [2022-12-18 20:52:52,800] {submission_evaluation.py:93} INFO - =============

# strategy covisit 40 fasttext 20 word2vec 50 matrix fact 10
# [2022-12-19 09:29:49,039] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5266859884889332
# [2022-12-19 09:29:49,039] {submission_evaluation.py:84} INFO - clicks hits@20 = 231153 / gt@20 = 438882
# [2022-12-19 09:29:49,039] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5266859884889332
# [2022-12-19 09:29:54,543] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.551882775368036
# [2022-12-19 09:29:54,543] {submission_evaluation.py:84} INFO - carts hits@20 = 58994 / gt@20 = 143878
# [2022-12-19 09:29:54,543] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4100279403383422
# [2022-12-19 09:29:59,837] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.734083428731436
# [2022-12-19 09:29:59,839] {submission_evaluation.py:84} INFO - orders hits@20 = 51191 / gt@20 = 79036
# [2022-12-19 09:29:59,839] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6476921909003492
# [2022-12-19 09:29:59,839] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 09:29:59,839] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5642922954906056 (covisit)
# [2022-12-19 09:29:59,839] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 09:30:08,328] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.5606860158311345
# [2022-12-19 09:30:08,329] {submission_evaluation.py:84} INFO - clicks hits@40 = 246075 / gt@40 = 438882
# [2022-12-19 09:30:08,329] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.5606860158311345
# [2022-12-19 09:30:16,223] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.5872979466395353
# [2022-12-19 09:30:16,224] {submission_evaluation.py:84} INFO - carts hits@40 = 64383 / gt@40 = 143878
# [2022-12-19 09:30:16,224] {submission_evaluation.py:85} INFO - carts recall@40 = 0.4474832844493251
# [2022-12-19 09:30:23,389] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.7510997453711953
# [2022-12-19 09:30:23,389] {submission_evaluation.py:84} INFO - orders hits@40 = 52893 / gt@40 = 79036
# [2022-12-19 09:30:23,390] {submission_evaluation.py:85} INFO - orders recall@40 = 0.6692266815122223
# [2022-12-19 09:30:23,390] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 09:30:23,390] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.5918495958252443 (covisit)
# [2022-12-19 09:30:23,390] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 09:30:33,872] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.5714315009501415
# [2022-12-19 09:30:33,873] {submission_evaluation.py:84} INFO - clicks hits@60 = 250791 / gt@60 = 438882
# [2022-12-19 09:30:33,874] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.5714315009501415
# [2022-12-19 09:30:44,494] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.5949126588506584
# [2022-12-19 09:30:44,494] {submission_evaluation.py:84} INFO - carts hits@60 = 65600 / gt@60 = 143878
# [2022-12-19 09:30:44,495] {submission_evaluation.py:85} INFO - carts recall@60 = 0.4559418396141175
# [2022-12-19 09:30:54,887] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.7563970789259021
# [2022-12-19 09:30:54,893] {submission_evaluation.py:84} INFO - orders hits@60 = 53318 / gt@60 = 79036
# [2022-12-19 09:30:54,893] {submission_evaluation.py:85} INFO - orders recall@60 = 0.674603977934106
# [2022-12-19 09:30:54,893] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 09:30:54,894] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.598688088739713 (fasttext)
# [2022-12-19 09:30:54,894] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 09:31:09,522] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@80 = 0.5819605269753602
# [2022-12-19 09:31:09,523] {submission_evaluation.py:84} INFO - clicks hits@80 = 255412 / gt@80 = 438882
# [2022-12-19 09:31:09,523] {submission_evaluation.py:85} INFO - clicks recall@80 = 0.5819605269753602
# [2022-12-19 09:31:22,463] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@80 = 0.6030806015892475
# [2022-12-19 09:31:22,464] {submission_evaluation.py:84} INFO - carts hits@80 = 66834 / gt@80 = 143878
# [2022-12-19 09:31:22,464] {submission_evaluation.py:85} INFO - carts recall@80 = 0.464518550438566
# [2022-12-19 09:31:35,926] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@80 = 0.7611957502099408
# [2022-12-19 09:31:35,927] {submission_evaluation.py:84} INFO - orders hits@80 = 53735 / gt@80 = 79036
# [2022-12-19 09:31:35,928] {submission_evaluation.py:85} INFO - orders recall@80 = 0.6798800546586365
# [2022-12-19 09:31:35,928] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 09:31:35,928] {submission_evaluation.py:92} INFO - Overall Recall@80 = 0.6054796506242878 (word2vec)
# [2022-12-19 09:31:35,928] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 09:31:53,848] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@100 = 0.5923801841952963
# [2022-12-19 09:31:53,849] {submission_evaluation.py:84} INFO - clicks hits@100 = 259985 / gt@100 = 438882
# [2022-12-19 09:31:53,849] {submission_evaluation.py:85} INFO - clicks recall@100 = 0.5923801841952963
# [2022-12-19 09:32:10,517] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@100 = 0.609942764673979
# [2022-12-19 09:32:10,518] {submission_evaluation.py:84} INFO - carts hits@100 = 67924 / gt@100 = 143878
# [2022-12-19 09:32:10,518] {submission_evaluation.py:85} INFO - carts recall@100 = 0.47209441332239815
# [2022-12-19 09:32:25,738] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@100 = 0.7656600305630835
# [2022-12-19 09:32:25,742] {submission_evaluation.py:84} INFO - orders hits@100 = 54094 / gt@100 = 79036
# [2022-12-19 09:32:25,742] {submission_evaluation.py:85} INFO - orders recall@100 = 0.6844222885773572
# [2022-12-19 09:32:25,742] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 09:32:25,742] {submission_evaluation.py:92} INFO - Overall Recall@100 = 0.6115197155626634 (word2vec)
# [2022-12-19 09:32:25,742] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 09:32:51,590] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@120 = 0.5980012850834621
# [2022-12-19 09:32:51,593] {submission_evaluation.py:84} INFO - clicks hits@120 = 262452 / gt@120 = 438882
# [2022-12-19 09:32:51,593] {submission_evaluation.py:85} INFO - clicks recall@120 = 0.5980012850834621
# [2022-12-19 09:33:16,615] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@120 = 0.6141554845120325
# [2022-12-19 09:33:16,618] {submission_evaluation.py:84} INFO - carts hits@120 = 68535 / gt@120 = 143878
# [2022-12-19 09:33:16,618] {submission_evaluation.py:85} INFO - carts recall@120 = 0.4763410667370967
# [2022-12-19 09:33:42,337] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@120 = 0.7679166625710806
# [2022-12-19 09:33:42,339] {submission_evaluation.py:84} INFO - orders hits@120 = 54264 / gt@120 = 79036
# [2022-12-19 09:33:42,339] {submission_evaluation.py:85} INFO - orders recall@120 = 0.6865732071461106
# [2022-12-19 09:33:42,340] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 09:33:42,340] {submission_evaluation.py:92} INFO - Overall Recall@120 = 0.6146463728171416 (word2vec + matrix factorization)
# [2022-12-19 09:33:42,341] {submission_evaluation.py:93} INFO - =============
# [2022-12-19 09:34:10,835] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@140 = 0.5980012850834621
# [2022-12-19 09:34:10,836] {submission_evaluation.py:84} INFO - clicks hits@140 = 262452 / gt@140 = 438882
# [2022-12-19 09:34:10,836] {submission_evaluation.py:85} INFO - clicks recall@140 = 0.5980012850834621
# [2022-12-19 09:34:36,155] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@140 = 0.6141554845120325
# [2022-12-19 09:34:36,156] {submission_evaluation.py:84} INFO - carts hits@140 = 68535 / gt@140 = 143878
# [2022-12-19 09:34:36,156] {submission_evaluation.py:85} INFO - carts recall@140 = 0.4763410667370967
# [2022-12-19 09:35:11,366] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@140 = 0.7679166625710806
# [2022-12-19 09:35:11,367] {submission_evaluation.py:84} INFO - orders hits@140 = 54264 / gt@140 = 79036
# [2022-12-19 09:35:11,367] {submission_evaluation.py:85} INFO - orders recall@140 = 0.6865732071461106
# [2022-12-19 09:35:11,367] {submission_evaluation.py:91} INFO - =============
# [2022-12-19 09:35:11,367] {submission_evaluation.py:92} INFO - Overall Recall@140 = 0.6146463728171416
# [2022-12-19 09:35:11,367] {submission_evaluation.py:93} INFO - =============

# strategy covisit 50 word2vec 60 fasttext 20 matrix fact 10
# [2022-12-20 12:11:37,058] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5266859884889332
# [2022-12-20 12:11:37,058] {submission_evaluation.py:84} INFO - clicks hits@20 = 231153 / gt@20 = 438882
# [2022-12-20 12:11:37,059] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5266859884889332
# [2022-12-20 12:11:42,734] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.551882775368036
# [2022-12-20 12:11:42,735] {submission_evaluation.py:84} INFO - carts hits@20 = 58994 / gt@20 = 143878
# [2022-12-20 12:11:42,735] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4100279403383422
# [2022-12-20 12:11:48,423] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.734083428731436
# [2022-12-20 12:11:48,423] {submission_evaluation.py:84} INFO - orders hits@20 = 51191 / gt@20 = 79036
# [2022-12-20 12:11:48,423] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6476921909003492
# [2022-12-20 12:11:48,423] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:11:48,423] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5642922954906056 (covisit)
# [2022-12-20 12:11:48,423] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:11:57,013] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.5611622258374689
# [2022-12-20 12:11:57,014] {submission_evaluation.py:84} INFO - clicks hits@40 = 246284 / gt@40 = 438882
# [2022-12-20 12:11:57,014] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.5611622258374689
# [2022-12-20 12:12:05,329] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.5873240276994695
# [2022-12-20 12:12:05,331] {submission_evaluation.py:84} INFO - carts hits@40 = 64385 / gt@40 = 143878
# [2022-12-20 12:12:05,332] {submission_evaluation.py:85} INFO - carts recall@40 = 0.447497185115167
# [2022-12-20 12:12:13,328] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.7512630099569003
# [2022-12-20 12:12:13,329] {submission_evaluation.py:84} INFO - orders hits@40 = 52901 / gt@40 = 79036
# [2022-12-20 12:12:13,329] {submission_evaluation.py:85} INFO - orders recall@40 = 0.6693279012095754
# [2022-12-20 12:12:13,330] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:12:13,330] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.5919621188440423 (covisit)
# [2022-12-20 12:12:13,330] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:12:23,877] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@50 = 0.567826887409372
# [2022-12-20 12:12:23,881] {submission_evaluation.py:84} INFO - clicks hits@50 = 249209 / gt@50 = 438882
# [2022-12-20 12:12:23,881] {submission_evaluation.py:85} INFO - clicks recall@50 = 0.567826887409372
# [2022-12-20 12:12:33,315] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@50 = 0.5931170438373424
# [2022-12-20 12:12:33,322] {submission_evaluation.py:84} INFO - carts hits@50 = 65312 / gt@50 = 143878
# [2022-12-20 12:12:33,322] {submission_evaluation.py:85} INFO - carts recall@50 = 0.4539401437328848
# [2022-12-20 12:12:43,426] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@50 = 0.7560189114739603
# [2022-12-20 12:12:43,429] {submission_evaluation.py:84} INFO - orders hits@50 = 53373 / gt@50 = 79036
# [2022-12-20 12:12:43,430] {submission_evaluation.py:85} INFO - orders recall@50 = 0.6752998633534085
# [2022-12-20 12:12:43,430] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:12:43,430] {submission_evaluation.py:92} INFO - Overall Recall@50 = 0.5981446498728478 (covisit)
# [2022-12-20 12:12:43,430] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:12:57,315] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.5837104278598804
# [2022-12-20 12:12:57,316] {submission_evaluation.py:84} INFO - clicks hits@70 = 256180 / gt@70 = 438882
# [2022-12-20 12:12:57,316] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.5837104278598804
# [2022-12-20 12:13:09,626] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.6042211829031735
# [2022-12-20 12:13:09,629] {submission_evaluation.py:84} INFO - carts hits@70 = 66996 / gt@70 = 143878
# [2022-12-20 12:13:09,629] {submission_evaluation.py:85} INFO - carts recall@70 = 0.4656445043717594
# [2022-12-20 12:13:24,597] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.7638908238768266
# [2022-12-20 12:13:24,600] {submission_evaluation.py:84} INFO - orders hits@70 = 53999 / gt@70 = 79036
# [2022-12-20 12:13:24,600] {submission_evaluation.py:85} INFO - orders recall@70 = 0.683220304671289
# [2022-12-20 12:13:24,600] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:13:24,600] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.6079965769002893 (word2vec)
# [2022-12-20 12:13:24,600] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:13:48,507] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@90 = 0.597126334641202
# [2022-12-20 12:13:48,509] {submission_evaluation.py:84} INFO - clicks hits@90 = 262068 / gt@90 = 438882
# [2022-12-20 12:13:48,510] {submission_evaluation.py:85} INFO - clicks recall@90 = 0.597126334641202
# [2022-12-20 12:14:06,846] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@90 = 0.6141448761680399
# [2022-12-20 12:14:06,849] {submission_evaluation.py:84} INFO - carts hits@90 = 68446 / gt@90 = 143878
# [2022-12-20 12:14:06,849] {submission_evaluation.py:85} INFO - carts recall@90 = 0.47572248710713244
# [2022-12-20 12:14:23,665] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@90 = 0.7705875154482896
# [2022-12-20 12:14:23,667] {submission_evaluation.py:84} INFO - orders hits@90 = 54482 / gt@90 = 79036
# [2022-12-20 12:14:23,668] {submission_evaluation.py:85} INFO - orders recall@90 = 0.6893314438989827
# [2022-12-20 12:14:23,668] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:14:23,668] {submission_evaluation.py:92} INFO - Overall Recall@90 = 0.6160282459356495 (word2vec)
# [2022-12-20 12:14:23,668] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:14:47,875] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@110 = 0.6042307499510119
# [2022-12-20 12:14:47,876] {submission_evaluation.py:84} INFO - clicks hits@110 = 265186 / gt@110 = 438882
# [2022-12-20 12:14:47,876] {submission_evaluation.py:85} INFO - clicks recall@110 = 0.6042307499510119
# [2022-12-20 12:15:11,335] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@110 = 0.6197721581573415
# [2022-12-20 12:15:11,336] {submission_evaluation.py:84} INFO - carts hits@110 = 69292 / gt@110 = 143878
# [2022-12-20 12:15:11,336] {submission_evaluation.py:85} INFO - carts recall@110 = 0.4816024687582535
# [2022-12-20 12:15:38,827] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@110 = 0.7746235396801108
# [2022-12-20 12:15:38,828] {submission_evaluation.py:84} INFO - orders hits@110 = 54840 / gt@110 = 79036
# [2022-12-20 12:15:38,828] {submission_evaluation.py:85} INFO - orders recall@110 = 0.6938610253555342
# [2022-12-20 12:15:38,828] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:15:38,828] {submission_evaluation.py:92} INFO - Overall Recall@110 = 0.6212204308358977 (word2vec)
# [2022-12-20 12:15:38,829] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:16:12,349] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@130 = 0.6074480156397392
# [2022-12-20 12:16:12,353] {submission_evaluation.py:84} INFO - clicks hits@130 = 266598 / gt@130 = 438882
# [2022-12-20 12:16:12,359] {submission_evaluation.py:85} INFO - clicks recall@130 = 0.6074480156397392
# [2022-12-20 12:17:00,870] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@130 = 0.621996666932099
# [2022-12-20 12:17:00,878] {submission_evaluation.py:84} INFO - carts hits@130 = 69629 / gt@130 = 143878
# [2022-12-20 12:17:00,878] {submission_evaluation.py:85} INFO - carts recall@130 = 0.48394473095261265
# [2022-12-20 12:17:43,413] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@130 = 0.7760909698029244
# [2022-12-20 12:17:43,416] {submission_evaluation.py:84} INFO - orders hits@130 = 54960 / gt@130 = 79036
# [2022-12-20 12:17:43,416] {submission_evaluation.py:85} INFO - orders recall@130 = 0.6953793208158308
# [2022-12-20 12:17:43,416] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:17:43,416] {submission_evaluation.py:92} INFO - Overall Recall@130 = 0.6231558133392561 (fasttext)
# [2022-12-20 12:17:43,416] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:18:25,143] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@140 = 0.6074480156397392
# [2022-12-20 12:18:25,147] {submission_evaluation.py:84} INFO - clicks hits@140 = 266598 / gt@140 = 438882
# [2022-12-20 12:18:25,148] {submission_evaluation.py:85} INFO - clicks recall@140 = 0.6074480156397392
# [2022-12-20 12:19:04,058] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@140 = 0.621996666932099
# [2022-12-20 12:19:04,063] {submission_evaluation.py:84} INFO - carts hits@140 = 69629 / gt@140 = 143878
# [2022-12-20 12:19:04,063] {submission_evaluation.py:85} INFO - carts recall@140 = 0.48394473095261265
# [2022-12-20 12:19:45,992] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@140 = 0.7763189098577157
# [2022-12-20 12:19:45,995] {submission_evaluation.py:84} INFO - orders hits@140 = 54976 / gt@140 = 79036
# [2022-12-20 12:19:45,995] {submission_evaluation.py:85} INFO - orders recall@140 = 0.695581760210537
# [2022-12-20 12:19:45,996] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:19:45,996] {submission_evaluation.py:92} INFO - Overall Recall@140 = 0.6232772769760799 (matrix factorization)
# [2022-12-20 12:19:45,996] {submission_evaluation.py:93} INFO - =============

# strategy covisit 60 word2vec 80
# 2022-12-20 12:51:43,871] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5266859884889332
# [2022-12-20 12:51:43,871] {submission_evaluation.py:84} INFO - clicks hits@20 = 231153 / gt@20 = 438882
# [2022-12-20 12:51:43,871] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5266859884889332
# [2022-12-20 12:51:49,142] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.551882775368036
# [2022-12-20 12:51:49,143] {submission_evaluation.py:84} INFO - carts hits@20 = 58994 / gt@20 = 143878
# [2022-12-20 12:51:49,143] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4100279403383422
# [2022-12-20 12:51:54,470] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.734083428731436
# [2022-12-20 12:51:54,470] {submission_evaluation.py:84} INFO - orders hits@20 = 51191 / gt@20 = 79036
# [2022-12-20 12:51:54,470] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6476921909003492
# [2022-12-20 12:51:54,470] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:51:54,470] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5642922954906056 (covisit)
# [2022-12-20 12:51:54,470] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:52:03,067] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.561169061387799
# [2022-12-20 12:52:03,069] {submission_evaluation.py:84} INFO - clicks hits@40 = 246287 / gt@40 = 438882
# [2022-12-20 12:52:03,071] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.561169061387799
# [2022-12-20 12:52:10,070] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.5873240276994695
# [2022-12-20 12:52:10,070] {submission_evaluation.py:84} INFO - carts hits@40 = 64385 / gt@40 = 143878
# [2022-12-20 12:52:10,070] {submission_evaluation.py:85} INFO - carts recall@40 = 0.447497185115167
# [2022-12-20 12:52:16,940] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.7512630099569003
# [2022-12-20 12:52:16,940] {submission_evaluation.py:84} INFO - orders hits@40 = 52901 / gt@40 = 79036
# [2022-12-20 12:52:16,940] {submission_evaluation.py:85} INFO - orders recall@40 = 0.6693279012095754
# [2022-12-20 12:52:16,940] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:52:16,940] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.5919628023990753 (covisit)
# [2022-12-20 12:52:16,940] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:52:26,044] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@50 = 0.5677972666912746
# [2022-12-20 12:52:26,046] {submission_evaluation.py:84} INFO - clicks hits@50 = 249196 / gt@50 = 438882
# [2022-12-20 12:52:26,046] {submission_evaluation.py:85} INFO - clicks recall@50 = 0.5677972666912746
# [2022-12-20 12:52:35,484] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@50 = 0.5931580128356558
# [2022-12-20 12:52:35,485] {submission_evaluation.py:84} INFO - carts hits@50 = 65314 / gt@50 = 143878
# [2022-12-20 12:52:35,485] {submission_evaluation.py:85} INFO - carts recall@50 = 0.4539540443987267
# [2022-12-20 12:52:43,599] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@50 = 0.7559784630405648
# [2022-12-20 12:52:43,599] {submission_evaluation.py:84} INFO - orders hits@50 = 53373 / gt@50 = 79036
# [2022-12-20 12:52:43,599] {submission_evaluation.py:85} INFO - orders recall@50 = 0.6752998633534085
# [2022-12-20 12:52:43,599] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:52:43,600] {submission_evaluation.py:92} INFO - Overall Recall@50 = 0.5981458580007906 (covisit)
# [2022-12-20 12:52:43,600] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:52:53,849] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.5758563805305299
# [2022-12-20 12:52:53,850] {submission_evaluation.py:84} INFO - clicks hits@60 = 252733 / gt@60 = 438882
# [2022-12-20 12:52:53,850] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.5758563805305299
# [2022-12-20 12:53:02,596] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.5988271087297409
# [2022-12-20 12:53:02,598] {submission_evaluation.py:84} INFO - carts hits@60 = 66210 / gt@60 = 143878
# [2022-12-20 12:53:02,598] {submission_evaluation.py:85} INFO - carts recall@60 = 0.4601815426958951
# [2022-12-20 12:53:11,874] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.760567974511694
# [2022-12-20 12:53:11,875] {submission_evaluation.py:84} INFO - orders hits@60 = 53783 / gt@60 = 79036
# [2022-12-20 12:53:11,875] {submission_evaluation.py:85} INFO - orders recall@60 = 0.6804873728427552
# [2022-12-20 12:53:11,875] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:53:11,875] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.6039325245674747 (covisit)
# [2022-12-20 12:53:11,876] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:53:25,654] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@80 = 0.5907100313979612
# [2022-12-20 12:53:25,658] {submission_evaluation.py:84} INFO - clicks hits@80 = 259252 / gt@80 = 438882
# [2022-12-20 12:53:25,691] {submission_evaluation.py:85} INFO - clicks recall@80 = 0.5907100313979612
# [2022-12-20 12:53:39,520] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@80 = 0.6092836239115915
# [2022-12-20 12:53:39,521] {submission_evaluation.py:84} INFO - carts hits@80 = 67751 / gt@80 = 143878
# [2022-12-20 12:53:39,521] {submission_evaluation.py:85} INFO - carts recall@80 = 0.47089200572707435
# [2022-12-20 12:53:52,943] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@80 = 0.7680326801968594
# [2022-12-20 12:53:52,944] {submission_evaluation.py:84} INFO - orders hits@80 = 54354 / gt@80 = 79036
# [2022-12-20 12:53:52,944] {submission_evaluation.py:85} INFO - orders recall@80 = 0.687711928741333
# [2022-12-20 12:53:52,944] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:53:52,944] {submission_evaluation.py:92} INFO - Overall Recall@80 = 0.6129657621027182 (word2vec)
# [2022-12-20 12:53:52,944] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:54:12,395] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@100 = 0.6026061674892113
# [2022-12-20 12:54:12,396] {submission_evaluation.py:84} INFO - clicks hits@100 = 264473 / gt@100 = 438882
# [2022-12-20 12:54:12,396] {submission_evaluation.py:85} INFO - clicks recall@100 = 0.6026061674892113
# [2022-12-20 12:54:30,337] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@100 = 0.6183736140930833
# [2022-12-20 12:54:30,354] {submission_evaluation.py:84} INFO - carts hits@100 = 69082 / gt@100 = 143878
# [2022-12-20 12:54:30,355] {submission_evaluation.py:85} INFO - carts recall@100 = 0.48014289884485467
# [2022-12-20 12:54:49,780] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@100 = 0.7736352654381728
# [2022-12-20 12:54:49,782] {submission_evaluation.py:84} INFO - orders hits@100 = 54787 / gt@100 = 79036
# [2022-12-20 12:54:49,782] {submission_evaluation.py:85} INFO - orders recall@100 = 0.6931904448605699
# [2022-12-20 12:54:49,782] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:54:49,782] {submission_evaluation.py:92} INFO - Overall Recall@100 = 0.6202177533187194 (word2vec)
# [2022-12-20 12:54:49,782] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:55:16,977] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@120 = 0.6120164417770608
# [2022-12-20 12:55:16,979] {submission_evaluation.py:84} INFO - clicks hits@120 = 268603 / gt@120 = 438882
# [2022-12-20 12:55:16,979] {submission_evaluation.py:85} INFO - clicks recall@120 = 0.6120164417770608
# [2022-12-20 12:55:44,924] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@120 = 0.6258577513433864
# [2022-12-20 12:55:44,927] {submission_evaluation.py:84} INFO - carts hits@120 = 70203 / gt@120 = 143878
# [2022-12-20 12:55:44,928] {submission_evaluation.py:85} INFO - carts recall@120 = 0.48793422204923614
# [2022-12-20 12:56:13,307] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@120 = 0.7777356047226947
# [2022-12-20 12:56:13,308] {submission_evaluation.py:84} INFO - orders hits@120 = 55135 / gt@120 = 79036
# [2022-12-20 12:56:13,308] {submission_evaluation.py:85} INFO - orders recall@120 = 0.6975935016954299
# [2022-12-20 12:56:13,308] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:56:13,308] {submission_evaluation.py:92} INFO - Overall Recall@120 = 0.6261380118097348 (word2vec)
# [2022-12-20 12:56:13,309] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:56:42,240] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@130 = 0.6120164417770608
# [2022-12-20 12:56:42,242] {submission_evaluation.py:84} INFO - clicks hits@130 = 268603 / gt@130 = 438882
# [2022-12-20 12:56:42,242] {submission_evaluation.py:85} INFO - clicks recall@130 = 0.6120164417770608
# [2022-12-20 12:57:16,704] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@130 = 0.6258577513433864
# [2022-12-20 12:57:16,708] {submission_evaluation.py:84} INFO - carts hits@130 = 70203 / gt@130 = 143878
# [2022-12-20 12:57:16,708] {submission_evaluation.py:85} INFO - carts recall@130 = 0.48793422204923614
# [2022-12-20 12:57:41,479] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@130 = 0.7789490305885147
# [2022-12-20 12:57:41,479] {submission_evaluation.py:84} INFO - orders hits@130 = 55245 / gt@130 = 79036
# [2022-12-20 12:57:41,480] {submission_evaluation.py:85} INFO - orders recall@130 = 0.6989852725340351
# [2022-12-20 12:57:41,480] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:57:41,480] {submission_evaluation.py:92} INFO - Overall Recall@130 = 0.626973074312898 (word2vec)
# [2022-12-20 12:57:41,480] {submission_evaluation.py:93} INFO - =============
# [2022-12-20 12:58:13,640] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@140 = 0.6120164417770608
# [2022-12-20 12:58:13,643] {submission_evaluation.py:84} INFO - clicks hits@140 = 268603 / gt@140 = 438882
# [2022-12-20 12:58:13,643] {submission_evaluation.py:85} INFO - clicks recall@140 = 0.6120164417770608
# [2022-12-20 12:58:37,293] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@140 = 0.6258577513433864
# [2022-12-20 12:58:37,294] {submission_evaluation.py:84} INFO - carts hits@140 = 70203 / gt@140 = 143878
# [2022-12-20 12:58:37,294] {submission_evaluation.py:85} INFO - carts recall@140 = 0.48793422204923614
# [2022-12-20 12:59:07,742] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@140 = 0.7796154004369291
# [2022-12-20 12:59:07,744] {submission_evaluation.py:84} INFO - orders hits@140 = 55317 / gt@140 = 79036
# [2022-12-20 12:59:07,744] {submission_evaluation.py:85} INFO - orders recall@140 = 0.699896249810213
# [2022-12-20 12:59:07,745] {submission_evaluation.py:91} INFO - =============
# [2022-12-20 12:59:07,746] {submission_evaluation.py:92} INFO - Overall Recall@140 = 0.6275196606786047 (word2vec)
# [2022-12-20 12:59:07,746] {submission_evaluation.py:93} INFO - =============

# strategy covisit 60 word2vec 60  fasttext 20 matrix-fact 10
# [2022-12-21 20:41:08,416] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5266859884889332
# [2022-12-21 20:41:08,416] {submission_evaluation.py:84} INFO - clicks hits@20 = 231153 / gt@20 = 438882
# [2022-12-21 20:41:08,416] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5266859884889332
# [2022-12-21 20:41:13,902] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.551882775368036
# [2022-12-21 20:41:13,904] {submission_evaluation.py:84} INFO - carts hits@20 = 58994 / gt@20 = 143878
# [2022-12-21 20:41:13,904] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4100279403383422
# [2022-12-21 20:41:19,198] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.734083428731436
# [2022-12-21 20:41:19,198] {submission_evaluation.py:84} INFO - orders hits@20 = 51191 / gt@20 = 79036
# [2022-12-21 20:41:19,198] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6476921909003492
# [2022-12-21 20:41:19,198] {submission_evaluation.py:91} INFO - =============
# [2022-12-21 20:41:19,198] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5642922954906056 (covisit)
# [2022-12-21 20:41:19,198] {submission_evaluation.py:93} INFO - =============
# [2022-12-21 20:41:27,772] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.5611622258374689
# [2022-12-21 20:41:27,773] {submission_evaluation.py:84} INFO - clicks hits@40 = 246284 / gt@40 = 438882
# [2022-12-21 20:41:27,773] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.5611622258374689
# [2022-12-21 20:41:34,948] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.5873240276994695
# [2022-12-21 20:41:34,951] {submission_evaluation.py:84} INFO - carts hits@40 = 64385 / gt@40 = 143878
# [2022-12-21 20:41:34,951] {submission_evaluation.py:85} INFO - carts recall@40 = 0.447497185115167
# [2022-12-21 20:41:42,168] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.7512630099569003
# [2022-12-21 20:41:42,169] {submission_evaluation.py:84} INFO - orders hits@40 = 52901 / gt@40 = 79036
# [2022-12-21 20:41:42,169] {submission_evaluation.py:85} INFO - orders recall@40 = 0.6693279012095754
# [2022-12-21 20:41:42,169] {submission_evaluation.py:91} INFO - =============
# [2022-12-21 20:41:42,169] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.5919621188440423 (covisit)
# [2022-12-21 20:41:42,169] {submission_evaluation.py:93} INFO - =============
# [2022-12-21 20:41:52,389] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.5758518234969764
# [2022-12-21 20:41:52,390] {submission_evaluation.py:84} INFO - clicks hits@60 = 252731 / gt@60 = 438882
# [2022-12-21 20:41:52,390] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.5758518234969764
# [2022-12-21 20:42:02,644] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.5986531681947082
# [2022-12-21 20:42:02,646] {submission_evaluation.py:84} INFO - carts hits@60 = 66193 / gt@60 = 143878
# [2022-12-21 20:42:02,646] {submission_evaluation.py:85} INFO - carts recall@60 = 0.46006338703623906
# [2022-12-21 20:42:11,738] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.7604106301057859
# [2022-12-21 20:42:11,738] {submission_evaluation.py:84} INFO - orders hits@60 = 53775 / gt@60 = 79036
# [2022-12-21 20:42:11,739] {submission_evaluation.py:85} INFO - orders recall@60 = 0.6803861531454021
# [2022-12-21 20:42:11,739] {submission_evaluation.py:91} INFO - =============
# [2022-12-21 20:42:11,739] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.6038358903478106 (covisit)
# [2022-12-21 20:42:11,739] {submission_evaluation.py:93} INFO - =============
# [2022-12-21 20:42:24,645] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@80 = 0.5907328165657284
# [2022-12-21 20:42:24,648] {submission_evaluation.py:84} INFO - clicks hits@80 = 259262 / gt@80 = 438882
# [2022-12-21 20:42:24,648] {submission_evaluation.py:85} INFO - clicks recall@80 = 0.5907328165657284
# [2022-12-21 20:42:36,754] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@80 = 0.6092807984634321
# [2022-12-21 20:42:36,755] {submission_evaluation.py:84} INFO - carts hits@80 = 67739 / gt@80 = 143878
# [2022-12-21 20:42:36,755] {submission_evaluation.py:85} INFO - carts recall@80 = 0.470808601732023
# [2022-12-21 20:42:50,257] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@80 = 0.7680051978906447
# [2022-12-21 20:42:50,259] {submission_evaluation.py:84} INFO - orders hits@80 = 54351 / gt@80 = 79036
# [2022-12-21 20:42:50,259] {submission_evaluation.py:85} INFO - orders recall@80 = 0.6876739713548257
# [2022-12-21 20:42:50,259] {submission_evaluation.py:91} INFO - =============
# [2022-12-21 20:42:50,260] {submission_evaluation.py:92} INFO - Overall Recall@80 = 0.6129202449890752 (word2vec)
# [2022-12-21 20:42:50,260] {submission_evaluation.py:93} INFO - =============
# [2022-12-21 20:43:09,826] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@100 = 0.6025218623684726
# [2022-12-21 20:43:09,829] {submission_evaluation.py:84} INFO - clicks hits@100 = 264436 / gt@100 = 438882
# [2022-12-21 20:43:09,829] {submission_evaluation.py:85} INFO - clicks recall@100 = 0.6025218623684726
# [2022-12-21 20:43:28,104] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@100 = 0.6184968272933742
# [2022-12-21 20:43:28,105] {submission_evaluation.py:84} INFO - carts hits@100 = 69085 / gt@100 = 143878
# [2022-12-21 20:43:28,105] {submission_evaluation.py:85} INFO - carts recall@100 = 0.4801637498436175
# [2022-12-21 20:43:45,955] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@100 = 0.7733564759242559
# [2022-12-21 20:43:45,958] {submission_evaluation.py:84} INFO - orders hits@100 = 54773 / gt@100 = 79036
# [2022-12-21 20:43:45,958] {submission_evaluation.py:85} INFO - orders recall@100 = 0.693013310390202
# [2022-12-21 20:43:45,958] {submission_evaluation.py:91} INFO - =============
# [2022-12-21 20:43:45,959] {submission_evaluation.py:92} INFO - Overall Recall@100 = 0.6201092974240536 (word2vec)
# [2022-12-21 20:43:45,959] {submission_evaluation.py:93} INFO - =============
# [2022-12-21 20:44:09,939] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@120 = 0.6065753437142558
# [2022-12-21 20:44:09,941] {submission_evaluation.py:84} INFO - clicks hits@120 = 266215 / gt@120 = 438882
# [2022-12-21 20:44:09,941] {submission_evaluation.py:85} INFO - clicks recall@120 = 0.6065753437142558
# [2022-12-21 20:44:34,483] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@120 = 0.6213156532456018
# [2022-12-21 20:44:34,485] {submission_evaluation.py:84} INFO - carts hits@120 = 69543 / gt@120 = 143878
# [2022-12-21 20:44:34,485] {submission_evaluation.py:85} INFO - carts recall@120 = 0.4833470023214112
# [2022-12-21 20:44:55,518] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@120 = 0.776612960653046
# [2022-12-21 20:44:55,519] {submission_evaluation.py:84} INFO - orders hits@120 = 55071 / gt@120 = 79036
# [2022-12-21 20:44:55,519] {submission_evaluation.py:85} INFO - orders recall@120 = 0.696783744116605
# [2022-12-21 20:44:55,521] {submission_evaluation.py:91} INFO - =============
# [2022-12-21 20:44:55,522] {submission_evaluation.py:92} INFO - Overall Recall@120 = 0.6237318815378119 (word2vec)
# [2022-12-21 20:44:55,522] {submission_evaluation.py:93} INFO - =============
# [2022-12-21 20:45:24,127] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@140 = 0.6074571297068461
# [2022-12-21 20:45:24,130] {submission_evaluation.py:84} INFO - clicks hits@140 = 266602 / gt@140 = 438882
# [2022-12-21 20:45:24,130] {submission_evaluation.py:85} INFO - clicks recall@140 = 0.6074571297068461
# [2022-12-21 20:46:02,711] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@140 = 0.6219792795588096
# [2022-12-21 20:46:02,716] {submission_evaluation.py:84} INFO - carts hits@140 = 69629 / gt@140 = 143878
# [2022-12-21 20:46:02,716] {submission_evaluation.py:85} INFO - carts recall@140 = 0.48394473095261265
# [2022-12-21 20:46:40,419] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@140 = 0.7777152202179486
# [2022-12-21 20:46:40,422] {submission_evaluation.py:84} INFO - orders hits@140 = 55167 / gt@140 = 79036
# [2022-12-21 20:46:40,422] {submission_evaluation.py:85} INFO - orders recall@140 = 0.6979983804848423
# [2022-12-21 20:46:40,422] {submission_evaluation.py:91} INFO - =============
# [2022-12-21 20:46:40,422] {submission_evaluation.py:92} INFO - Overall Recall@140 = 0.6247281605473738 (fasttext)
# [2022-12-21 20:46:40,422] {submission_evaluation.py:93} INFO - =============
# [2022-12-21 20:47:18,724] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@150 = 0.6074571297068461
# [2022-12-21 20:47:18,727] {submission_evaluation.py:84} INFO - clicks hits@150 = 266602 / gt@150 = 438882
# [2022-12-21 20:47:18,727] {submission_evaluation.py:85} INFO - clicks recall@150 = 0.6074571297068461
# [2022-12-21 20:47:46,407] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@150 = 0.6219792795588096
# [2022-12-21 20:47:46,409] {submission_evaluation.py:84} INFO - carts hits@150 = 69629 / gt@150 = 143878
# [2022-12-21 20:47:46,409] {submission_evaluation.py:85} INFO - carts recall@150 = 0.48394473095261265
# [2022-12-21 20:48:16,789] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@150 = 0.7778549091453317
# [2022-12-21 20:48:16,793] {submission_evaluation.py:84} INFO - orders hits@150 = 55178 / gt@150 = 79036
# [2022-12-21 20:48:16,794] {submission_evaluation.py:85} INFO - orders recall@150 = 0.6981375575687029
# [2022-12-21 20:48:16,794] {submission_evaluation.py:91} INFO - =============
# [2022-12-21 20:48:16,794] {submission_evaluation.py:92} INFO - Overall Recall@150 = 0.6248116667976902 (matrix factorization)
# [2022-12-21 20:48:16,794] {submission_evaluation.py:93} INFO - =============

# strategy covisit 60 word2vec 60  fasttext 20 matrix-fact 10 popular daily 20
# 22-12-23 00:29:22,882] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5261987218190725
# [2022-12-23 00:29:22,882] {submission_evaluation.py:84} INFO - clicks hits@20 = 184761 / gt@20 = 351124
# [2022-12-23 00:29:22,882] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5261987218190725
# [2022-12-23 00:29:34,388] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.5535259834428123
# [2022-12-23 00:29:34,389] {submission_evaluation.py:84} INFO - carts hits@20 = 47137 / gt@20 = 114307
# [2022-12-23 00:29:34,389] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4123719457251087
# [2022-12-23 00:29:45,899] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.733916321023236
# [2022-12-23 00:29:45,899] {submission_evaluation.py:84} INFO - orders hits@20 = 40780 / gt@20 = 62805
# [2022-12-23 00:29:45,900] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6493113605604649
# [2022-12-23 00:29:45,900] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 00:29:45,900] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5659182722357188
# [2022-12-23 00:29:45,900] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 00:30:37,992] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@150 = 0.6083890591358039
# [2022-12-23 00:30:37,996] {submission_evaluation.py:84} INFO - clicks hits@150 = 213620 / gt@150 = 351124
# [2022-12-23 00:30:38,017] {submission_evaluation.py:85} INFO - clicks recall@150 = 0.6083890591358039
# [2022-12-23 00:32:28,071] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@150 = 0.622800473443139
# [2022-12-23 00:32:28,073] {submission_evaluation.py:84} INFO - carts hits@150 = 55680 / gt@150 = 114307
# [2022-12-23 00:32:28,073] {submission_evaluation.py:85} INFO - carts recall@150 = 0.48710927589736414
# [2022-12-23 00:34:06,328] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@150 = 0.774642267528888
# [2022-12-23 00:34:06,332] {submission_evaluation.py:84} INFO - orders hits@150 = 43757 / gt@150 = 62805
# [2022-12-23 00:34:06,332] {submission_evaluation.py:85} INFO - orders recall@150 = 0.6967120452193297
# [2022-12-23 00:34:06,333] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 00:34:06,333] {submission_evaluation.py:92} INFO - Overall Recall@150 = 0.6249989158143875
# [2022-12-23 00:34:06,334] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 00:36:06,987] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@170 = 0.6083890591358039
# [2022-12-23 00:36:06,992] {submission_evaluation.py:84} INFO - clicks hits@170 = 213620 / gt@170 = 351124
# [2022-12-23 00:36:06,992] {submission_evaluation.py:85} INFO - clicks recall@170 = 0.6083890591358039
# [2022-12-23 00:37:51,549] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@170 = 0.622800473443139
# [2022-12-23 00:37:51,552] {submission_evaluation.py:84} INFO - carts hits@170 = 55680 / gt@170 = 114307
# [2022-12-23 00:37:51,552] {submission_evaluation.py:85} INFO - carts recall@170 = 0.48710927589736414
# [2022-12-23 00:39:40,230] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@170 = 0.7751770455429657
# [2022-12-23 00:39:40,233] {submission_evaluation.py:84} INFO - orders hits@170 = 43802 / gt@170 = 62805
# [2022-12-23 00:39:40,234] {submission_evaluation.py:85} INFO - orders recall@170 = 0.6974285486824298
# [2022-12-23 00:39:40,234] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 00:39:40,234] {submission_evaluation.py:92} INFO - Overall Recall@170 = 0.6254288178922475
# [2022-12-23 00:39:40,234] {submission_evaluation.py:93} INFO - =============


# strategy covisit 60 popular hour 20 popular datehour 20 popular daily 20
# [2022-12-23 08:28:25,319] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.5535259834428123
# [2022-12-23 08:28:25,319] {submission_evaluation.py:84} INFO - carts hits@20 = 47137 / gt@20 = 114307
# [2022-12-23 08:28:25,319] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4123719457251087
# [2022-12-23 08:28:32,310] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.733916321023236
# [2022-12-23 08:28:32,310] {submission_evaluation.py:84} INFO - orders hits@20 = 40780 / gt@20 = 62805
# [2022-12-23 08:28:32,310] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6493113605604649
# [2022-12-23 08:28:32,310] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 08:28:32,311] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5659182722357188
# [2022-12-23 08:28:32,311] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 08:28:45,773] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@60 = 0.5609585217757829
# [2022-12-23 08:28:45,774] {submission_evaluation.py:84} INFO - clicks hits@60 = 196966 / gt@60 = 351124
# [2022-12-23 08:28:45,774] {submission_evaluation.py:85} INFO - clicks recall@60 = 0.5609585217757829
# [2022-12-23 08:28:59,915] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@60 = 0.5889894460562088
# [2022-12-23 08:28:59,916] {submission_evaluation.py:84} INFO - carts hits@60 = 51558 / gt@60 = 114307
# [2022-12-23 08:28:59,916] {submission_evaluation.py:85} INFO - carts recall@60 = 0.4510484922183243
# [2022-12-23 08:29:13,402] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@60 = 0.7555312685149795
# [2022-12-23 08:29:13,403] {submission_evaluation.py:84} INFO - orders hits@60 = 42587 / gt@60 = 62805
# [2022-12-23 08:29:13,403] {submission_evaluation.py:85} INFO - orders recall@60 = 0.6780829551787279
# [2022-12-23 08:29:13,403] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 08:29:13,403] {submission_evaluation.py:92} INFO - Overall Recall@60 = 0.5982601729503123
# [2022-12-23 08:29:13,403] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 08:29:31,541] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@80 = 0.5620265205454483
# [2022-12-23 08:29:31,541] {submission_evaluation.py:84} INFO - clicks hits@80 = 197341 / gt@80 = 351124
# [2022-12-23 08:29:31,541] {submission_evaluation.py:85} INFO - clicks recall@80 = 0.5620265205454483
# [2022-12-23 08:29:48,723] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@80 = 0.5895151971608823
# [2022-12-23 08:29:48,723] {submission_evaluation.py:84} INFO - carts hits@80 = 51627 / gt@80 = 114307
# [2022-12-23 08:29:48,723] {submission_evaluation.py:85} INFO - carts recall@80 = 0.4516521297908265
# [2022-12-23 08:30:05,445] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@80 = 0.7559660653859923
# [2022-12-23 08:30:05,445] {submission_evaluation.py:84} INFO - orders hits@80 = 42616 / gt@80 = 62805
# [2022-12-23 08:30:05,446] {submission_evaluation.py:85} INFO - orders recall@80 = 0.6785447018549479
# [2022-12-23 08:30:05,446] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 08:30:05,446] {submission_evaluation.py:92} INFO - Overall Recall@80 = 0.5988251121047615
# [2022-12-23 08:30:05,446] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 08:30:33,260] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@100 = 0.5624736560303483
# [2022-12-23 08:30:33,261] {submission_evaluation.py:84} INFO - clicks hits@100 = 197498 / gt@100 = 351124
# [2022-12-23 08:30:33,261] {submission_evaluation.py:85} INFO - clicks recall@100 = 0.5624736560303483
# [2022-12-23 08:30:55,939] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@100 = 0.5904857051577969
# [2022-12-23 08:30:55,939] {submission_evaluation.py:84} INFO - carts hits@100 = 51743 / gt@100 = 114307
# [2022-12-23 08:30:55,939] {submission_evaluation.py:85} INFO - carts recall@100 = 0.4526669407822793
# [2022-12-23 08:31:18,723] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@100 = 0.7564877864517199
# [2022-12-23 08:31:18,723] {submission_evaluation.py:84} INFO - orders hits@100 = 42643 / gt@100 = 62805
# [2022-12-23 08:31:18,724] {submission_evaluation.py:85} INFO - orders recall@100 = 0.678974603932808
# [2022-12-23 08:31:18,724] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 08:31:18,724] {submission_evaluation.py:92} INFO - Overall Recall@100 = 0.5994322101974033
# [2022-12-23 08:31:18,724] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 08:31:51,111] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@120 = 0.5624736560303483
# [2022-12-23 08:31:51,112] {submission_evaluation.py:84} INFO - clicks hits@120 = 197498 / gt@120 = 351124
# [2022-12-23 08:31:51,113] {submission_evaluation.py:85} INFO - clicks recall@120 = 0.5624736560303483
# [2022-12-23 08:32:22,088] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@120 = 0.5904857051577969
# [2022-12-23 08:32:22,089] {submission_evaluation.py:84} INFO - carts hits@120 = 51743 / gt@120 = 114307
# [2022-12-23 08:32:22,090] {submission_evaluation.py:85} INFO - carts recall@120 = 0.4526669407822793
# [2022-12-23 08:32:53,140] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@120 = 0.756841598401876
# [2022-12-23 08:32:53,142] {submission_evaluation.py:84} INFO - orders hits@120 = 42678 / gt@120 = 62805
# [2022-12-23 08:32:53,143] {submission_evaluation.py:85} INFO - orders recall@120 = 0.6795318844041079
# [2022-12-23 08:32:53,143] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 08:32:53,144] {submission_evaluation.py:92} INFO - Overall Recall@120 = 0.5997665784801833
# [2022-12-23 08:32:53,144] {submission_evaluation.py:93} INFO - =============


# strategy covisit 40 word2vec 60 fasttext 20 matrix fact 10 popular datehour 20 recall @150
# [2022-12-23 11:57:03,931] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5261987218190725
# [2022-12-23 11:57:03,932] {submission_evaluation.py:84} INFO - clicks hits@20 = 184761 / gt@20 = 351124
# [2022-12-23 11:57:03,932] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5261987218190725
# [2022-12-23 11:57:08,646] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.5535259834428123
# [2022-12-23 11:57:08,647] {submission_evaluation.py:84} INFO - carts hits@20 = 47137 / gt@20 = 114307
# [2022-12-23 11:57:08,648] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4123719457251087
# [2022-12-23 11:57:13,119] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.733916321023236
# [2022-12-23 11:57:13,119] {submission_evaluation.py:84} INFO - orders hits@20 = 40780 / gt@20 = 62805
# [2022-12-23 11:57:13,119] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6493113605604649
# [2022-12-23 11:57:13,119] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 11:57:13,119] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5659182722357188
# [2022-12-23 11:57:13,119] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 11:57:20,092] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.5601867146649047
# [2022-12-23 11:57:20,092] {submission_evaluation.py:84} INFO - clicks hits@40 = 196695 / gt@40 = 351124
# [2022-12-23 11:57:20,092] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.5601867146649047
# [2022-12-23 11:57:26,463] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.5873421403283092
# [2022-12-23 11:57:26,466] {submission_evaluation.py:84} INFO - carts hits@40 = 51347 / gt@40 = 114307
# [2022-12-23 11:57:26,466] {submission_evaluation.py:85} INFO - carts recall@40 = 0.44920258601835406
# [2022-12-23 11:57:32,186] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.7500622254328378
# [2022-12-23 11:57:32,187] {submission_evaluation.py:84} INFO - orders hits@40 = 42095 / gt@40 = 62805
# [2022-12-23 11:57:32,188] {submission_evaluation.py:85} INFO - orders recall@40 = 0.670249183982167
# [2022-12-23 11:57:32,188] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 11:57:32,188] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.5929289576612968
# [2022-12-23 11:57:32,188] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 11:57:41,957] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.5804843872819858
# [2022-12-23 11:57:41,958] {submission_evaluation.py:84} INFO - clicks hits@70 = 203822 / gt@70 = 351124
# [2022-12-23 11:57:41,958] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.5804843872819858
# [2022-12-23 11:57:51,147] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.6019571277830458
# [2022-12-23 11:57:51,150] {submission_evaluation.py:84} INFO - carts hits@70 = 53207 / gt@70 = 114307
# [2022-12-23 11:57:51,150] {submission_evaluation.py:85} INFO - carts recall@70 = 0.46547455536406346
# [2022-12-23 11:57:59,778] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.7585851287924682
# [2022-12-23 11:57:59,782] {submission_evaluation.py:84} INFO - orders hits@70 = 42682 / gt@70 = 62805
# [2022-12-23 11:57:59,782] {submission_evaluation.py:85} INFO - orders recall@70 = 0.679595573600828
# [2022-12-23 11:57:59,782] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 11:57:59,783] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.6054481494979145
# [2022-12-23 11:57:59,783] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 11:58:13,336] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@100 = 0.5959604014536175
# [2022-12-23 11:58:13,338] {submission_evaluation.py:84} INFO - clicks hits@100 = 209256 / gt@100 = 351124
# [2022-12-23 11:58:13,338] {submission_evaluation.py:85} INFO - clicks recall@100 = 0.5959604014536175
# [2022-12-23 11:58:27,582] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@100 = 0.6135225236560703
# [2022-12-23 11:58:27,583] {submission_evaluation.py:84} INFO - carts hits@100 = 54611 / gt@100 = 114307
# [2022-12-23 11:58:27,584] {submission_evaluation.py:85} INFO - carts recall@100 = 0.4777572677088892
# [2022-12-23 11:58:41,312] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@100 = 0.7658039852387953
# [2022-12-23 11:58:41,313] {submission_evaluation.py:84} INFO - orders hits@100 = 43114 / gt@100 = 62805
# [2022-12-23 11:58:41,313] {submission_evaluation.py:85} INFO - orders recall@100 = 0.6864740068465887
# [2022-12-23 11:58:41,313] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 11:58:41,313] {submission_evaluation.py:92} INFO - Overall Recall@100 = 0.6148076245659817
# [2022-12-23 11:58:41,313] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 11:59:00,922] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@120 = 0.5998991809161436
# [2022-12-23 11:59:00,925] {submission_evaluation.py:84} INFO - clicks hits@120 = 210639 / gt@120 = 351124
# [2022-12-23 11:59:00,925] {submission_evaluation.py:85} INFO - clicks recall@120 = 0.5998991809161436
# [2022-12-23 11:59:19,363] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@120 = 0.6158697960341387
# [2022-12-23 11:59:19,367] {submission_evaluation.py:84} INFO - carts hits@120 = 54904 / gt@120 = 114307
# [2022-12-23 11:59:19,367] {submission_evaluation.py:85} INFO - carts recall@120 = 0.48032054029936927
# [2022-12-23 11:59:37,738] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@120 = 0.7671818908197582
# [2022-12-23 11:59:37,740] {submission_evaluation.py:84} INFO - orders hits@120 = 43208 / gt@120 = 62805
# [2022-12-23 11:59:37,740] {submission_evaluation.py:85} INFO - orders recall@120 = 0.6879707029695088
# [2022-12-23 11:59:37,740] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 11:59:37,740] {submission_evaluation.py:92} INFO - Overall Recall@120 = 0.6168685019631304
# [2022-12-23 11:59:37,741] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 12:00:01,744] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@130 = 0.6007792119023478
# [2022-12-23 12:00:01,748] {submission_evaluation.py:84} INFO - clicks hits@130 = 210948 / gt@130 = 351124
# [2022-12-23 12:00:01,749] {submission_evaluation.py:85} INFO - clicks recall@130 = 0.6007792119023478
# [2022-12-23 12:00:26,138] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@130 = 0.6164793283209518
# [2022-12-23 12:00:26,139] {submission_evaluation.py:84} INFO - carts hits@130 = 54977 / gt@130 = 114307
# [2022-12-23 12:00:26,139] {submission_evaluation.py:85} INFO - carts recall@130 = 0.48095917135433525
# [2022-12-23 12:00:48,960] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@130 = 0.7676214519655195
# [2022-12-23 12:00:48,961] {submission_evaluation.py:84} INFO - orders hits@130 = 43234 / gt@130 = 62805
# [2022-12-23 12:00:48,961] {submission_evaluation.py:85} INFO - orders recall@130 = 0.6883846827481889
# [2022-12-23 12:00:48,961] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 12:00:48,961] {submission_evaluation.py:92} INFO - Overall Recall@130 = 0.6173964822454486
# [2022-12-23 12:00:48,961] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 12:01:18,308] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@150 = 0.6025848418222622
# [2022-12-23 12:01:18,312] {submission_evaluation.py:84} INFO - clicks hits@150 = 211582 / gt@150 = 351124
# [2022-12-23 12:01:18,312] {submission_evaluation.py:85} INFO - clicks recall@150 = 0.6025848418222622
# [2022-12-23 12:01:50,685] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@150 = 0.617480167743527
# [2022-12-23 12:01:50,687] {submission_evaluation.py:84} INFO - carts hits@150 = 55103 / gt@150 = 114307
# [2022-12-23 12:01:50,687] {submission_evaluation.py:85} INFO - carts recall@150 = 0.48206146605194783
# [2022-12-23 12:02:20,349] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@150 = 0.7678128677260597
# [2022-12-23 12:02:20,350] {submission_evaluation.py:84} INFO - orders hits@150 = 43247 / gt@150 = 62805
# [2022-12-23 12:02:20,350] {submission_evaluation.py:85} INFO - orders recall@150 = 0.6885916726375289
# [2022-12-23 12:02:20,351] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 12:02:20,351] {submission_evaluation.py:92} INFO - Overall Recall@150 = 0.6180319275803279
# [2022-12-23 12:02:20,351] {submission_evaluation.py:93} INFO - =============

# strategy covisit 50 word2vec 50 fasttext 20 matrix fact 10 popular datehour 20 hour 20 daily 20 recall @190
# [2022-12-23 13:33:32,321] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@20 = 0.5261987218190725
# [2022-12-23 13:33:32,321] {submission_evaluation.py:84} INFO - clicks hits@20 = 184761 / gt@20 = 351124
# [2022-12-23 13:33:32,321] {submission_evaluation.py:85} INFO - clicks recall@20 = 0.5261987218190725
# [2022-12-23 13:33:37,314] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@20 = 0.5535259834428123
# [2022-12-23 13:33:37,314] {submission_evaluation.py:84} INFO - carts hits@20 = 47137 / gt@20 = 114307
# [2022-12-23 13:33:37,314] {submission_evaluation.py:85} INFO - carts recall@20 = 0.4123719457251087
# [2022-12-23 13:33:42,234] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@20 = 0.733916321023236
# [2022-12-23 13:33:42,234] {submission_evaluation.py:84} INFO - orders hits@20 = 40780 / gt@20 = 62805
# [2022-12-23 13:33:42,234] {submission_evaluation.py:85} INFO - orders recall@20 = 0.6493113605604649
# [2022-12-23 13:33:42,234] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 13:33:42,234] {submission_evaluation.py:92} INFO - Overall Recall@20 = 0.5659182722357188
# [2022-12-23 13:33:42,234] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 13:33:49,283] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@40 = 0.5601867146649047
# [2022-12-23 13:33:49,284] {submission_evaluation.py:84} INFO - clicks hits@40 = 196695 / gt@40 = 351124
# [2022-12-23 13:33:49,284] {submission_evaluation.py:85} INFO - clicks recall@40 = 0.5601867146649047
# [2022-12-23 13:33:56,174] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@40 = 0.5873421403283092
# [2022-12-23 13:33:56,175] {submission_evaluation.py:84} INFO - carts hits@40 = 51347 / gt@40 = 114307
# [2022-12-23 13:33:56,176] {submission_evaluation.py:85} INFO - carts recall@40 = 0.44920258601835406
# [2022-12-23 13:34:02,440] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@40 = 0.7500622254328378
# [2022-12-23 13:34:02,441] {submission_evaluation.py:84} INFO - orders hits@40 = 42095 / gt@40 = 62805
# [2022-12-23 13:34:02,441] {submission_evaluation.py:85} INFO - orders recall@40 = 0.670249183982167
# [2022-12-23 13:34:02,441] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 13:34:02,441] {submission_evaluation.py:92} INFO - Overall Recall@40 = 0.5929289576612968
# [2022-12-23 13:34:02,441] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 13:34:13,049] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@70 = 0.5804843872819858
# [2022-12-23 13:34:13,051] {submission_evaluation.py:84} INFO - clicks hits@70 = 203822 / gt@70 = 351124
# [2022-12-23 13:34:13,051] {submission_evaluation.py:85} INFO - clicks recall@70 = 0.5804843872819858
# [2022-12-23 13:34:23,425] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@70 = 0.6019571277830458
# [2022-12-23 13:34:23,427] {submission_evaluation.py:84} INFO - carts hits@70 = 53207 / gt@70 = 114307
# [2022-12-23 13:34:23,427] {submission_evaluation.py:85} INFO - carts recall@70 = 0.46547455536406346
# [2022-12-23 13:34:33,225] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@70 = 0.7585851287924682
# [2022-12-23 13:34:33,229] {submission_evaluation.py:84} INFO - orders hits@70 = 42682 / gt@70 = 62805
# [2022-12-23 13:34:33,229] {submission_evaluation.py:85} INFO - orders recall@70 = 0.679595573600828
# [2022-12-23 13:34:33,229] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 13:34:33,229] {submission_evaluation.py:92} INFO - Overall Recall@70 = 0.6054481494979145
# [2022-12-23 13:34:33,229] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 13:34:47,665] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@100 = 0.5959604014536175
# [2022-12-23 13:34:47,666] {submission_evaluation.py:84} INFO - clicks hits@100 = 209256 / gt@100 = 351124
# [2022-12-23 13:34:47,666] {submission_evaluation.py:85} INFO - clicks recall@100 = 0.5959604014536175
# [2022-12-23 13:35:01,856] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@100 = 0.6135225236560703
# [2022-12-23 13:35:01,857] {submission_evaluation.py:84} INFO - carts hits@100 = 54611 / gt@100 = 114307
# [2022-12-23 13:35:01,857] {submission_evaluation.py:85} INFO - carts recall@100 = 0.4777572677088892
# [2022-12-23 13:35:18,116] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@100 = 0.7658039852387953
# [2022-12-23 13:35:18,116] {submission_evaluation.py:84} INFO - orders hits@100 = 43114 / gt@100 = 62805
# [2022-12-23 13:35:18,116] {submission_evaluation.py:85} INFO - orders recall@100 = 0.6864740068465887
# [2022-12-23 13:35:18,116] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 13:35:18,116] {submission_evaluation.py:92} INFO - Overall Recall@100 = 0.6148076245659817
# [2022-12-23 13:35:18,116] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 13:35:38,318] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@120 = 0.5998991809161436
# [2022-12-23 13:35:38,320] {submission_evaluation.py:84} INFO - clicks hits@120 = 210639 / gt@120 = 351124
# [2022-12-23 13:35:38,320] {submission_evaluation.py:85} INFO - clicks recall@120 = 0.5998991809161436
# [2022-12-23 13:35:58,295] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@120 = 0.6158697960341387
# [2022-12-23 13:35:58,299] {submission_evaluation.py:84} INFO - carts hits@120 = 54904 / gt@120 = 114307
# [2022-12-23 13:35:58,299] {submission_evaluation.py:85} INFO - carts recall@120 = 0.48032054029936927
# [2022-12-23 13:36:17,070] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@120 = 0.7671818908197582
# [2022-12-23 13:36:17,072] {submission_evaluation.py:84} INFO - orders hits@120 = 43208 / gt@120 = 62805
# [2022-12-23 13:36:17,072] {submission_evaluation.py:85} INFO - orders recall@120 = 0.6879707029695088
# [2022-12-23 13:36:17,073] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 13:36:17,073] {submission_evaluation.py:92} INFO - Overall Recall@120 = 0.6168685019631304
# [2022-12-23 13:36:17,073] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 13:36:40,093] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@130 = 0.6007792119023478
# [2022-12-23 13:36:40,096] {submission_evaluation.py:84} INFO - clicks hits@130 = 210948 / gt@130 = 351124
# [2022-12-23 13:36:40,096] {submission_evaluation.py:85} INFO - clicks recall@130 = 0.6007792119023478
# [2022-12-23 13:37:02,833] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@130 = 0.6164793283209518
# [2022-12-23 13:37:02,834] {submission_evaluation.py:84} INFO - carts hits@130 = 54977 / gt@130 = 114307
# [2022-12-23 13:37:02,835] {submission_evaluation.py:85} INFO - carts recall@130 = 0.48095917135433525
# [2022-12-23 13:37:25,084] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@130 = 0.7676214519655195
# [2022-12-23 13:37:25,086] {submission_evaluation.py:84} INFO - orders hits@130 = 43234 / gt@130 = 62805
# [2022-12-23 13:37:25,086] {submission_evaluation.py:85} INFO - orders recall@130 = 0.6883846827481889
# [2022-12-23 13:37:25,087] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 13:37:25,087] {submission_evaluation.py:92} INFO - Overall Recall@130 = 0.6173964822454486
# [2022-12-23 13:37:25,087] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 13:37:53,647] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@150 = 0.6026190177828915
# [2022-12-23 13:37:53,649] {submission_evaluation.py:84} INFO - clicks hits@150 = 211594 / gt@150 = 351124
# [2022-12-23 13:37:53,649] {submission_evaluation.py:85} INFO - clicks recall@150 = 0.6026190177828915
# [2022-12-23 13:38:25,274] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@150 = 0.617546613380335
# [2022-12-23 13:38:25,276] {submission_evaluation.py:84} INFO - carts hits@150 = 55110 / gt@150 = 114307
# [2022-12-23 13:38:25,276] {submission_evaluation.py:85} INFO - carts recall@150 = 0.4821227046462596
# [2022-12-23 13:38:54,541] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@150 = 0.7679227596518952
# [2022-12-23 13:38:54,542] {submission_evaluation.py:84} INFO - orders hits@150 = 43253 / gt@150 = 62805
# [2022-12-23 13:38:54,542] {submission_evaluation.py:85} INFO - orders recall@150 = 0.6886872064326088
# [2022-12-23 13:38:54,543] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 13:38:54,543] {submission_evaluation.py:92} INFO - Overall Recall@150 = 0.6181110370317323
# [2022-12-23 13:38:54,543] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 13:39:26,920] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@170 = 0.603185769129994
# [2022-12-23 13:39:26,923] {submission_evaluation.py:84} INFO - clicks hits@170 = 211793 / gt@170 = 351124
# [2022-12-23 13:39:26,923] {submission_evaluation.py:85} INFO - clicks recall@170 = 0.603185769129994
# [2022-12-23 13:40:08,179] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@170 = 0.618695165756007
# [2022-12-23 13:40:08,182] {submission_evaluation.py:84} INFO - carts hits@170 = 55260 / gt@170 = 114307
# [2022-12-23 13:40:08,182] {submission_evaluation.py:85} INFO - carts recall@170 = 0.48343496023865556
# [2022-12-23 13:40:45,814] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@170 = 0.7685629935818005
# [2022-12-23 13:40:45,818] {submission_evaluation.py:84} INFO - orders hits@170 = 43294 / gt@170 = 62805
# [2022-12-23 13:40:45,818] {submission_evaluation.py:85} INFO - orders recall@170 = 0.6893400206989889
# [2022-12-23 13:40:45,819] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 13:40:45,819] {submission_evaluation.py:92} INFO - Overall Recall@170 = 0.6189530774039894
# [2022-12-23 13:40:45,819] {submission_evaluation.py:93} INFO - =============
# [2022-12-23 13:41:42,157] {submission_evaluation.py:83} INFO - clicks mean_recall_per_sample@190 = 0.6035901846641073
# [2022-12-23 13:41:42,162] {submission_evaluation.py:84} INFO - clicks hits@190 = 211935 / gt@190 = 351124
# [2022-12-23 13:41:42,162] {submission_evaluation.py:85} INFO - clicks recall@190 = 0.6035901846641073
# [2022-12-23 13:42:23,694] {submission_evaluation.py:83} INFO - carts mean_recall_per_sample@190 = 0.6196567896022462
# [2022-12-23 13:42:23,698] {submission_evaluation.py:84} INFO - carts hits@190 = 55374 / gt@190 = 114307
# [2022-12-23 13:42:23,698] {submission_evaluation.py:85} INFO - carts recall@190 = 0.48443227448887644
# [2022-12-23 13:43:12,389] {submission_evaluation.py:83} INFO - orders mean_recall_per_sample@190 = 0.7694163681490847
# [2022-12-23 13:43:12,393] {submission_evaluation.py:84} INFO - orders hits@190 = 43355 / gt@190 = 62805
# [2022-12-23 13:43:12,393] {submission_evaluation.py:85} INFO - orders recall@190 = 0.690311280948969
# [2022-12-23 13:43:12,393] {submission_evaluation.py:91} INFO - =============
# [2022-12-23 13:43:12,393] {submission_evaluation.py:92} INFO - Overall Recall@190 = 0.6198754693824551
# [2022-12-23 13:43:12,393] {submission_evaluation.py:93} INFO - =============
