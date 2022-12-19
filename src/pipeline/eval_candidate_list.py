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
):
    labels = []
    sessions = []
    for session in unique_sessions:
        # get candidates for specific session
        # covisitation candidates
        cands = list(covisit_ses2candidates[session])
        # fasttext candidates
        fasttext_cands = list(fasttext_ses2candidates[session])
        cands.extend(fasttext_cands)
        # word2vec candidates
        word2vec_cands = list(word2vec_ses2candidates[session])
        cands.extend(word2vec_cands)
        # matrix fact candidates
        matrix_fact_cands = list(matrix_fact_ses2candidates[session])
        cands.extend(matrix_fact_cands)
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

            # concat candidates, output as df_chunk
            df_chunk = concat_candidates(
                unique_sessions=unique_sessions,
                event=event,
                covisit_ses2candidates=covisit_ses2candidates,
                fasttext_ses2candidates=fasttext_ses2candidates,
                word2vec_ses2candidates=word2vec_ses2candidates,
                matrix_fact_ses2candidates=matrix_fact_ses2candidates,
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
        Ks=[20, 40, 60, 80, 100, 110, 120, 130, 140, 150],
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
