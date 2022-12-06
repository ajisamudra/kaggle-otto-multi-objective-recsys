import pandas as pd
import numpy as np
import pandas as pd
import gc
import numpy as np
from src.utils.constants import (
    get_processed_local_validation_dir,
    get_data_output_dir,
)
from src.utils.logger import get_logger
from tqdm import tqdm

logging = get_logger()


def measure_recall(df_pred: pd.DataFrame, df_truth: pd.DataFrame, Ks: list = [20]):
    """
    df_pred is in submission format
    session_type | labels
    123_clicks | AID1 AID2 AID3
    123_carts | AID1 AID2
    123_orders | AID1

    df_truth is in following format
    session | type | ground_truth
    123 | clicks | AID1 AID4
    123 | carts | AID1
    123 | orders | AID1

    Ks list of recall@K want to measure

    """
    # competition eval metrics
    dict_metrics = {}

    logging.info("create prediction session column")
    df_pred["session"] = df_pred.session_type.apply(lambda x: int(x.split("_")[0]))
    logging.info("create prediction type column")
    df_pred["type"] = df_pred.session_type.apply(lambda x: x.split("_")[1])

    for K in Ks:
        score = 0
        weights = {"clicks": 0.10, "carts": 0.30, "orders": 0.60}
        for t in ["clicks", "carts", "orders"]:
            # logging.info(f"filter submission event {t}")
            sub_session = df_pred.loc[df_pred["type"] == t]["session"].values
            sub_labels = (
                df_pred.loc[df_pred["type"] == t]["labels"]
                .apply(lambda x: [int(i) for i in x.split(" ")[:K]])
                .values
            )
            # logging.info(f"filter label event {t}")
            label_session = df_truth.loc[df_truth["type"] == t]["session"].values
            label_truth = df_truth.loc[df_truth["type"] == t]["ground_truth"].values

            sub_dict = dict(zip(sub_session, sub_labels))
            tmp_labels_dict = dict(zip(label_session, label_truth))

            label_session = list(label_session)
            recall_scores, hit_scores, gt_counts = [], [], []
            for session_id in label_session:
                targets = set(tmp_labels_dict[session_id])
                preds = set(sub_dict[session_id])

                # calc
                hit = len((targets & preds))
                len_truth = min(len(targets), 20)
                recall_score = hit / len_truth

                recall_scores.append(recall_score)
                hit_scores.append(hit)
                gt_counts.append(len_truth)

            recall_scores = np.array(recall_scores)
            hit_scores = np.array(hit_scores)
            gt_counts = np.array(gt_counts)

            n_hits = np.sum(hit_scores)
            n_gt = np.sum(gt_counts)
            recall = np.sum(hit_scores) / np.sum(gt_counts)
            mean_recall_per_sample = np.mean(recall_scores)
            score += weights[t] * recall
            logging.info(f"{t} mean_recall_per_sample@{K} = {mean_recall_per_sample}")
            logging.info(f"{t} hits@{K} = {n_hits} / gt@{K} = {n_gt}")
            logging.info(f"{t} recall@{K} = {recall}")

            dict_metrics[f"{t}_hits@{K}"] = str(n_hits)
            dict_metrics[f"{t}_gt@{K}"] = str(n_gt)
            dict_metrics[f"{t}_recall@{K}"] = str(recall)

        logging.info("=============")
        logging.info(f"Overall Recall@{K} = {score}")
        logging.info("=============")
        dict_metrics[f"overall_recall@{K}"] = str(score)

    return dict_metrics


if __name__ == "__main__":

    DATA_DIR = get_processed_local_validation_dir()
    OUTPUT_DIR = get_data_output_dir()
    filepath = OUTPUT_DIR / "covisit_retrieval"
    logging.info("read prediction df")
    pred_df = pd.read_csv(filepath / "validation_preds.csv")
    logging.info("read test label df")
    test_labels = pd.read_parquet(DATA_DIR / "test_labels.parquet")
    recall_20 = measure_recall(
        df_pred=pred_df, df_truth=test_labels, Ks=[20, 30, 40, 50]
    )
