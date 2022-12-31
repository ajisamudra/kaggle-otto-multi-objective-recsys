import click
import pandas as pd
from tqdm import tqdm
import gc
from pathlib import Path
import random
from src.utils.constants import (
    CFG,
    get_processed_training_train_candidates_dir,
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
    get_processed_local_validation_dir,
    get_processed_full_data_dir,
)
from src.utils.logger import get_logger

logging = get_logger()


# read chunk parquet
# for each chunk -> pivot N candidates into rows
# so for each session, there's N-1 additional rows


def get_ses2candidates(df: pd.DataFrame) -> dict:
    """
    session | labels
    123_clicks | [aid1, aid2]
    123_orders | [aid1, aid2, aid3]
    """

    df["session"] = df["session"].apply(lambda x: int(x.split("_")[0]))
    ses2candidates = dict(zip(df["session"], df["labels"]))

    return ses2candidates


def pivot_candidates_list_to_rows(
    unique_sessions: list,
    covisit_ses2candidates: dict,
    fasttext_ses2candidates: dict,
    word2vec_ses2candidates: dict,
    matrix_fact_ses2candidates: dict,
    popular_hour_ses2candidates: dict,
    is_train: bool,
    include_all_gt: bool,
    drop_zero_positive_sample: bool,
    ses2truth: dict = {},
    ratio_negative_sample: int = 1,
):
    """
    cand_df
    session | labels
    123_clicks | [aid1, aid2, aid3]
    123_carts | [aid1, aid2, aid3]
    123_orders | [aid1, aid2, aid3]
    """

    sessions = []
    candidates = []
    labels = []
    candidates_covisit = []
    candidates_word2vec = []
    candidates_fasttext = []
    candidates_popular_hour = []
    candidates_matrix_fact = []

    logging.info(
        f"pivot in train_mode: {is_train} | include_all_gt: {include_all_gt} | drop_zero_positive_sample: {drop_zero_positive_sample}"
    )
    for session in tqdm(unique_sessions):
        truths = set()
        if is_train:
            # get truths for specific session
            truths = set(ses2truth.get(session, []))

            if drop_zero_positive_sample:
                # if there's no truth for specific event & session
                # drop from training data
                if len(truths) == 0:
                    continue

        # get candidates for specific session
        # covisitation candidates
        covisit_cands = list(covisit_ses2candidates[session])
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
        # popular hour candidates
        popular_hour_cands = list(popular_hour_ses2candidates[session])
        cands.extend(popular_hour_cands)
        # drop duplicate
        cands = set(cands)

        if include_all_gt:
            # add all truths if it's training data
            cands = cands | truths

        # check whether it's in truths
        label_ls = [1 if c in truths else 0 for c in cands]
        covisit_ls = [1 if c in covisit_cands else 0 for c in cands]
        fasttext_ls = [1 if c in fasttext_cands else 0 for c in cands]
        word2vec_ls = [1 if c in word2vec_cands else 0 for c in cands]
        matrix_fact_ls = [1 if c in matrix_fact_cands else 0 for c in cands]
        popular_hour_ls = [1 if c in popular_hour_cands else 0 for c in cands]
        session_ls = [session for i in range(len(cands))]
        cands_ls = cands

        if ratio_negative_sample > 1:
            # ratio * num positive samples
            n_negative = ratio_negative_sample * sum(label_ls)
            positive_aids = [aid for i, aid in enumerate(cands_ls) if label_ls[i] == 1]
            if n_negative < len(cands) - len(positive_aids):
                negative_aids = [
                    aid for i, aid in enumerate(cands_ls) if label_ls[i] == 0
                ]
                sampled_negative_aids = random.sample(negative_aids, n_negative)
                positive_labels = [1 for i in positive_aids]
                negative_labels = [0 for i in sampled_negative_aids]

                # add positive + sampled negative sample
                positive_aids.extend(sampled_negative_aids)
                positive_labels.extend(negative_labels)

                # create final list
                cands_ls = positive_aids
                label_ls = positive_labels
                session_ls = [session for i in range(len(cands_ls))]

            # else: # will take all negative samples

        sessions.extend(session_ls)
        candidates.extend(cands_ls)
        labels.extend(label_ls)

        candidates_covisit.extend(covisit_ls)
        candidates_word2vec.extend(fasttext_ls)
        candidates_fasttext.extend(word2vec_ls)
        candidates_popular_hour.extend(matrix_fact_ls)
        candidates_matrix_fact.extend(popular_hour_ls)

    # save as df
    data = {
        "session": sessions,
        "candidate_aid": candidates,
        "retrieval_covisit": candidates_covisit,
        "retrieval_word2vec": candidates_word2vec,
        "retrieval_fasttext": candidates_fasttext,
        "retrieval_popular_hour": candidates_popular_hour,
        "retrieval_matrix_fact": candidates_matrix_fact,
        "label": labels,
    }

    return pd.DataFrame(data)


def pivot_candidates(
    name: str,
    is_train: bool,
    include_all_gt: bool,
    drop_zero_positive_sample: bool,
    input_path: Path,
    output_path: Path,
    df_truth: pd.DataFrame = pd.DataFrame(),
    ratio_negative_sample: int = 1,
):
    """
    df_truth
    session | type | ground_truth
    123 | clicks | [AID1 AID4]
    123 | carts | [AID1]
    123 | orders | [AID1]
    """

    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(n)):
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

            # candidate #5 popular hour
            filepath = f"{input_path}/{name}_{ix}_{event}_popular_hour_list.parquet"
            cand_df = pd.read_parquet(filepath)
            popular_hour_ses2candidates = get_ses2candidates(cand_df)

            ses2truth = {}
            if is_train:
                # logging.info("create ses2truth")
                label_session = df_truth.loc[df_truth["type"] == event][
                    "session"
                ].values
                label_truth = df_truth.loc[df_truth["type"] == event][
                    "ground_truth"
                ].values
                ses2truth = dict(zip(label_session, label_truth))

            # logging.info(f"start pivoting candidate")
            df_output = pivot_candidates_list_to_rows(
                unique_sessions=unique_sessions,
                covisit_ses2candidates=covisit_ses2candidates,
                word2vec_ses2candidates=word2vec_ses2candidates,
                fasttext_ses2candidates=fasttext_ses2candidates,
                matrix_fact_ses2candidates=matrix_fact_ses2candidates,
                popular_hour_ses2candidates=popular_hour_ses2candidates,
                is_train=is_train,
                include_all_gt=include_all_gt,
                ratio_negative_sample=ratio_negative_sample,
                drop_zero_positive_sample=drop_zero_positive_sample,
                ses2truth=ses2truth,
            )

            filepath = output_path / f"{name}_{ix}_{event}_rows.parquet"
            logging.info(f"save chunk {ix}_{event} to: {filepath}")
            df_output.to_parquet(f"{filepath}")
            logging.info(f"output df shape {df_output.shape}")

            del df_output
            gc.collect()


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    if mode == "training_train":
        input_path = get_processed_training_train_candidates_dir()
        output_path = get_processed_training_train_candidates_dir()
        logging.info(f"will read input from & save output to: {input_path}")
        truth = get_processed_local_validation_dir()
        truth = f"{truth}/train_labels.parquet"
        logging.info(f"read ses2truth from: {truth}")
        df_truth = pd.read_parquet(truth)
        pivot_candidates(
            name="train",
            is_train=True,
            include_all_gt=False,
            drop_zero_positive_sample=True,
            input_path=input_path,
            output_path=output_path,
            df_truth=df_truth,
            ratio_negative_sample=10,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_candidates_dir()
        output_path = get_processed_training_test_candidates_dir()
        logging.info(f"will read input from & save output to: {input_path}")
        truth = get_processed_local_validation_dir()
        truth = f"{truth}/test_labels.parquet"
        logging.info(f"read ses2truth from: {truth}")
        df_truth = pd.read_parquet(truth)
        logging.info(f"df_truth shape: {df_truth.shape}")
        pivot_candidates(
            name="test",
            is_train=True,
            include_all_gt=True,
            drop_zero_positive_sample=True,
            input_path=input_path,
            output_path=output_path,
            df_truth=df_truth,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_candidates_dir()
        output_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"will read input from & save output to: {input_path}")
        truth = get_processed_full_data_dir()
        truth = f"{truth}/train_labels.parquet"
        logging.info(f"read ses2truth from: {truth}")
        df_truth = pd.read_parquet(truth)
        pivot_candidates(
            name="train",
            is_train=True,
            include_all_gt=False,
            drop_zero_positive_sample=True,
            input_path=input_path,
            output_path=output_path,
            df_truth=df_truth,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_candidates_dir()
        output_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"will read input from & save output to: {input_path}")
        pivot_candidates(
            name="test",
            is_train=False,
            include_all_gt=False,
            drop_zero_positive_sample=False,
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
