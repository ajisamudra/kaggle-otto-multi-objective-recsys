import click
import pandas as pd
import polars as pl
from tqdm import tqdm
import gc
from pathlib import Path
import random
from sklearn.utils import resample
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
TARGET = "label"

# read chunk parquet
# for each chunk -> pivot N candidates into rows
# so for each session, there's N-1 additional rows


def downsample(df: pd.DataFrame, negative_ratio: int):
    # desired_ratio = 20
    positive_class = df[df[TARGET] == 1]
    negative_class = df[df[TARGET] == 0]
    negative_downsample = resample(
        negative_class,
        replace=False,
        n_samples=len(positive_class) * negative_ratio,
        random_state=777,
    )

    df = pd.concat([positive_class, negative_downsample], ignore_index=True)
    df = df.sort_values(by=["session", TARGET], ascending=[True, True])

    return df


def get_ses2candidates(df: pd.DataFrame):
    """
    session | labels
    123_clicks | [aid1, aid2]
    123_orders | [aid1, aid2, aid3]
    """

    df["session"] = df["session"].apply(lambda x: int(x.split("_")[0]))
    ses2candidates = dict(zip(df["session"], df["labels"]))
    ses2cand_ranks = dict(zip(df["session"], df["ranks"]))

    return ses2candidates, ses2cand_ranks


def pivot_candidates_list_to_rows(
    unique_sessions: list,
    covisit_ses2candidates: dict,
    fasttext_ses2candidates: dict,
    word2vec_ses2candidates: dict,
    matrix_fact_ses2candidates: dict,
    popular_week_ses2candidates: dict,
    # popular_hour_ses2candidates: dict,
    covisit_ses2cand_ranks: dict,
    fasttext_ses2cand_ranks: dict,
    word2vec_ses2cand_ranks: dict,
    matrix_fact_ses2cand_ranks: dict,
    popular_week_ses2cand_ranks: dict,
    # popular_hour_ses2cand_ranks: dict,
    is_train: bool,
    exclude_training_zero_positive: bool,
    drop_zero_positive_sample: bool,
    ses2truth: dict = {},
    ratio_negative_sample: int = 1,
) -> pl.DataFrame:
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

    # candidate aid
    candidates_covisit = []
    candidates_word2vec = []
    candidates_fasttext = []
    candidates_popular_hour = []
    candidates_popular_week = []
    candidates_matrix_fact = []

    # rank per candidate strategy
    ranks_covisit = []
    ranks_word2vec = []
    ranks_fasttext = []
    ranks_popular_hour = []
    ranks_popular_week = []
    ranks_matrix_fact = []

    logging.info(
        f"pivot in train_mode: {is_train} | exclude_training_zero_positive: {exclude_training_zero_positive} | drop_zero_positive_sample: {drop_zero_positive_sample}"
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

        # placeholder for rank
        rank_covisit = []
        rank_word2vec = []
        rank_fasttext = []
        # rank_popular_hour = []
        rank_popular_week = []
        rank_matrix_fact = []

        # get rank based on strategies
        covisit_ranks = list(covisit_ses2cand_ranks[session])
        fasttext_ranks = list(fasttext_ses2cand_ranks[session])
        word2vec_ranks = list(word2vec_ses2cand_ranks[session])
        matrix_fact_ranks = list(matrix_fact_ses2cand_ranks[session])
        popular_week_ranks = list(popular_week_ses2cand_ranks[session])
        # popular_hour_ranks = list(popular_hour_ses2cand_ranks[session])

        # get candidates for specific session
        # covisitation candidates
        covisit_cands = list(covisit_ses2candidates[session])
        cands = list(covisit_ses2candidates[session])

        # update rank for covisit and fill max(rank) for each strategy
        rank_covisit.extend(covisit_ranks)
        rank_word2vec.extend([len(word2vec_ranks) for i in range(len(covisit_cands))])
        rank_fasttext.extend([len(fasttext_ranks) for i in range(len(covisit_cands))])
        # rank_popular_hour.extend(
        #     [len(popular_hour_ranks) for i in range(len(covisit_cands))]
        # )
        rank_popular_week.extend(
            [len(popular_week_ranks) for i in range(len(covisit_cands))]
        )
        rank_matrix_fact.extend(
            [len(matrix_fact_ranks) for i in range(len(covisit_cands))]
        )

        # fasttext candidates
        fasttext_cands = list(fasttext_ses2candidates[session])
        cands.extend(fasttext_cands)

        # update rank for fasttext and fill max(rank) for each strategy
        rank_covisit.extend([len(covisit_ranks) for i in range(len(fasttext_cands))])
        rank_word2vec.extend([len(word2vec_ranks) for i in range(len(fasttext_cands))])
        rank_fasttext.extend(fasttext_ranks)
        # rank_popular_hour.extend(
        #     [len(popular_hour_ranks) for i in range(len(fasttext_cands))]
        # )
        rank_popular_week.extend(
            [len(popular_week_ranks) for i in range(len(fasttext_cands))]
        )
        rank_matrix_fact.extend(
            [len(matrix_fact_ranks) for i in range(len(fasttext_cands))]
        )

        # word2vec candidates
        word2vec_cands = list(word2vec_ses2candidates[session])
        cands.extend(word2vec_cands)

        # update rank for word2vec and fill max(rank) for each strategy
        rank_covisit.extend([len(covisit_ranks) for i in range(len(word2vec_cands))])
        rank_word2vec.extend(word2vec_ranks)
        rank_fasttext.extend([len(fasttext_ranks) for i in range(len(word2vec_cands))])
        # rank_popular_hour.extend(
        #     [len(popular_hour_ranks) for i in range(len(word2vec_cands))]
        # )
        rank_popular_week.extend(
            [len(popular_week_ranks) for i in range(len(word2vec_cands))]
        )
        rank_matrix_fact.extend(
            [len(matrix_fact_ranks) for i in range(len(word2vec_cands))]
        )

        # matrix fact candidates
        matrix_fact_cands = list(matrix_fact_ses2candidates[session])
        cands.extend(matrix_fact_cands)

        # update rank for matrix fact and fill max(rank) for each strategy
        rank_covisit.extend([len(covisit_ranks) for i in range(len(matrix_fact_cands))])
        rank_word2vec.extend(
            [len(word2vec_ranks) for i in range(len(matrix_fact_cands))]
        )
        rank_fasttext.extend(
            [len(fasttext_ranks) for i in range(len(matrix_fact_cands))]
        )
        # rank_popular_hour.extend(
        #     [len(popular_hour_ranks) for i in range(len(matrix_fact_cands))]
        # )
        rank_popular_week.extend(
            [len(popular_week_ranks) for i in range(len(matrix_fact_cands))]
        )
        rank_matrix_fact.extend(matrix_fact_ranks)

        # popular week candidates
        popular_week_cands = list(popular_week_ses2candidates[session])
        cands.extend(popular_week_cands)

        # update rank for popular week and fill max(rank) for each strategy
        rank_covisit.extend(
            [len(covisit_ranks) for i in range(len(popular_week_cands))]
        )
        rank_word2vec.extend(
            [len(word2vec_ranks) for i in range(len(popular_week_cands))]
        )
        rank_fasttext.extend(
            [len(fasttext_ranks) for i in range(len(popular_week_cands))]
        )
        # rank_popular_hour.extend(
        #     [len(popular_hour_ranks) for i in range(len(popular_week_cands))]
        # )
        rank_popular_week.extend(popular_week_ranks)
        rank_matrix_fact.extend(
            [len(matrix_fact_ranks) for i in range(len(popular_week_cands))]
        )

        # # popular hour candidates
        # popular_hour_cands = list(popular_hour_ses2candidates[session])
        # cands.extend(popular_hour_cands)

        # # update rank for popular hour and fill max(rank) for each strategy
        # rank_covisit.extend(
        #     [len(covisit_ranks) for i in range(len(popular_hour_cands))]
        # )
        # rank_word2vec.extend(
        #     [len(word2vec_ranks) for i in range(len(popular_hour_cands))]
        # )
        # rank_fasttext.extend(
        #     [len(fasttext_ranks) for i in range(len(popular_hour_cands))]
        # )
        # rank_popular_hour.extend(popular_hour_ranks)
        # rank_popular_week.extend(
        #     [len(popular_hour_ranks) for i in range(len(popular_hour_cands))]
        # )
        # rank_matrix_fact.extend(
        #     [len(matrix_fact_ranks) for i in range(len(popular_hour_cands))]
        # )

        # # drop duplicate
        # cands = set(cands)

        # if include_all_gt:
        #     # add all truths if it's training data
        #     cands = cands | truths

        # check whether it's in truths
        label_ls = [1 if c in truths else 0 for c in cands]
        if exclude_training_zero_positive:
            # if there's no truth for specific event & session
            # drop from training data
            if sum(label_ls) == 0:
                continue

        session_ls = [session for i in range(len(cands))]
        cands_ls = cands

        # ohe strategy retrieval
        covisit_ls = [1 if c in covisit_cands else 0 for c in cands]
        fasttext_ls = [1 if c in fasttext_cands else 0 for c in cands]
        word2vec_ls = [1 if c in word2vec_cands else 0 for c in cands]
        matrix_fact_ls = [1 if c in matrix_fact_cands else 0 for c in cands]
        # popular_hour_ls = [1 if c in popular_hour_cands else 0 for c in cands]
        popular_week_ls = [1 if c in popular_week_cands else 0 for c in cands]

        # rank strategy retrieval
        rank_covisit_ls = rank_covisit
        rank_word2vec_ls = rank_word2vec
        rank_fasttext_ls = rank_fasttext
        # rank_popular_hour_ls = rank_popular_hour
        rank_popular_week_ls = rank_popular_week
        rank_matrix_fact_ls = rank_matrix_fact

        # extend the overall list
        sessions.extend(session_ls)
        candidates.extend(cands_ls)
        labels.extend(label_ls)

        candidates_covisit.extend(covisit_ls)
        candidates_word2vec.extend(word2vec_ls)
        candidates_fasttext.extend(fasttext_ls)
        # candidates_popular_hour.extend(popular_hour_ls)
        candidates_popular_week.extend(popular_week_ls)
        candidates_matrix_fact.extend(matrix_fact_ls)

        ranks_covisit.extend(rank_covisit_ls)
        ranks_word2vec.extend(rank_word2vec_ls)
        ranks_fasttext.extend(rank_fasttext_ls)
        # ranks_popular_hour.extend(rank_popular_hour_ls)
        ranks_popular_week.extend(rank_popular_week_ls)
        ranks_matrix_fact.extend(rank_matrix_fact_ls)

    # save as df
    data = {
        "session": sessions,
        "candidate_aid": candidates,
        "retrieval_covisit": candidates_covisit,
        "retrieval_word2vec": candidates_word2vec,
        "retrieval_fasttext": candidates_fasttext,
        # "retrieval_popular_hour": candidates_popular_hour,
        "retrieval_popular_week": candidates_popular_week,
        "retrieval_matrix_fact": candidates_matrix_fact,
        "rank_covisit": ranks_covisit,
        "rank_word2vec": ranks_word2vec,
        "rank_fasttext": ranks_fasttext,
        # "rank_popular_hour": ranks_popular_hour,
        "rank_popular_week": ranks_popular_week,
        "rank_matrix_fact": ranks_matrix_fact,
        "label": labels,
    }

    data = pl.DataFrame(data)
    # remove duplicate
    data = data.groupby(["session", "candidate_aid"]).agg(
        [
            pl.col("label").max(),
            pl.col("retrieval_covisit").max(),
            pl.col("retrieval_word2vec").max(),
            pl.col("retrieval_fasttext").max(),
            # pl.col("retrieval_popular_hour").max(),
            pl.col("retrieval_popular_week").max(),
            pl.col("retrieval_matrix_fact").max(),
            pl.col("rank_covisit").min(),
            pl.col("rank_word2vec").min(),
            pl.col("rank_fasttext").min(),
            # pl.col("rank_popular_hour").min(),
            pl.col("rank_popular_week").min(),
            pl.col("rank_matrix_fact").min(),
        ]
    )

    # downsample
    if ratio_negative_sample > 1:
        data = downsample(df=data.to_pandas(), negative_ratio=ratio_negative_sample)
        data = pl.from_pandas(data)

    return data


def pivot_candidates(
    name: str,
    is_train: bool,
    exclude_training_zero_positive: bool,
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
            covisit_ses2candidates, covisit_ses2cand_ranks = get_ses2candidates(cand_df)
            unique_sessions = list(cand_df["session"].values)

            del cand_df
            gc.collect()

            # candidate #2 fasttext
            filepath = f"{input_path}/{name}_{ix}_{event}_fasttext_list.parquet"
            cand_df = pd.read_parquet(filepath)
            fasttext_ses2candidates, fasttext_ses2cand_ranks = get_ses2candidates(
                cand_df
            )

            del cand_df
            gc.collect()

            # candidate #3 word2vec
            filepath = f"{input_path}/{name}_{ix}_{event}_word2vec_list.parquet"
            cand_df = pd.read_parquet(filepath)
            word2vec_ses2candidates, word2vec_ses2cand_ranks = get_ses2candidates(
                cand_df
            )

            del cand_df
            gc.collect()

            # candidate #3 matrix factorization
            filepath = f"{input_path}/{name}_{ix}_{event}_matrix_fact_list.parquet"
            cand_df = pd.read_parquet(filepath)
            matrix_fact_ses2candidates, matrix_fact_ses2cand_ranks = get_ses2candidates(
                cand_df
            )

            del cand_df
            gc.collect()

            # candidate #5 popular week
            filepath = f"{input_path}/{name}_{ix}_{event}_popular_week_list.parquet"
            cand_df = pd.read_parquet(filepath)
            (
                popular_week_ses2candidates,
                popular_week_ses2cand_ranks,
            ) = get_ses2candidates(cand_df)

            del cand_df
            gc.collect()

            # # candidate #6 popular hour
            # filepath = f"{input_path}/{name}_{ix}_{event}_popular_hour_list.parquet"
            # cand_df = pd.read_parquet(filepath)
            # (
            #     popular_hour_ses2candidates,
            #     popular_hour_ses2cand_ranks,
            # ) = get_ses2candidates(cand_df)

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
                popular_week_ses2candidates=popular_week_ses2candidates,
                # popular_hour_ses2candidates=popular_hour_ses2candidates,
                covisit_ses2cand_ranks=covisit_ses2cand_ranks,
                word2vec_ses2cand_ranks=word2vec_ses2cand_ranks,
                fasttext_ses2cand_ranks=fasttext_ses2cand_ranks,
                matrix_fact_ses2cand_ranks=matrix_fact_ses2cand_ranks,
                popular_week_ses2cand_ranks=popular_week_ses2cand_ranks,
                # popular_hour_ses2cand_ranks=popular_hour_ses2cand_ranks,
                is_train=is_train,
                exclude_training_zero_positive=exclude_training_zero_positive,
                ratio_negative_sample=ratio_negative_sample,
                drop_zero_positive_sample=drop_zero_positive_sample,
                ses2truth=ses2truth,
            )

            filepath = output_path / f"{name}_{ix}_{event}_rows.parquet"
            logging.info(f"save chunk {ix}_{event} to: {filepath}")
            df_output.write_parquet(f"{filepath}")
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
            exclude_training_zero_positive=True,
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
            exclude_training_zero_positive=False,
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
            exclude_training_zero_positive=False,
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
            exclude_training_zero_positive=False,
            drop_zero_positive_sample=False,
            input_path=input_path,
            output_path=output_path,
        )


if __name__ == "__main__":
    main()
