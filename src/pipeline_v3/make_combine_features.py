import click
import polars as pl
import numpy as np
from tqdm import tqdm
import gc
from src.utils.constants import (
    CFG,
    get_processed_training_train_candidates_dir,  # candidates dir
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
    get_processed_training_train_sess_features_dir,  # session features dir
    get_processed_training_test_sess_features_dir,
    get_processed_scoring_train_sess_features_dir,
    get_processed_scoring_test_sess_features_dir,
    get_processed_training_train_sess_item_features_dir,  # sessionXaid features dir
    get_processed_training_test_sess_item_features_dir,
    get_processed_scoring_train_sess_item_features_dir,
    get_processed_scoring_test_sess_item_features_dir,
    get_processed_training_train_item_features_dir,  # item features dir
    get_processed_training_test_item_features_dir,
    get_processed_scoring_train_item_features_dir,
    get_processed_scoring_test_item_features_dir,
    get_processed_training_train_item_hour_features_dir,  # itemXhour features dir
    get_processed_training_test_item_hour_features_dir,
    get_processed_scoring_train_item_hour_features_dir,
    get_processed_scoring_test_item_hour_features_dir,
    get_processed_training_train_item_weekday_features_dir,  # itemXweekday features dir
    get_processed_training_test_item_weekday_features_dir,
    get_processed_scoring_train_item_weekday_features_dir,
    get_processed_scoring_test_item_weekday_features_dir,
    get_processed_training_train_item_covisitation_features_dir,  # iemCovisitation feature dir
    get_processed_training_test_item_covisitation_features_dir,
    get_processed_scoring_train_item_covisitation_features_dir,
    get_processed_scoring_test_item_covisitation_features_dir,
    get_processed_training_train_matrix_fact_features_dir,  # Matrix Factorization feature dir
    get_processed_training_test_matrix_fact_features_dir,
    get_processed_scoring_train_matrix_fact_features_dir,
    get_processed_scoring_test_matrix_fact_features_dir,
    get_processed_training_train_word2vec_features_dir,  # Word2Vec dir
    get_processed_training_test_word2vec_features_dir,
    get_processed_scoring_train_word2vec_features_dir,
    get_processed_scoring_test_word2vec_features_dir,
    get_processed_training_train_fasttext_features_dir,  # Fasttext dir
    get_processed_training_test_fasttext_features_dir,
    get_processed_scoring_train_fasttext_features_dir,
    get_processed_scoring_test_fasttext_features_dir,
    get_processed_training_train_dataset_dir,  # final dataset dir
    get_processed_training_test_dataset_dir,
    get_processed_scoring_train_dataset_dir,
    get_processed_scoring_test_dataset_dir,
)
from src.utils.memory import freemem
from src.utils.logger import get_logger

logging = get_logger()


def calc_relative_diff_w_mean(df: pl.DataFrame, feature: str) -> pl.DataFrame:
    df = df.with_columns(
        [
            pl.col(f"{feature}").mean().over("session").alias(f"mean_{feature}"),
        ]
    )

    # difference with mean
    df = df.with_columns(
        [
            (pl.col(f"{feature}") - pl.col(f"mean_{feature}")).alias(
                f"diff_w_mean_{feature}"
            ),
        ]
    )

    # relative difference with mean
    df = df.with_columns(
        [
            (pl.col(f"diff_w_mean_{feature}") / pl.col(f"mean_{feature}")).alias(
                f"relative_diff_w_mean_{feature}"
            ),
        ]
    )

    # drop unused cols
    df = df.drop(
        columns=[
            f"mean_{feature}",
        ]
    )

    return df


def fcombine_features(mode: str, event: str, ix: int):
    candidate_path = ""
    session_fea_path = ""
    sessionXaid_fea_path = ""
    item_fea_path = ""
    itemXhour_fea_path = ""
    itemXweekday_fea_path = ""
    itemXcovisit_fea_path = ""
    MatrixFact_fea_path = ""
    Word2Vec_fea_path = ""
    Fasttext_fea_path = ""
    output_path = ""
    name = ""

    if mode == "training_train":
        candidate_path = get_processed_training_train_candidates_dir()
        session_fea_path = get_processed_training_train_sess_features_dir()
        sessionXaid_fea_path = get_processed_training_train_sess_item_features_dir()
        item_fea_path = get_processed_training_train_item_features_dir()
        itemXhour_fea_path = get_processed_training_train_item_hour_features_dir()
        itemXweekday_fea_path = get_processed_training_train_item_weekday_features_dir()
        itemXcovisit_fea_path = (
            get_processed_training_train_item_covisitation_features_dir()
        )
        MatrixFact_fea_path = get_processed_training_train_matrix_fact_features_dir()
        Word2Vec_fea_path = get_processed_training_train_word2vec_features_dir()
        Fasttext_fea_path = get_processed_training_train_fasttext_features_dir()
        output_path = get_processed_training_train_dataset_dir()
        name = "train"

    elif mode == "training_test":
        candidate_path = get_processed_training_test_candidates_dir()
        session_fea_path = get_processed_training_test_sess_features_dir()
        sessionXaid_fea_path = get_processed_training_test_sess_item_features_dir()
        item_fea_path = get_processed_training_test_item_features_dir()
        itemXhour_fea_path = get_processed_training_test_item_hour_features_dir()
        itemXweekday_fea_path = get_processed_training_test_item_weekday_features_dir()
        itemXcovisit_fea_path = (
            get_processed_training_test_item_covisitation_features_dir()
        )
        MatrixFact_fea_path = get_processed_training_test_matrix_fact_features_dir()
        Word2Vec_fea_path = get_processed_training_test_word2vec_features_dir()
        Fasttext_fea_path = get_processed_training_test_fasttext_features_dir()
        output_path = get_processed_training_test_dataset_dir()
        name = "test"

    elif mode == "scoring_train":
        candidate_path = get_processed_scoring_train_candidates_dir()
        session_fea_path = get_processed_scoring_train_sess_features_dir()
        sessionXaid_fea_path = get_processed_scoring_train_sess_item_features_dir()
        item_fea_path = get_processed_scoring_train_item_features_dir()
        itemXhour_fea_path = get_processed_scoring_train_item_hour_features_dir()
        itemXweekday_fea_path = get_processed_scoring_train_item_weekday_features_dir()
        itemXcovisit_fea_path = (
            get_processed_scoring_train_item_covisitation_features_dir()
        )
        MatrixFact_fea_path = get_processed_scoring_train_matrix_fact_features_dir()
        Word2Vec_fea_path = get_processed_scoring_train_word2vec_features_dir()
        Fasttext_fea_path = get_processed_scoring_train_fasttext_features_dir()
        output_path = get_processed_scoring_train_dataset_dir()
        name = "train"

    elif mode == "scoring_test":
        candidate_path = get_processed_scoring_test_candidates_dir()
        session_fea_path = get_processed_scoring_test_sess_features_dir()
        sessionXaid_fea_path = get_processed_scoring_test_sess_item_features_dir()
        item_fea_path = get_processed_scoring_test_item_features_dir()
        itemXhour_fea_path = get_processed_scoring_test_item_hour_features_dir()
        itemXweekday_fea_path = get_processed_scoring_test_item_weekday_features_dir()
        itemXcovisit_fea_path = (
            get_processed_scoring_test_item_covisitation_features_dir()
        )
        MatrixFact_fea_path = get_processed_scoring_test_matrix_fact_features_dir()
        Word2Vec_fea_path = get_processed_scoring_test_word2vec_features_dir()
        Fasttext_fea_path = get_processed_scoring_test_fasttext_features_dir()
        output_path = get_processed_scoring_test_dataset_dir()
        name = "test"

    logging.info(f"read candidate from: {candidate_path}")
    logging.info(f"will save chunks data to: {output_path}")

    c_path = f"{candidate_path}/{name}_{ix}_{event}_rows.parquet"
    sfea_path = f"{session_fea_path}/{name}_{ix}_session_feas.parquet"
    sfeaXaid_path = f"{sessionXaid_fea_path}/{name}_{ix}_session_item_feas.parquet"
    sfeaXaid_l2_path = (
        f"{sessionXaid_fea_path}/{name}_{ix}_session_item_feas_l2.parquet"
    )
    item_path = f"{item_fea_path}/{name}_item_feas.parquet"
    itemXhour_path = f"{itemXhour_fea_path}/{name}_item_hour_feas.parquet"
    itemXweekday_path = f"{itemXweekday_fea_path}/{name}_item_weekday_feas.parquet"
    itemXcovisit_path = (
        f"{itemXcovisit_fea_path}/{name}_{ix}_{event}_item_covisitation_feas.parquet"
    )
    matrix_fact_path = (
        f"{MatrixFact_fea_path}/{name}_{ix}_{event}_matrix_fact_feas.parquet"
    )
    word2vec_path = f"{Word2Vec_fea_path}/{name}_{ix}_{event}_word2vec_feas.parquet"
    fasttext_path = f"{Fasttext_fea_path}/{name}_{ix}_{event}_fasttext_feas.parquet"

    cand_df = pl.read_parquet(c_path)
    # make sure to cast session id & candidate_aid to int32
    cand_df = cand_df.with_columns(
        [
            pl.col("session").cast(pl.Int32).alias("session"),
            pl.col("candidate_aid").cast(pl.Int32).alias("candidate_aid"),
        ]
    )
    logging.info(f"read candidates with shape {cand_df.shape}")

    # read session features
    ses_agg = pl.read_parquet(sfea_path)
    logging.info(f"read session features with shape {ses_agg.shape}")
    cand_df = cand_df.join(ses_agg, how="left", on=["session"])
    logging.info(f"joined with session features! shape {cand_df.shape}")

    del ses_agg
    gc.collect()

    logging.info(f"add log_recency rank_covisit")
    cand_df = cand_df.with_columns(
        [
            pl.col("rank_covisit").max().over("session").alias("max_rank_covisit"),
        ]
    )
    # add log_rank_covisit_score
    linear_interpolation = 0.1 + ((1 - 0.1) / (cand_df["max_rank_covisit"] - 1)) * (
        cand_df["max_rank_covisit"] - cand_df["rank_covisit"]
    )
    cand_df = cand_df.with_columns(
        [
            pl.Series(2**linear_interpolation - 1)
            .alias("log_rank_covisit_score")
            .fill_nan(1),
        ]
    )
    cand_df = cand_df.fill_null(0)

    # drop cols
    cand_df = cand_df.drop(
        columns=[
            "max_rank_covisit",
        ]
    )

    logging.info(f"the log_recency rank_covisit features added! shape {cand_df.shape}")

    # read item-covisitation features
    item_covisit_agg = pl.read_parquet(itemXcovisit_path)
    logging.info(
        f"read sessionXcovisitation features with shape {item_covisit_agg.shape}"
    )
    cand_df = cand_df.join(
        item_covisit_agg,
        how="left",
        left_on=["session", "candidate_aid"],
        right_on=["session", "candidate_aid"],
    ).fill_null(0)

    del item_covisit_agg
    gc.collect()

    cand_df = cand_df.fill_null(0)

    cand_df = cand_df.with_columns(
        [
            np.mean(
                [
                    pl.col("click_weight_with_last_event_in_session_aid"),
                    pl.col("click_weight_with_max_recency_event_in_session_aid"),
                    pl.col(
                        "click_weight_with_max_weighted_recency_event_in_session_aid"
                    ),
                    pl.col("click_weight_with_max_duration_event_in_session_aid"),
                    # pl.col(
                    #     "click_weight_with_max_weighted_log_duration_event_in_session_aid"
                    # ),
                ]
            ).alias("click_weight_mean"),
            pl.max(
                [
                    pl.col("click_weight_with_last_event_in_session_aid"),
                    pl.col("click_weight_with_max_recency_event_in_session_aid"),
                    pl.col(
                        "click_weight_with_max_weighted_recency_event_in_session_aid"
                    ),
                    pl.col("click_weight_with_max_duration_event_in_session_aid"),
                    # pl.col(
                    #     "click_weight_with_max_weighted_log_duration_event_in_session_aid"
                    # ),
                ]
            ).alias("click_weight_max"),
            pl.min(
                [
                    pl.col("click_weight_with_last_event_in_session_aid"),
                    pl.col("click_weight_with_max_recency_event_in_session_aid"),
                    pl.col(
                        "click_weight_with_max_weighted_recency_event_in_session_aid"
                    ),
                    pl.col("click_weight_with_max_duration_event_in_session_aid"),
                    # pl.col(
                    #     "click_weight_with_max_weighted_log_duration_event_in_session_aid"
                    # ),
                ]
            ).alias("click_weight_min"),
            np.mean(
                [
                    pl.col("buys_weight_with_last_event_in_session_aid"),
                    pl.col("buys_weight_with_max_recency_event_in_session_aid"),
                    pl.col(
                        "buys_weight_with_max_weighted_recency_event_in_session_aid"
                    ),
                    pl.col("buys_weight_with_max_duration_event_in_session_aid"),
                    # pl.col(
                    #     "buys_weight_with_max_weighted_log_duration_event_in_session_aid"
                    # ),
                ]
            ).alias("buys_weight_mean"),
            pl.max(
                [
                    pl.col("buys_weight_with_last_event_in_session_aid"),
                    pl.col("buys_weight_with_max_recency_event_in_session_aid"),
                    pl.col(
                        "buys_weight_with_max_weighted_recency_event_in_session_aid"
                    ),
                    pl.col("buys_weight_with_max_duration_event_in_session_aid"),
                    # pl.col(
                    #     "buys_weight_with_max_weighted_log_duration_event_in_session_aid"
                    # ),
                ]
            ).alias("buys_weight_max"),
            pl.min(
                [
                    pl.col("buys_weight_with_last_event_in_session_aid"),
                    pl.col("buys_weight_with_max_recency_event_in_session_aid"),
                    pl.col(
                        "buys_weight_with_max_weighted_recency_event_in_session_aid"
                    ),
                    pl.col("buys_weight_with_max_duration_event_in_session_aid"),
                    # pl.col(
                    #     "buys_weight_with_max_weighted_log_duration_event_in_session_aid"
                    # ),
                ]
            ).alias("buys_weight_min"),
            np.mean(
                [
                    pl.col("buy2buy_weight_with_last_event_in_session_aid"),
                    pl.col("buy2buy_weight_with_max_recency_event_in_session_aid"),
                    pl.col(
                        "buy2buy_weight_with_max_weighted_recency_event_in_session_aid"
                    ),
                    pl.col("buy2buy_weight_with_max_duration_event_in_session_aid"),
                    # pl.col(
                    #     "buy2buy_weight_with_max_weighted_log_duration_event_in_session_aid"
                    # ),
                ]
            ).alias("buy2buy_weight_mean"),
            pl.max(
                [
                    pl.col("buy2buy_weight_with_last_event_in_session_aid"),
                    pl.col("buy2buy_weight_with_max_recency_event_in_session_aid"),
                    pl.col(
                        "buy2buy_weight_with_max_weighted_recency_event_in_session_aid"
                    ),
                    pl.col("buy2buy_weight_with_max_duration_event_in_session_aid"),
                    # pl.col(
                    #     "buy2buy_weight_with_max_weighted_log_duration_event_in_session_aid"
                    # ),
                ]
            ).alias("buy2buy_weight_max"),
            pl.min(
                [
                    pl.col("buy2buy_weight_with_last_event_in_session_aid"),
                    pl.col("buy2buy_weight_with_max_recency_event_in_session_aid"),
                    pl.col(
                        "buy2buy_weight_with_max_weighted_recency_event_in_session_aid"
                    ),
                    pl.col("buy2buy_weight_with_max_duration_event_in_session_aid"),
                    # pl.col(
                    #     "buy2buy_weight_with_max_weighted_log_duration_event_in_session_aid"
                    # ),
                ]
            ).alias("buy2buy_weight_min"),
        ]
    )

    cand_df = cand_df.with_columns(
        [
            (pl.col("buy2buy_weight_max") - pl.col("buy2buy_weight_min")).alias(
                "buy2buy_weight_diff_max_min"
            ),
            (pl.col("buys_weight_max") - pl.col("buys_weight_min")).alias(
                "buys_weight_diff_max_min"
            ),
            (pl.col("click_weight_max") - pl.col("click_weight_min")).alias(
                "click_weight_diff_max_min"
            ),
        ]
    )

    logging.info(f"joined with sessionXcovisitation features! shape {cand_df.shape}")

    # # read matrix factorization features
    # matrix_fact_fea = pl.read_parquet(matrix_fact_path)
    # logging.info(
    #     f"read sessionXmatrix_fact features with shape {matrix_fact_fea.shape}"
    # )
    # cand_df = cand_df.join(
    #     matrix_fact_fea,
    #     how="left",
    #     left_on=["session", "candidate_aid"],
    #     right_on=["session", "candidate_aid"],
    # )
    # logging.info(f"joined with sessionXmatrix_fact features! shape {cand_df.shape}")

    # del matrix_fact_fea
    # gc.collect()

    # read word2vec features
    word2vec_fea_df = pl.read_parquet(word2vec_path)
    logging.info(f"read sessionXword2vec features with shape {word2vec_fea_df.shape}")
    cand_df = cand_df.join(
        word2vec_fea_df,
        how="left",
        left_on=["session", "candidate_aid"],
        right_on=["session", "candidate_aid"],
    )

    del word2vec_fea_df
    gc.collect()

    cand_df = cand_df.with_columns(
        [
            pl.min(
                [
                    pl.col("word2vec_skipgram_last_event_euclidean_distance"),
                    # pl.col("word2vec_skipgram_max_recency_euclidean_distance"),
                    # pl.col("word2vec_skipgram_max_weighted_recency_euclidean_distance"),
                    pl.col("word2vec_skipgram_max_duration_euclidean_distance"),
                    # pl.col(
                    #     "word2vec_skipgram_max_weighted_duration_euclidean_distance"
                    # ),
                ]
            ).alias("word2vec_euclidean_distance_min"),
        ]
    )

    logging.info(f"joined with sessionXword2vec features! shape {cand_df.shape}")

    # # read fasttext features
    # fasttext_fea_df = pl.read_parquet(fasttext_path)
    # logging.info(f"read sessionXfasttext features with shape {fasttext_fea_df.shape}")
    # cand_df = cand_df.join(
    #     fasttext_fea_df,
    #     how="left",
    #     left_on=["session", "candidate_aid"],
    #     right_on=["session", "candidate_aid"],
    # )
    # logging.info(f"joined with sessionXfasttext features! shape {cand_df.shape}")

    # del fasttext_fea_df
    # gc.collect()

    # read session-item features
    ses_aid_agg = pl.read_parquet(sfeaXaid_path)
    logging.info(f"read sessionXaid features with shape {ses_aid_agg.shape}")
    cand_df = cand_df.join(
        ses_aid_agg,
        how="left",
        left_on=["session", "candidate_aid"],
        right_on=["session", "aid"],
    )
    logging.info(f"joined with sessionXaid features! shape {cand_df.shape}")

    # get ratio of sesXaid_mins_from_last_event with sess_durations
    cand_df = cand_df.with_columns(
        [
            (pl.col("sesXaid_mins_from_last_event") / pl.col("sess_duration_mins"))
            .fill_null(99999)
            .alias("sesXaid_frac_mins_from_last_event_to_sess_duration")
        ]
    )

    cand_df = cand_df.with_columns(
        [
            # the higher the better, fill_null with 0
            pl.col("sesXaid_type_weighted_log_recency_score")
            .fill_null(0)
            .alias("sesXaid_type_weighted_log_recency_score"),
            pl.col("sesXaid_log_recency_score")
            .fill_null(0)
            .alias("sesXaid_log_recency_score"),
            # pl.col("sesXaid_events_count").fill_null(0).alias("sesXaid_events_count"),
            pl.col("sesXaid_click_count").fill_null(0).alias("sesXaid_click_count"),
            pl.col("sesXaid_cart_count").fill_null(0).alias("sesXaid_cart_count"),
            # pl.col("sesXaid_order_count").fill_null(0).alias("sesXaid_order_count"),
            pl.col("sesXaid_avg_click_dur_sec")
            .fill_null(0)
            .alias("sesXaid_avg_click_dur_sec"),
            pl.col("sesXaid_avg_cart_dur_sec")
            .fill_null(0)
            .alias("sesXaid_avg_cart_dur_sec"),
            pl.col("sesXaid_type_dcount").fill_null(0).alias("sesXaid_type_dcount"),
            # the lower the better, fill_null with high number
            pl.col("sesXaid_action_num_reverse_chrono")
            .fill_null(500)
            .alias("sesXaid_action_num_reverse_chrono"),
            pl.col("sesXaid_mins_from_last_event")
            .fill_null(9999.0)
            .alias("sesXaid_mins_from_last_event"),
            pl.col("sesXaid_mins_from_last_event_log1p")
            .fill_null(99.0)
            .alias("sesXaid_mins_from_last_event_log1p"),
        ]
    )
    cand_df = cand_df.fill_null(0)

    del ses_aid_agg
    gc.collect()

    # additional features in covisit weights
    # expressing how far the weight of candidate over avg of weight with others candidate
    # mean over session
    logging.info(f"start calc relative dist w mean features!")
    DIF_FEAS = [
        "click_weight_with_last_event_in_session_aid",
        "buys_weight_with_last_event_in_session_aid",
        "buy2buy_weight_with_last_event_in_session_aid",
        "word2vec_skipgram_last_event_cosine_distance",
        "word2vec_skipgram_last_event_euclidean_distance",
        "word2vec_skipgram_max_duration_cosine_distance",
    ]
    for f in tqdm(DIF_FEAS):
        cand_df = calc_relative_diff_w_mean(df=cand_df, feature=f)

    # fill 0
    cand_df = cand_df.fill_nan(0)

    logging.info(f"complete calc relative dist w mean features! shape {cand_df.shape}")

    # read item features
    item_agg = pl.read_parquet(item_path)
    logging.info(f"read item features with shape {item_agg.shape}")
    cand_df = cand_df.join(
        item_agg,
        how="left",
        left_on=["candidate_aid"],
        right_on=["aid"],
    )
    logging.info(f"joined with item features! shape {cand_df.shape}")

    del item_agg
    gc.collect()

    cand_df = cand_df.with_columns(
        [
            # np.abs(pl.col("sess_hour") - pl.col("item_avg_hour_click"))
            # .cast(pl.Int32)
            # .fill_null(99)
            # .alias("sessXitem_abs_diff_avg_hour_click"),
            # np.abs(pl.col("sess_hour") - pl.col("item_avg_hour_cart"))
            # .cast(pl.Int32)
            # .fill_null(99)
            # .alias("sessXitem_abs_diff_avg_hour_cart"),
            np.abs(pl.col("sess_hour") - pl.col("item_avg_hour_order"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sessXitem_abs_diff_avg_hour_order"),
            # np.abs(pl.col("sess_weekday") - pl.col("item_avg_weekday_click"))
            # .cast(pl.Int32)
            # .fill_null(99)
            # .alias("sessXitem_abs_diff_avg_weekday_click"),
            # np.abs(pl.col("sess_weekday") - pl.col("item_avg_weekday_cart"))
            # .cast(pl.Int32)
            # .fill_null(99)
            # .alias("sessXitem_abs_diff_avg_weekday_cart"),
            np.abs(pl.col("sess_weekday") - pl.col("item_avg_weekday_order"))
            .cast(pl.Int32)
            .fill_null(99)
            .alias("sessXitem_abs_diff_avg_weekday_order"),
        ]
    )
    cand_df = cand_df.fill_null(0)

    # # read session-item l2 features
    # ses_aid_agg = pl.read_parquet(sfeaXaid_l2_path)
    # logging.info(f"read sessionXaid l2 features with shape {ses_aid_agg.shape}")
    # cand_df = cand_df.join(
    #     ses_aid_agg,
    #     how="left",
    #     left_on=["session", "candidate_aid"],
    #     right_on=["session", "candidate_aid"],
    # )
    # logging.info(f"joined with sessionXaid l2 features! shape {cand_df.shape}")

    # read item-hour features
    item_hour_agg = pl.read_parquet(itemXhour_path)
    logging.info(f"read item-hour features with shape {item_hour_agg.shape}")
    cand_df = cand_df.join(
        item_hour_agg,
        how="left",
        left_on=["candidate_aid", "sess_hour"],
        right_on=["aid", "hour"],
    ).fill_null(0)
    logging.info(f"joined with item-hour features! shape {cand_df.shape}")

    del item_hour_agg
    gc.collect()

    # read item-weekday features
    item_weekday_agg = pl.read_parquet(itemXweekday_path)
    logging.info(f"read item-weekday features with shape {item_weekday_agg.shape}")
    cand_df = cand_df.join(
        item_weekday_agg,
        how="left",
        left_on=["candidate_aid", "sess_weekday"],
        right_on=["aid", "weekday"],
    ).fill_null(0)
    logging.info(f"joined with item-weekday features! shape {cand_df.shape}")

    del item_weekday_agg
    gc.collect()

    # drop cols
    cand_df = cand_df.drop(
        columns=[
            "sess_hour",
            "sess_weekday",
            "buy2buy_weight_min",
            "click_weight_min",
            "itemXhour_all_events_count",
            "itemXweekday_order_count",
        ]
    )

    logging.info(f"drop some features! shape {cand_df.shape}")

    filepath = f"{output_path}/{name}_{ix}_{event}_combined.parquet"
    logging.info(f"save chunk to: {filepath}")
    cand_df = freemem(cand_df)
    cand_df.write_parquet(f"{filepath}")
    logging.info(f"output df shape {cand_df.shape}")

    del cand_df
    gc.collect()


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
@click.option(
    "--istart",
    default=0,
    help="index start",
)
@click.option(
    "--iend",
    default=10,
    help="index end",
)
def main(mode: str, istart: int, iend: int):
    if mode == "training_train":
        n = CFG.N_train
    elif mode == "training_test":
        n = CFG.N_local_test
    else:
        n = CFG.N_test

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(istart, iend)):
        for event in ["clicks", "carts", "orders"]:

            if (mode == "training_train") & (event == "clicks") & (ix > 6):
                logging.info("click ix > 6 continue")
                continue

            logging.info(f"start combining features")
            fcombine_features(mode=mode, event=event, ix=ix)


if __name__ == "__main__":
    main()
