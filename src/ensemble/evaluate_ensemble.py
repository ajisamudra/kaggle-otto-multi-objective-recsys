import click
import polars as pl
from tqdm import tqdm
import numpy as np
import gc
import joblib
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_local_validation_dir,
    get_data_output_local_submission_dir,  # scoring output dir
    get_data_output_local_ensemble_submission_dir,
)

from src.metrics.submission_evaluation import measure_recall
from src.utils.memory import freemem, round_float_3decimals
from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def evaluate_ensemble():
    N_test = 100
    week_model = "w2"
    CONFIG = {
        "carts_models": [
            "2022-12-25_carts_cat_ranker_65389_94260",
            "2022-12-25_carts_lgbm_classifier_66769_94771",
            "2022-12-26_carts_cat_classifier_67369_94709",
            "2022-12-26_carts_lgbm_ranker_66196_93867",
        ],
        "carts_powers": [1, 1, 1, 1],
        "carts_weights": [
            0.21217365264243337,
            0.13146330225407726,
            0.5424576834298213,
            0.9647752025119785,
        ],
        "clicks_models": [
            "2022-12-25_clicks_cat_ranker_45439_89900",
            "2022-12-25_clicks_lgbm_classifier_48807_90926",
            "2022-12-26_clicks_cat_classifier_49243_90940",
            "2022-12-26_clicks_lgbm_ranker_48370_89600",
        ],
        "clicks_powers": [1, 1, 1, 1],
        "clicks_weights": [
            0.3077559158920544,
            0.7141962809446453,
            0.9794595914392629,
            0.02759986680942235,
        ],
        "orders_models": [
            "2022-12-25_orders_cat_ranker_80132_96912",
            "2022-12-25_orders_lgbm_classifier_81484_97385",
            "2022-12-26_orders_cat_classifier_81621_97386",
            "2022-12-26_orders_lgbm_ranker_78372_95814",
        ],
        "orders_powers": [1, 1, 1, 1],
        "orders_weights": [
            0.6103638863604604,
            0.9505596395979229,
            0.8177860281174677,
            0.05283426937328066,
        ],
    }

    events_model_name = []
    pred_rows = 0
    for EVENT in ["clicks", "carts", "orders"]:
        logging.info(f"start applying weights in event: {EVENT.upper()}")
        ARTIFACTS = CONFIG[f"{EVENT}_models"]
        WEIGHTS = CONFIG[f"{EVENT}_weights"]
        POWERS = CONFIG[f"{EVENT}_powers"]
        event_model_name = ""  # for accessing submission later
        for i in tqdm(range(N_test)):
            df_chunk = pl.DataFrame()
            for ix in range(len(ARTIFACTS)):
                input_path = get_data_output_local_submission_dir(
                    event=EVENT, model=ARTIFACTS[ix], week_model=week_model
                )
                tmp_path = f"{input_path}/test_{i}_{EVENT}_scores.parquet"
                df_tmp = pl.read_parquet(tmp_path)
                # apply weights
                df_tmp = df_tmp.with_columns(
                    [(pow(pl.col("score"), POWERS[ix])) * WEIGHTS[ix]]
                )
                df_chunk = pl.concat([df_chunk, df_tmp])

                del df_tmp
                gc.collect()

            # sum weightes scores per candidate aid
            df_chunk = df_chunk.groupby(["session", "candidate_aid", "label"]).agg(
                [pl.col("score").sum()]
            )

            # save weighted scores
            event_model_name = "_".join(ARTIFACTS)
            output_path = get_data_output_local_ensemble_submission_dir(
                event=EVENT, model=event_model_name, week_model=week_model
            )

            tmp_path = f"{output_path}/test_{i}_{EVENT}_ensemble_scores.parquet"
            df_chunk = freemem(df_chunk)
            df_chunk = round_float_3decimals(df_chunk)
            df_chunk.write_parquet(tmp_path)

            # take top 20 candidate aid and save it as list
            test_predictions = (
                df_chunk.sort(["session", "score"], reverse=True)
                .groupby("session")
                .agg([pl.col("candidate_aid").limit(20).list().alias("labels")])
            )

            del df_chunk
            gc.collect()

            test_predictions = test_predictions.select([pl.col(["session", "labels"])])
            test_predictions = test_predictions.with_columns(
                [(pl.col(["session"]) + f"_{EVENT}").alias("session_type")]
            )
            test_predictions = test_predictions.select(
                [pl.col(["session_type", "labels"])]
            )
            test_predictions = test_predictions.with_columns(
                [
                    pl.col("labels")
                    .apply(lambda x: " ".join(map(str, x)))
                    .alias("labels")
                ]
            )

            output_path = get_data_output_local_ensemble_submission_dir(
                event="submission", model=event_model_name, week_model=week_model
            )
            tmp_path = f"{output_path}/test_{i}_{EVENT}_submission.parquet"
            test_predictions = freemem(test_predictions)
            test_predictions.write_parquet(f"{tmp_path}")

            pred_rows += test_predictions.shape[0]

            del test_predictions
            gc.collect()

        events_model_name.append(event_model_name)
        logging.info(f"ensemble calculation complete for {EVENT.upper()}!")

    logging.info(f"predictions rows {pred_rows}")
    logging.info("start collecting submission")
    df_pred = pl.DataFrame()
    for ix, EVENT in enumerate(["clicks", "carts", "orders"]):
        sub_path = get_data_output_local_ensemble_submission_dir(
            event="submission", model=events_model_name[ix], week_model=week_model
        )
        for i in tqdm(range(N_test)):
            tmp_path = f"{sub_path}/test_{i}_{EVENT}_submission.parquet"
            df_chunk = pl.read_parquet(f"{tmp_path}")
            df_pred = pl.concat([df_pred, df_chunk])

            del df_chunk
            gc.collect()

    logging.info("start eval submission")
    # read ground truth
    ground_truth_path = get_processed_local_validation_dir()
    df_truth = pl.read_parquet(f"{ground_truth_path}/test_labels.parquet")
    logging.info(f"ground truth shape {df_truth.shape}")
    logging.info(f"prediction shape {df_pred.shape}")
    # compute metrics
    dict_metrics = measure_recall(
        df_pred=df_pred.to_pandas(), df_truth=df_truth.to_pandas(), Ks=[20]
    )

    recall20 = float(dict_metrics["overall_recall@20"])
    logging.info(f"overall recall@20: {recall20}")

    del df_pred, df_truth
    gc.collect()


def main():
    evaluate_ensemble()


if __name__ == "__main__":
    main()
