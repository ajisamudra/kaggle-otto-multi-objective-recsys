import click
import polars as pl
from tqdm import tqdm
import numpy as np
import gc
import joblib
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_data_output_ensemble_submission_dir,
)

from src.metrics.submission_evaluation import measure_recall
from src.utils.memory import freemem, round_float_3decimals
from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def evaluate_ensemble():
    N_tests = [80, 100]
    week_model = "w2"
    CONFIG = {
        "carts_models": [
            "2022-12-20_carts_cat_ranker_62107_93343_2022-12-20_carts_lgbm_classifier_63895_93925",
            "2022-12-23_carts_cat_ranker_62356_93099_2022-12-23_carts_lgbm_classifier_63946_93662",
        ],
        "clicks_models": [
            "2022-12-20_clicks_cat_ranker_42966_88609_2022-12-20_clicks_lgbm_classifier_46744_89740",
            "2022-12-23_clicks_cat_ranker_43213_88370_2022-12-23_clicks_lgbm_classifier_46496_89386",
        ],
        "orders_models": [
            "2022-12-20_orders_cat_ranker_78471_96474_2022-12-20_orders_lgbm_classifier_79792_97018",
            "2022-12-23_orders_cat_ranker_78326_96000_2022-12-23_orders_lgbm_classifier_79580_96565",
        ],
    }

    events_model_name = []
    pred_rows = 0
    for EVENT in ["clicks", "carts", "orders"]:
        logging.info(f"start aggregating candidate aids in event: {EVENT.upper()}")
        ARTIFACTS = CONFIG[f"{EVENT}_models"]
        event_model_name = ""  # for accessing submission later
        df_pred = pl.DataFrame()
        for i in tqdm(range(100)):
            # artifact#0
            if i <= 79:
                input_path = get_data_output_ensemble_submission_dir(
                    event=EVENT, model=ARTIFACTS[0], week_model=week_model
                )
                tmp_path = f"{input_path}/test_{i}_{EVENT}_ensemble_scores.parquet"
                df_tmp = pl.read_parquet(tmp_path)
                df_tmp = df_tmp.select(["session", "candidate_aid", "score"])
                # # apply weights
                # df_tmp = df_tmp.with_columns(
                #     [(pow(pl.col("score"), POWERS[ix])) * WEIGHTS[ix]]
                # )
                df_pred = pl.concat([df_pred, df_tmp])
                del df_tmp
                gc.collect()

            # artifact#1
            input_path = get_data_output_ensemble_submission_dir(
                event=EVENT, model=ARTIFACTS[1], week_model=week_model
            )
            tmp_path = f"{input_path}/test_{i}_{EVENT}_ensemble_scores.parquet"
            df_tmp = pl.read_parquet(tmp_path)
            df_tmp = df_tmp.select(["session", "candidate_aid", "score"])
            # # apply weights
            # df_tmp = df_tmp.with_columns(
            #     [(pow(pl.col("score"), POWERS[ix])) * WEIGHTS[ix]]
            # )
            df_pred = pl.concat([df_pred, df_tmp])

            del df_tmp
            gc.collect()

            # simple average each candidate aid
            # to reduce num rows in df_pred
            df_pred = df_pred.groupby(["session", "candidate_aid"]).agg(
                [pl.col("score").mean()]
            )

        # save weighted scores
        event_model_name = "_".join(ARTIFACTS)
        output_path = get_data_output_ensemble_submission_dir(
            event=f"{EVENT}_multi_retrieval",
            model=ARTIFACTS[0],
            week_model=week_model,
        )

        tmp_path = f"{output_path}/test_{EVENT}_multiretrieval_scores.parquet"
        df_pred = freemem(df_pred)
        df_pred = round_float_3decimals(df_pred)
        df_pred.write_parquet(tmp_path)

        # take top 20 candidate aid and save it as list
        test_predictions = (
            df_pred.sort(["session", "score"], reverse=True)
            .groupby("session")
            .agg([pl.col("candidate_aid").limit(20).list().alias("labels")])
        )

        del df_pred
        gc.collect()

        test_predictions = test_predictions.select([pl.col(["session", "labels"])])
        test_predictions = test_predictions.with_columns(
            [(pl.col(["session"]) + f"_{EVENT}").alias("session_type")]
        )
        test_predictions = test_predictions.select([pl.col(["session_type", "labels"])])
        test_predictions = test_predictions.with_columns(
            [pl.col("labels").apply(lambda x: " ".join(map(str, x))).alias("labels")]
        )

        output_path = get_data_output_ensemble_submission_dir(
            event="submission_multi_retrieval",
            model=ARTIFACTS[0],
            week_model=week_model,
        )
        tmp_path = f"{output_path}/test_{EVENT}_submission.parquet"
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
        sub_path = get_data_output_ensemble_submission_dir(
            event="submission_multi_retrieval",
            model=ARTIFACTS[0],
            week_model=week_model,
        )

        tmp_path = f"{sub_path}/test_{EVENT}_submission.parquet"
        df_chunk = pl.read_parquet(f"{tmp_path}")
        df_pred = pl.concat([df_pred, df_chunk])

        del df_chunk
        gc.collect()

    # save final submission to csv
    logging.info("save submission to csv")
    sub_path = get_data_output_ensemble_submission_dir(
        event="final_submission_multi_retrieval",
        model=ARTIFACTS[0],
        week_model=week_model,
    )
    filepath = f"{sub_path}/submission.csv"
    df_pred = freemem(df_pred)
    df_pred.write_csv(f"{filepath}")
    logging.info(f"save prediction submission to: {filepath}")
    logging.info(f"output df shape {df_pred.shape}")
    logging.info(f"make submission complete!")


def main():
    evaluate_ensemble()


if __name__ == "__main__":
    main()
