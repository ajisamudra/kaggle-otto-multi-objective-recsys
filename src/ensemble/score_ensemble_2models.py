import click
import polars as pl
from tqdm import tqdm
import numpy as np
import gc
import joblib
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_data_output_submission_dir,  # scoring output dir
    get_data_output_ensemble_submission_dir,
)
from src.utils.memory import freemem, round_float_3decimals
from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def score_ensemble():
    N_test = 100
    week_model = "w2"
    CONFIG = {
        "carts_models": [
            "2023-01-02_carts_cat_ranker_75502_94516",
            "2023-01-02_carts_lgbm_ranker_75627_94287",
        ],
        "carts_powers": [1, 1],
        "carts_weights": [0.13914664356159773, 0.8608533564384022],
        "clicks_models": [
            "2023-01-02_clicks_cat_ranker_60409_91085",
            "2023-01-02_clicks_lgbm_ranker_61486_91133",
        ],
        "clicks_powers": [1, 1],
        "clicks_weights": [0.7648897303135639, 0.2351102696864361],
        "orders_models": [
            "2023-01-02_orders_cat_ranker_86674_97221",
            "2023-01-02_orders_lgbm_ranker_85360_96765",
        ],
        "orders_powers": [1, 1],
        "orders_weights": [0.2687816835097516, 0.7312183164902484],
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
                input_path = get_data_output_submission_dir(
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
            output_path = get_data_output_ensemble_submission_dir(
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

            output_path = get_data_output_ensemble_submission_dir(
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
        sub_path = get_data_output_ensemble_submission_dir(
            event="submission", model=events_model_name[ix], week_model=week_model
        )
        for i in tqdm(range(N_test)):
            tmp_path = f"{sub_path}/test_{i}_{EVENT}_submission.parquet"
            df_chunk = pl.read_parquet(f"{tmp_path}")
            df_pred = pl.concat([df_pred, df_chunk])

            del df_chunk
            gc.collect()

    # save final submission to csv
    logging.info("save submission to csv")
    sub_path = get_data_output_ensemble_submission_dir(
        event="final_submission", model=events_model_name[ix], week_model=week_model
    )
    filepath = f"{sub_path}/submission.csv"
    df_pred = freemem(df_pred)
    df_pred.write_csv(f"{filepath}")
    logging.info(f"save prediction submission to: {filepath}")
    logging.info(f"output df shape {df_pred.shape}")
    logging.info(f"make submission complete!")


def main():
    score_ensemble()


if __name__ == "__main__":
    main()
