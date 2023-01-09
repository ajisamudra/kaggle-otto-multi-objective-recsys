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


def read_sub(
    path, weight=1
):  # by default let us assing the weight of 1 to predictions from each submission, this will be akin to a standard vote ensemble
    """a helper function for loading and preprocessing submissions"""
    return (
        pl.read_parquet(path)
        .with_column(pl.col("labels").str.split(by=" "))
        .with_column(pl.lit(weight).alias("vote"))
        .explode("labels")
        .rename({"labels": "aid"})
        .with_column(
            pl.col("aid").cast(pl.UInt32)
        )  # we are casting the `aids` to `Int32`! memory management is super important to ensure we don't run out of resources
        .with_column(pl.col("vote").cast(pl.UInt8))
    )


def evaluate_ensemble():
    N_test = CFG.N_local_test
    week_model = "w2"
    CONFIG = {
        "carts_models": [
            "2023-01-02_carts_cat_ranker_75708_94697",
            "2023-01-02_carts_lgbm_ranker_75879_94504",
        ],
        "carts_weights": [
            1,
            1,
        ],
        "clicks_models": [
            "2023-01-02_clicks_cat_ranker_60593_91362",
            "2023-01-02_clicks_lgbm_ranker_61844_91502",
        ],
        "clicks_weights": [
            1,
            1,
        ],
        "orders_models": [
            "2023-01-02_orders_cat_ranker_86779_97309",
            "2023-01-02_orders_lgbm_ranker_85371_96813",
        ],
        "orders_weights": [
            1,
            1,
        ],
    }

    events_model_name = []
    pred_rows = 0
    for EVENT in ["clicks", "carts", "orders"]:
        logging.info(f"start applying weights in event: {EVENT.upper()}")
        ARTIFACTS = CONFIG[f"{EVENT}_models"]
        WEIGHTS = CONFIG[f"{EVENT}_weights"]
        event_model_name = ""  # for accessing submission later
        for i in tqdm(range(N_test)):
            df_chunk = pl.DataFrame()
            for ix in range(len(ARTIFACTS)):
                input_path = get_data_output_local_submission_dir(
                    event=EVENT, model=ARTIFACTS[ix], week_model=week_model
                )
                tmp_path = f"{input_path}/test_{i}_{EVENT}_submission.parquet"
                df_tmp = read_sub(path=tmp_path, weight=WEIGHTS[ix])
                # df_tmp = pl.read_parquet(tmp_path)
                # # apply weights
                # df_tmp = df_tmp.with_columns(
                #     [(pow(pl.col("score"), POWERS[ix])) * WEIGHTS[ix]]
                # )
                df_chunk = pl.concat([df_chunk, df_tmp])

                del df_tmp
                gc.collect()

            # sum weightes scores per candidate aid
            df_chunk = df_chunk.groupby(["session_type", "aid"]).agg(
                [pl.col("vote").sum()]
            )

            # save weighted scores
            event_model_name = "_".join(ARTIFACTS)
            output_path = get_data_output_local_ensemble_submission_dir(
                event=EVENT, model=event_model_name, week_model=week_model
            )

            # tmp_path = f"{output_path}/test_{i}_{EVENT}_ensemble_scores.parquet"
            # df_chunk = freemem(df_chunk)
            # df_chunk = round_float_3decimals(df_chunk)
            # df_chunk.write_parquet(tmp_path)

            # take top 20 candidate aid and save it as list
            test_predictions = (
                df_chunk.sort(["session_type", "vote"], reverse=True)
                .groupby("session_type")
                .agg([pl.col("aid").limit(20).list().alias("labels")])
            )

            del df_chunk
            gc.collect()

            test_predictions = test_predictions.select(
                [pl.col(["session_type", "labels"])]
            )
            # test_predictions = test_predictions.with_columns(
            #     [(pl.col(["session"]) + f"_{EVENT}").alias("session_type")]
            # )
            # test_predictions = test_predictions.select(
            #     [pl.col(["session_type", "labels"])]
            # )
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
