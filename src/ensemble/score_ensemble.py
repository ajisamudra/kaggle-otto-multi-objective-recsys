import click
import polars as pl
from tqdm import tqdm
import numpy as np
import gc
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
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
    # Overall Recall@20 = 0.5713838793150127
    CONFIG = {
        "carts_models": [
            "2022-12-25_carts_cat_ranker_65389_94260",  # cv 0.56873
            "2023-01-02_carts_cat_ranker_75708_94697",  # cv 0.57012
            "2023-01-02_carts_lgbm_ranker_75879_94504",  # cv 0.56920
            "2023-01-10_carts_cat_ranker_76221_94688",  # cv 0.56993
            "2023-01-17_carts_cat_classifier_72864_93779",  # cv 0.5690
            "2023-01-17_carts_cat_ranker_74566_93914",  # cv 0.56974
        ],
        "carts_powers": [2, 2, 2, 2, 2, 2],
        "carts_weights": [
            0.167,
            0.167,
            0.167,
            0.167,
            0.167,
            0.167,
        ],
        "clicks_models": [
            "2022-12-25_clicks_cat_ranker_45439_89900",  # cv 0.56873
            "2023-01-02_clicks_cat_ranker_60593_91362",  # cv 0.57012
            "2023-01-02_clicks_lgbm_ranker_61844_91502",  # cv 0.56920
            "2023-01-10_clicks_cat_ranker_61483_91316",  # cv 0.56993
            "2023-01-17_clicks_cat_classifier_69783_92833",  # cv 0.5690
            "2023-01-17_clicks_cat_ranker_71442_93554",  # cv 0.56974
        ],
        "clicks_powers": [2, 2, 2, 2, 2, 2],
        "clicks_weights": [
            0.167,
            0.167,
            0.167,
            0.167,
            0.167,
            0.167,
        ],
        "orders_models": [
            "2022-12-25_orders_cat_ranker_80132_96912",  # cv 0.56873
            "2023-01-02_orders_cat_ranker_86779_97309",  # cv 0.57012
            "2023-01-02_orders_lgbm_ranker_85371_96813",  # cv 0.56920
            "2023-01-10_orders_cat_ranker_87315_97301",  # cv 0.56993
            "2023-01-17_orders_cat_classifier_89226_97489",  # cv 0.5690
            "2023-01-17_orders_cat_ranker_88789_97503",  # cv 0.56974
        ],
        "orders_powers": [2, 2, 2, 2, 2, 2],
        "orders_weights": [
            0.167,
            0.167,
            0.167,
            0.167,
            0.167,
            0.167,
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
                input_path = get_data_output_submission_dir(
                    event=EVENT, model=ARTIFACTS[ix], week_model=week_model
                )
                tmp_path = f"{input_path}/test_{i}_{EVENT}_scores.parquet"
                df_tmp = pl.read_parquet(tmp_path)
                # min max scale
                scaler = MinMaxScaler()
                scaled_score = scaler.fit_transform(
                    df_tmp["score"].to_pandas().values.reshape(-1, 1)
                )
                # reshape the scaled score
                scaled_score = scaled_score.reshape(1, -1)[0]
                df_tmp = df_tmp.with_columns(
                    [pl.Series(name="scaled_score", values=scaled_score)]
                )
                # apply weights
                df_tmp = df_tmp.with_columns(
                    [(pow(pl.col("scaled_score"), POWERS[ix])) * WEIGHTS[ix]]
                )
                df_tmp = freemem(df_tmp)
                df_chunk = pl.concat([df_chunk, df_tmp])

                del df_tmp, scaler
                gc.collect()

            # sum weightes scores per candidate aid
            df_chunk = df_chunk.groupby(["session", "candidate_aid", "label"]).agg(
                [pl.col("scaled_score").sum()]
            )

            # save weighted scores
            event_model_name = "_".join(ARTIFACTS)
            output_path = get_data_output_ensemble_submission_dir(
                event=EVENT, model=event_model_name, week_model=week_model
            )

            # tmp_path = f"{output_path}/test_{i}_{EVENT}_ensemble_scores.parquet"
            # df_chunk = freemem(df_chunk)
            # df_chunk = round_float_3decimals(df_chunk)
            # df_chunk.write_parquet(tmp_path)

            # take top 20 candidate aid and save it as list
            test_predictions = (
                df_chunk.sort(["session", "scaled_score"], reverse=True)
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
