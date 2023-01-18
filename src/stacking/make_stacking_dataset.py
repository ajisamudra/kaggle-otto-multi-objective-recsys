import click
import polars as pl
from tqdm import tqdm
import numpy as np
import gc
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import datetime
import optuna
from optuna.samplers import TPESampler
from src.utils.constants import (
    CFG,
    write_json,
    check_directory,
    get_processed_local_validation_dir,
    get_data_output_local_submission_dir,  # scoring output dir
    get_data_output_local_ensemble_submission_dir,
    get_data_output_submission_dir,
    get_data_output_ensemble_submission_dir,
    get_artifacts_tuning_dir,
)

from src.utils.constants import get_artifacts_training_dir, check_directory

from src.metrics.submission_evaluation import measure_recall
from src.utils.memory import freemem, round_float_3decimals
from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"

CFG_MODEL = {
    "clicks_models": [
        "2022-12-25_clicks_cat_ranker_45439_89900",  # cv 0.56873
        # "2022-12-26_clicks_lgbm_ranker_48370_89600",  # cv 0.567242
        "2023-01-02_clicks_cat_ranker_60593_91362",  # cv 0.57012
        "2023-01-02_clicks_lgbm_ranker_61844_91502",  # cv 0.56920
        "2023-01-10_clicks_cat_ranker_61483_91316",  # cv 0.56993
        # "2023-01-17_clicks_cat_regressor_69418_93202",  # cv 0.5696
        "2023-01-17_clicks_cat_classifier_69783_92833",  # cv 0.5690
        # "2023-01-17_clicks_cat_ranker_71442_93554",  # cv 0.56974
    ],
    "carts_models": [
        "2022-12-25_carts_cat_ranker_65389_94260",  # cv 0.56873
        # "2022-12-26_carts_lgbm_ranker_66196_93867",  # cv 0.567242
        "2023-01-02_carts_cat_ranker_75708_94697",  # cv 0.57012
        "2023-01-02_carts_lgbm_ranker_75879_94504",  # cv 0.56920
        "2023-01-10_carts_cat_ranker_76221_94688",  # cv 0.56993
        # "2023-01-17_carts_cat_regressor_70657_93800",  # cv 0.5696
        "2023-01-17_carts_cat_classifier_72864_93779",  # cv 0.5690
        # "2023-01-17_carts_cat_ranker_74566_93914",  # cv 0.56974
    ],
    "orders_models": [
        "2022-12-25_orders_cat_ranker_80132_96912",  # cv 0.56873
        # "2022-12-26_orders_lgbm_ranker_78372_95814",  # cv 0.567242
        "2023-01-02_orders_cat_ranker_86779_97309",  # cv 0.57012
        "2023-01-02_orders_lgbm_ranker_85371_96813",  # cv 0.56920
        "2023-01-10_orders_cat_ranker_87315_97301",  # cv 0.56993
        # "2023-01-17_orders_cat_regressor_86921_97516",  # cv 0.5696
        "2023-01-17_orders_cat_classifier_89226_97489",  # cv 0.5690
        # "2023-01-17_orders_cat_ranker_88789_97503",  # cv 0.56974
    ],
}


def make_stacking_dataset(mode: str):
    input_path = ""
    output_path = ""
    if mode == "training_test":
        N_test = CFG.N_local_test
    else:
        N_test = CFG.N_test

    week_model = "w2"
    for EVENT in ["clicks", "carts", "orders"]:
        logging.info(f"start creating stacking dataset in event: {EVENT.upper()}")
        ARTIFACTS = CFG_MODEL[f"{EVENT}_models"]
        event_model_name = ""  # for accessing submission later
        for i in tqdm(range(N_test)):
            df_chunk = pl.DataFrame()
            for ix in range(len(ARTIFACTS)):
                if mode == "training_test":
                    input_path = get_data_output_local_submission_dir(
                        event=EVENT, model=ARTIFACTS[ix], week_model=week_model
                    )

                elif mode == "scoring_test":
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
                    [pl.Series(name=ARTIFACTS[ix], values=scaled_score)]
                )

                # select columns
                df_tmp = df_tmp.select(
                    ["session", "candidate_aid", "label", ARTIFACTS[ix]]
                )
                df_tmp = df_tmp.with_columns([pl.col("label").alias(f"label_{ix}")])
                df_tmp = df_tmp.select(
                    ["session", "candidate_aid", f"label_{ix}", ARTIFACTS[ix]]
                )
                df_tmp = freemem(df_tmp)
                if ix == 0:
                    df_chunk = df_tmp

                    del scaler
                    gc.collect()

                else:
                    df_chunk = df_chunk.join(
                        df_tmp, how="outer", on=["session", "candidate_aid"]
                    )

                    del df_tmp, scaler
                    gc.collect()

            # create 1 label and only select the final label
            df_chunk = df_chunk.with_columns(
                [
                    pl.max(
                        [
                            pl.col("label_0"),
                            pl.col("label_1"),
                            pl.col("label_2"),
                            pl.col("label_3"),
                            pl.col("label_4"),
                            # pl.col("label_5"),
                            # pl.col("label_6"),
                            # pl.col("label_7"),
                        ]
                    ).alias("label")
                ]
            )

            # drop cols
            df_chunk = df_chunk.drop(
                columns=[
                    "label_0",
                    "label_1",
                    "label_2",
                    "label_3",
                    "label_4",
                    # "label_5",
                    # "label_6",
                    # "label_7",
                ]
            )

            # save stacking dataset
            event_model_name = "stacking"
            if mode == "training_test":
                output_path = get_data_output_local_ensemble_submission_dir(
                    event=EVENT, model=event_model_name, week_model=week_model
                )

            elif mode == "scoring_test":
                output_path = get_data_output_ensemble_submission_dir(
                    event=EVENT, model=event_model_name, week_model=week_model
                )

            tmp_path = f"{output_path}/test_{i}_{EVENT}_stacking_dataset.parquet"
            df_chunk = freemem(df_chunk)
            # df_chunk = round_float_3decimals(df_chunk)
            logging.info(f"save output to: {tmp_path}")
            df_chunk.write_parquet(tmp_path)

            del df_chunk
            gc.collect()


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    make_stacking_dataset(mode=mode)


if __name__ == "__main__":
    main()
