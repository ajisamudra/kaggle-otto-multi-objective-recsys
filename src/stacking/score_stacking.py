# step1: score input: model & test combined parquet | output -> rows & list (only take top 20)
# step2.a: make_submisssion input: scores click, cart, order | output -> concat all rows event
# step3: evaluate input: submission (all event)

# step2.b-2: optimize_ensemble_submisssion input: scores click, cart, order | intermiediate: add weight to model, create list per event |output -> concat all rows event
# step2.b-2: make_ensemble_submisssion input: scores click, cart, order | intermiediate: create list per event |output -> concat all rows event

import click
import polars as pl
from tqdm import tqdm
import numpy as np
import gc
import joblib
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_scoring_test_dataset_dir,
    get_processed_training_test_dataset_dir,
    get_data_output_local_submission_dir,  # scoring output dir
    get_data_output_submission_dir,
    get_data_output_local_ensemble_submission_dir,
    get_data_output_ensemble_submission_dir,
)

from src.utils.constants import get_artifacts_training_dir
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


def scoring_stacking(artifact: str, event: str, week_data: str, week_model: str):
    # for each event
    # read trained ensemble model
    # iterate 10 chunk of test data
    # for each N: read training data, predict, merge score to dataframe, save dataframe
    if week_data == "w2":
        N_test = CFG.N_local_test
    else:
        N_test = CFG.N_test

    event_model_name = "stacking"
    EVENT = event
    input_path = ""
    if week_data == "w2":
        input_path = get_data_output_local_ensemble_submission_dir(
            event=EVENT, model=event_model_name, week_model=week_model
        )

    elif week_data == "w1":
        input_path = get_data_output_ensemble_submission_dir(
            event=EVENT, model=event_model_name, week_model=week_model
        )

    artifact_path = get_artifacts_training_dir(event=f"{EVENT}_stacking", week="w2")
    model_path = f"{artifact_path}/{artifact}/model.pkl"
    model = joblib.load(model_path)
    logging.info(f"stacking model for event {EVENT.upper()} loaded!")

    logging.info("start scoring")
    ARTIFACTS = CFG_MODEL[f"{EVENT}_models"]
    for IX in tqdm(range(N_test)):
        # logging.info(f"read test data for chunk: {IX}")
        filepath = f"{input_path}/test_{IX}_{EVENT}_stacking_dataset.parquet"
        test_df = pl.read_parquet(filepath)

        # preprocess: fill_null with median for each cols
        for fea in ARTIFACTS:
            test_df = test_df.with_column(
                pl.col(fea).fill_null(pl.median(fea)),
            )

        # # preprocess: create pow2/4 as new features
        for fea in ARTIFACTS:
            test_df = test_df.with_columns(
                [
                    np.power(pl.col(fea), 4),
                ]
            )

        test_df = test_df.sort(by=["session", "candidate_aid"], reverse=[True, False])
        logging.info(f"input shape {test_df.shape}")

        # select features
        # X_test = test_df[selected_features].to_pandas()
        X_test = test_df[ARTIFACTS].to_pandas()

        # perform scoring
        # logging.info("perform scoring")
        scores = model.predict(X_test)
        # select only session & candidate_aid cols
        test_df = test_df.select([pl.col(["session", "candidate_aid", "label"])])
        # add scores columns
        # logging.info("merge with test_df")
        test_df = test_df.with_columns([pl.Series(name="score", values=scores)])

        del X_test
        gc.collect()

        # save to parquet
        if week_data == "w1":
            output_path = get_data_output_submission_dir(
                event=f"{EVENT}_stacking", model=artifact, week_model=week_model
            )
        elif week_data == "w2":
            output_path = get_data_output_local_submission_dir(
                event=f"{EVENT}_stacking", model=artifact, week_model=week_model
            )

        # filepath = f"{output_path}/test_{IX}_{EVENT}_scores.parquet"
        # test_df = freemem(test_df)
        # test_df = round_float_3decimals(test_df)
        # test_df.write_parquet(f"{filepath}")

        # logging.info(f"save prediction scores to: {filepath}")
        # logging.info(f"output df shape {test_df.shape}")

        # logging.info(f"rank & select top 20")
        # take top 20 candidate aid and save it as list
        test_predictions = (
            test_df.sort(["session", "score"], reverse=True)
            .groupby("session")
            .agg([pl.col("candidate_aid").limit(20).list().alias("labels")])
        )

        del test_df
        gc.collect()

        test_predictions = test_predictions.select([pl.col(["session", "labels"])])
        test_predictions = test_predictions.with_columns(
            [(pl.col(["session"]) + f"_{EVENT}").alias("session_type")]
        )
        test_predictions = test_predictions.select([pl.col(["session_type", "labels"])])
        test_predictions = test_predictions.with_columns(
            [pl.col("labels").apply(lambda x: " ".join(map(str, x))).alias("labels")]
        )

        filepath = f"{output_path}/test_{IX}_{EVENT}_submission.parquet"
        test_predictions = freemem(test_predictions)
        test_predictions.write_parquet(f"{filepath}")
        logging.info(f"save prediction submission to: {filepath}")
        logging.info(f"output df shape {test_predictions.shape}")

        del test_predictions
        gc.collect()

    logging.info(f"scoring complete for {EVENT.upper()}!")


@click.command()
@click.option(
    "--artifact",
    default="2022-12-05_catboost_46313_82997",
    help="artifact folder for reading model.pkl",
)
@click.option(
    "--event",
    default="orders",
    help="avaiable event: clicks/carts/orders/all",
)
@click.option(
    "--week_data",
    default="w2",
    help="subset of test data, w1/w2; w1:scoring dir, w2:training dir",
)
@click.option(
    "--week_model",
    default="w2",
    help="on which training data the model was trained, w1/w2; w1:scoring dir, w2:training dir",
)
def main(
    event: str = "all",
    artifact: str = "lgbm",
    week_data: str = "w2",
    week_model: str = "w2",
):
    if event not in ["clicks", "carts", "orders"]:
        raise ValueError("available event: clicks, carts, orders")

    scoring_stacking(
        artifact=artifact, event=event, week_data=week_data, week_model=week_model
    )


if __name__ == "__main__":
    main()
