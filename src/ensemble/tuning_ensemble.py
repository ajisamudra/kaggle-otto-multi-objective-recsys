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
    get_artifacts_tuning_dir,
)

from src.metrics.submission_evaluation import measure_recall
from src.utils.memory import freemem, round_float_3decimals
from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"
NTRIAL = 15

CFG_MODEL = {
    "clicks_models": [
        "2022-12-25_clicks_cat_ranker_45439_89900",
        "2022-12-26_clicks_lgbm_ranker_48370_89600",
        "2023-01-02_clicks_cat_ranker_60593_91362",
        "2023-01-02_clicks_lgbm_ranker_61844_91502",
        # "2023-01-10_clicks_cat_ranker_61483_91316",
        "2023-01-13_clicks_cat_classifier_71560_93440",
    ],
    "carts_models": [
        "2022-12-25_carts_cat_ranker_65389_94260",
        "2022-12-26_carts_lgbm_ranker_66196_93867",
        "2023-01-02_carts_cat_ranker_75708_94697",
        "2023-01-02_carts_lgbm_ranker_75879_94504",
        # "2023-01-10_carts_cat_ranker_76221_94688",
        "2023-01-13_carts_cat_classifier_74699_93978",
    ],
    "orders_models": [
        "2022-12-25_orders_cat_ranker_80132_96912",
        "2022-12-26_orders_lgbm_ranker_78372_95814",
        "2023-01-02_orders_cat_ranker_86779_97309",
        "2023-01-02_orders_lgbm_ranker_85371_96813",
        # "2023-01-10_orders_cat_ranker_87315_97301",
        "2023-01-13_orders_cat_classifier_89862_97679",
    ],
}


def measure_ensemble_scores(hyperparams: dict):
    N_test = 40
    week_model = "w2"
    CONFIG = {
        "clicks_weights": [
            hyperparams["click_wgt_1"],
            hyperparams["click_wgt_2"],
            hyperparams["click_wgt_3"],
            hyperparams["click_wgt_4"],
            hyperparams["click_wgt_5"],
            # hyperparams["click_wgt_6"],
        ],
        "carts_weights": [
            hyperparams["cart_wgt_1"],
            hyperparams["cart_wgt_2"],
            hyperparams["cart_wgt_3"],
            hyperparams["cart_wgt_4"],
            hyperparams["cart_wgt_5"],
            # hyperparams["cart_wgt_6"],
        ],
        "orders_weights": [
            hyperparams["order_wgt_1"],
            hyperparams["order_wgt_2"],
            hyperparams["order_wgt_3"],
            hyperparams["order_wgt_4"],
            hyperparams["order_wgt_5"],
            # hyperparams["order_wgt_6"],
        ],
        "clicks_powers": [
            hyperparams["click_pow"],
            hyperparams["click_pow"],
            hyperparams["click_pow"],
            hyperparams["click_pow"],
            hyperparams["click_pow"],
            # hyperparams["click_pow"],
        ],
        "carts_powers": [
            hyperparams["cart_pow"],
            hyperparams["cart_pow"],
            hyperparams["cart_pow"],
            hyperparams["cart_pow"],
            hyperparams["cart_pow"],
            # hyperparams["cart_pow"],
        ],
        "orders_powers": [
            hyperparams["order_pow"],
            hyperparams["order_pow"],
            hyperparams["order_pow"],
            hyperparams["order_pow"],
            hyperparams["order_pow"],
            # hyperparams["order_pow"],
        ],
    }
    CONFIG.update(CFG_MODEL)
    events_model_name = []
    # pred rows 2212692
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
            output_path = get_data_output_local_ensemble_submission_dir(
                event=EVENT, model=event_model_name, week_model=week_model
            )

            tmp_path = f"{output_path}/test_{i}_{EVENT}_ensemble_scores.parquet"
            df_chunk = freemem(df_chunk)
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

    return recall20


class ObjectiveEnsemble:
    def __init__(self):
        self.N_test = 80

    def __call__(self, trial):
        hyperparams = {
            "click_wgt_1": trial.suggest_float("click_wgt_1", 0.01, 0.99),
            "cart_wgt_1": trial.suggest_float("cart_wgt_1", 0.01, 0.99),
            "order_wgt_1": trial.suggest_float("order_wgt_1", 0.01, 0.99),
            "click_wgt_2": trial.suggest_float("click_wgt_2", 0.01, 0.99),
            "cart_wgt_2": trial.suggest_float("cart_wgt_2", 0.01, 0.99),
            "order_wgt_2": trial.suggest_float("order_wgt_2", 0.01, 0.99),
            "click_wgt_3": trial.suggest_float("click_wgt_3", 0.01, 0.99),
            "cart_wgt_3": trial.suggest_float("cart_wgt_3", 0.01, 0.99),
            "order_wgt_3": trial.suggest_float("order_wgt_3", 0.01, 0.99),
            "click_wgt_4": trial.suggest_float("click_wgt_4", 0.01, 0.99),
            "cart_wgt_4": trial.suggest_float("cart_wgt_4", 0.01, 0.99),
            "order_wgt_4": trial.suggest_float("order_wgt_4", 0.01, 0.99),
            "click_wgt_5": trial.suggest_float("click_wgt_5", 0.01, 0.99),
            "cart_wgt_5": trial.suggest_float("cart_wgt_5", 0.01, 0.99),
            "order_wgt_5": trial.suggest_float("order_wgt_5", 0.01, 0.99),
        }

        # constraints sum of weights <= 1
        cclick = (
            hyperparams["click_wgt_1"]
            + hyperparams["click_wgt_2"]
            + hyperparams["click_wgt_3"]
            + hyperparams["click_wgt_4"]
            - 1
        )
        ccart = (
            hyperparams["cart_wgt_1"]
            + hyperparams["cart_wgt_2"]
            + hyperparams["cart_wgt_3"]
            + hyperparams["cart_wgt_4"]
            - 1
        )
        corder = (
            hyperparams["order_wgt_1"]
            + hyperparams["order_wgt_2"]
            + hyperparams["order_wgt_3"]
            + hyperparams["order_wgt_4"]
            - 1
        )

        # Store the constraints as user attributes so that they can be restored after optimization.
        trial.set_user_attr("constraint", (cclick, ccart, corder))

        wgts_2 = {
            "click_pow": 2,
            "cart_pow": 2,
            "order_pow": 2,
        }

        hyperparams.update(wgts_2)

        recall20 = measure_ensemble_scores(hyperparams)

        return recall20


def constraints(trial):
    return trial.user_attrs["constraint"]


def tune_ensemble():
    logging.info("measure ensemble recall@20 before tuning")
    hyperparams = {
        "click_wgt_1": 0.2,
        "click_wgt_2": 0.2,
        "click_wgt_3": 0.2,
        "click_wgt_4": 0.2,
        "click_wgt_5": 0.2,
        # "click_wgt_6": 0.2,
        "cart_wgt_1": 0.2,
        "cart_wgt_2": 0.2,
        "cart_wgt_3": 0.2,
        "cart_wgt_4": 0.2,
        "cart_wgt_5": 0.2,
        # "cart_wgt_6": 0.2,
        "order_wgt_1": 0.2,
        "order_wgt_2": 0.2,
        "order_wgt_3": 0.2,
        "order_wgt_4": 0.2,
        "order_wgt_5": 0.2,
        # "order_wgt_6": 0.2,
        "click_pow": 2,
        "cart_pow": 2,
        "order_pow": 2,
    }

    recall20_before = measure_ensemble_scores(hyperparams)
    logging.info(f"ensemble recall@20 before tuning: {recall20_before}")

    # logging.info("perform tuning")
    # study = optuna.create_study(direction="maximize", sampler=TPESampler())
    # study.optimize(ObjectiveEnsemble(), n_trials=NTRIAL)

    # logging.info("complete tuning!")
    # # find best hyperparams
    # recall20_after = study.best_value
    # best_hyperparams = study.best_params
    # logging.info(f"Previous value: {recall20_before}")
    # logging.info(f"Found best tuned value {recall20_after}")
    # logging.info(f"Best hyperparams {best_hyperparams}")
    # # if better, save to artifact dir
    # performance_metric = {
    #     "before": recall20_before,
    #     "after": recall20_after,
    # }
    # current_date = datetime.datetime.now().strftime("%Y-%m-%d")  # get current date
    # unique_model_name = f"{current_date}_ensemble_{recall20_before}_{recall20_after}"
    # artifact_path = get_artifacts_tuning_dir(event="ensemble", week="w2")
    # # create artifact dir
    # filepath = artifact_path / unique_model_name
    # check_directory(filepath)
    # logging.info(f"saving artifacts to: {filepath}")
    # hyper_path = f"{filepath}/tuned_hyperparams.json"
    # write_json(filepath=hyper_path, data=best_hyperparams)

    # CONFIG = {
    #     "clicks_weights": [
    #         best_hyperparams["click_wgt_1"],
    #         best_hyperparams["click_wgt_2"],
    #         best_hyperparams["click_wgt_3"],
    #         best_hyperparams["click_wgt_4"],
    #         best_hyperparams["click_wgt_5"],
    #     ],
    #     "carts_weights": [
    #         best_hyperparams["cart_wgt_1"],
    #         best_hyperparams["cart_wgt_2"],
    #         best_hyperparams["cart_wgt_3"],
    #         best_hyperparams["cart_wgt_4"],
    #         best_hyperparams["cart_wgt_5"],
    #     ],
    #     "orders_weights": [
    #         best_hyperparams["order_wgt_1"],
    #         best_hyperparams["order_wgt_2"],
    #         best_hyperparams["order_wgt_3"],
    #         best_hyperparams["order_wgt_4"],
    #         best_hyperparams["order_wgt_5"],
    #     ],
    #     "clicks_powers": [2, 2, 2, 2, 2],
    #     "carts_powers": [2, 2, 2, 2, 2],
    #     "orders_powers": [2, 2, 2, 2, 2],
    # }
    # CONFIG.update(CFG_MODEL)
    # complete_cfg = f"{filepath}/complete_cfg.json"
    # write_json(filepath=complete_cfg, data=CONFIG)

    # # output performance metrics
    # perf_path = f"{filepath}/performance_metric.json"
    # write_json(filepath=perf_path, data=performance_metric)

    # if recall20_after > recall20_before:
    #     logging.info(
    #         "Tuned value is higher than previous value! you should update hyperparams in training!"
    #     )
    #     logging.info(performance_metric)


def main():
    tune_ensemble()


if __name__ == "__main__":
    main()
