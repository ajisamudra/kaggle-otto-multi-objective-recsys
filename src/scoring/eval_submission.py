# step1: score input: model & test combined parquet | output -> rows & list (only take top 20)
# step2.a: make_submisssion input: scores click, cart, order | output -> concat all rows event
# step3: evaluate input: submission (all event)

# step2.b-2: optimize_ensemble_submisssion input: scores click, cart, order | intermiediate: add weight to model, create list per event |output -> concat all rows event
# step2.b-2: make_ensemble_submisssion input: scores click, cart, order | intermiediate: create list per event |output -> concat all rows event

import click
import polars as pl
from tqdm import tqdm
from pathlib import Path
from src.utils.constants import (
    get_data_output_local_submission_dir,  # scoring output dir
    get_processed_local_validation_dir,
    ROOT_DIR,
)
from src.metrics.submission_evaluation import measure_recall


from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def eval_submission(
    click_model: str,
    cart_model: str,
    order_model: str,
    week_data: str,
    week_model: str,
):
    # read submission and measure recall@20
    models = [click_model, cart_model, order_model]
    output_path = ROOT_DIR
    model_name = "_".join(models)
    output_path = get_data_output_local_submission_dir(
        event="submission", model=model_name, week_model=week_model
    )

    filepath = f"{output_path}/submission.csv"
    logging.info(f"read prediction submission from: {filepath}")
    df_pred = pl.read_csv(f"{filepath}")
    logging.info(f"prediction df_pred shape {df_pred.shape}")

    logging.info("start computing metrics")
    # read ground truth
    ground_truth_path = get_processed_local_validation_dir()
    df_truth = pl.read_parquet(f"{ground_truth_path}/test_labels.parquet")
    logging.info(f"ground truth shape {df_truth.shape}")
    # compute metrics
    measure_recall(df_pred=df_pred.to_pandas(), df_truth=df_truth.to_pandas(), Ks=[20])


@click.command()
@click.option(
    "--click_model",
    help="click model for submission",
)
@click.option(
    "--cart_model",
    help="cart model for submission",
)
@click.option(
    "--order_model",
    help="order model for submission",
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
    click_model: str,
    cart_model: str,
    order_model: str,
    week_data: str = "w2",
    week_model: str = "w2",
):
    if week_data == "w1" or week_model == "w1":
        raise ValueError(
            "we cannot evaluate submission for model that's trained on w1 data"
        )

    eval_submission(
        click_model=click_model,
        cart_model=cart_model,
        order_model=order_model,
        week_data=week_data,
        week_model=week_model,
    )


if __name__ == "__main__":
    main()
