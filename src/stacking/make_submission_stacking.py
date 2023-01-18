# step1: score input: model & test combined parquet | output -> rows & list (only take top 20)
# step2.a: make_submisssion input: scores click, cart, order | output -> concat all rows event
# step3: evaluate input: submission (all event)

# step2.b-2: optimize_ensemble_submisssion input: scores click, cart, order | intermiediate: add weight to model, create list per event |output -> concat all rows event
# step2.b-2: make_ensemble_submisssion input: scores click, cart, order | intermiediate: create list per event |output -> concat all rows event

import click
import polars as pl
import gc
from tqdm import tqdm
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_data_output_local_submission_dir,  # scoring output dir
    get_data_output_submission_dir,
    ROOT_DIR,
)
from src.utils.memory import freemem

from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def make_submission(
    click_model: str,
    cart_model: str,
    order_model: str,
    week_data: str,
    week_model: str,
):
    # for each event
    # read trained ensemble model
    # iterate 10 chunk of test data
    # for each N: read training data, predict, merge score to dataframe, save dataframe
    if week_data == "w2":
        N_test = CFG.N_local_test
    else:
        N_test = CFG.N_test
    models = [click_model, cart_model, order_model]
    events = ["clicks", "carts", "orders"]
    df = pl.DataFrame()
    for ix, EVENT in enumerate(events):
        logging.info(f"start reading submission for event: {EVENT.upper()}")

        input_path: Path
        if week_data == "w1":
            input_path = get_data_output_submission_dir(
                event=f"{EVENT}_stacking", model=models[ix], week_model=week_model
            )
        elif week_data == "w2":
            input_path = get_data_output_local_submission_dir(
                event=f"{EVENT}_stacking", model=models[ix], week_model=week_model
            )
        else:
            raise NotImplementedError("week not implemented! (w1/w2)")

        if week_data == "w2":
            for i in tqdm(range(1, N_test)):
                chunk_path = f"{input_path}/test_{i}_{EVENT}_submission.parquet"
                df_chunk = pl.read_parquet(chunk_path)
                df = pl.concat([df, df_chunk])

                del df_chunk
                gc.collect()
        else:
            for i in tqdm(range(N_test)):
                chunk_path = f"{input_path}/test_{i}_{EVENT}_submission.parquet"
                df_chunk = pl.read_parquet(chunk_path)
                df = pl.concat([df, df_chunk])

                del df_chunk
                gc.collect()

        logging.info(f"submission shape: {df.shape}")

    output_path = ROOT_DIR
    model_name = "_".join(models)
    if week_data == "w1":
        output_path = get_data_output_submission_dir(
            event="submission", model=model_name, week_model=week_model
        )
    elif week_data == "w2":
        output_path = get_data_output_local_submission_dir(
            event="submission", model=model_name, week_model=week_model
        )
    filepath = f"{output_path}/submission.csv"
    df = freemem(df)
    df.write_csv(f"{filepath}")
    logging.info(f"save prediction submission to: {filepath}")
    logging.info(f"output df shape {df.shape}")
    logging.info(f"make submission complete!")


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
    make_submission(
        click_model=click_model,
        cart_model=cart_model,
        order_model=order_model,
        week_data=week_data,
        week_model=week_model,
    )


if __name__ == "__main__":
    main()
