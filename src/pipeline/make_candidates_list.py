import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    CFG,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
    get_processed_training_train_candidates_dir,
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
)
from src.utils.data import (
    get_top15_covisitation_buys,
    get_top15_covisitation_buy2buy,
    get_top20_covisitation_click,
)
from src.retrieval.covisit_retrieval import suggest_buys, suggest_carts, suggest_clicks
from src.utils.logger import get_logger

logging = get_logger()


# read chunk parquet
# for each chunk -> suggest 40 candidates clicks, carts, buys
# save 40 candidates clickc, carts, buys in different files


def generate_candidates_covisitation(name: str, input_path: Path, output_path: Path):
    if name == "train":
        n = CFG.N_train
    else:
        n = CFG.N_test

    logging.info("read covisitation buys")
    top_15_buys = get_top15_covisitation_buys()
    logging.info("read covisitation buy2buy")
    top_15_buy2buy = get_top15_covisitation_buy2buy()
    logging.info("read covisitation click")
    top_20_clicks = get_top20_covisitation_click()

    # iterate over chunks
    logging.info(f"iterate {n} chunks")
    for ix in tqdm(range(n)):
        logging.info(f"chunk {ix}: read input")
        filepath = f"{input_path}/{name}_{ix}.parquet"
        df = pd.read_parquet(filepath)
        # input df as follow
        # session | aid | ts | type
        # A     | 1234  | 1  | 0
        # A     | 123   | 2  | 0
        # A     | 1234  | 3  | 1
        # logging.info("top clicks in dataset")
        top_clicks = df.loc[df["type"] == 0, "aid"].value_counts().index.values[:20]
        # logging.info("top carts in dataset")
        top_carts = df.loc[df["type"] == 1, "aid"].value_counts().index.values[:20]
        # logging.info("top orders in dataset")
        top_orders = df.loc[df["type"] == 2, "aid"].value_counts().index.values[:20]
        # logging.info("create ses2aids")
        ses2aids = df.groupby("session")["aid"].apply(list).to_dict()
        # logging.info("create ses2types")
        ses2types = df.groupby("session")["type"].apply(list).to_dict()

        logging.info("input type class proportion")
        logging.info(df["type"].value_counts(ascending=False))

        del df
        gc.collect()

        candidates_list = pd.Series()
        for event in ["clicks", "carts", "orders"]:
            logging.info(f"start of suggesting {event}")
            if event == "clicks":
                candidates_list = suggest_clicks(
                    n_candidate=40,
                    ses2aids=ses2aids,
                    ses2types=ses2types,
                    top_clicks=top_clicks,
                    covisit_click=top_20_clicks,
                )
            elif event == "carts":
                candidates_list = suggest_carts(
                    n_candidate=40,
                    ses2aids=ses2aids,
                    ses2types=ses2types,
                    top_carts=top_carts,
                    covisit_click=top_20_clicks,
                    covisit_buys=top_15_buys,
                )
            elif event == "orders":
                candidates_list = suggest_buys(
                    n_candidate=40,
                    ses2aids=ses2aids,
                    ses2types=ses2types,
                    top_orders=top_orders,
                    covisit_buy2buy=top_15_buy2buy,
                    covisit_buys=top_15_buys,
                )
            logging.info(f"end of suggesting {event}")

            logging.info("create candidates df")
            candidate_list_df = pd.DataFrame(
                candidates_list.add_suffix(f"_{event}"), columns=["labels"]
            ).reset_index()

            filepath = output_path / f"{name}_{ix}_{event}_list.parquet"
            logging.info(f"save chunk {ix} to: {filepath}")
            candidate_list_df.to_parquet(f"{filepath}")

            del candidate_list_df
            gc.collect()


@click.command()
@click.option(
    "--mode",
    help="avaiable mode: training_train/training_test/scoring_train/scoring_test",
)
def main(mode: str):
    if mode == "training_train":
        input_path = get_processed_training_train_splitted_dir()
        output_path = get_processed_training_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_covisitation(
            name="train", input_path=input_path, output_path=output_path
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_covisitation(
            name="test", input_path=input_path, output_path=output_path
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_covisitation(
            name="train", input_path=input_path, output_path=output_path
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_covisitation(
            name="test", input_path=input_path, output_path=output_path
        )


if __name__ == "__main__":
    main()
