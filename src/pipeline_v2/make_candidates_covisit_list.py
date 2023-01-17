import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
import itertools
from collections import Counter
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
from src.utils.logger import get_logger

logging = get_logger()


def suggest_clicks(
    n_past_aids_cand: int,
    n_covisit_cand: int,
    ses2aids: dict,
    ses2types: dict,
    covisit_click: dict,
):
    """
    covisit_click is dict of aid as key and list of suggested aid as value
    """

    sessions = []
    candidates = []
    ranks_list = []
    for session, aids in tqdm(ses2aids.items()):
        # unique_aids = set(aids)
        unique_aids = list(dict.fromkeys(aids[::-1]))
        types = ses2types[session]

        # past aids
        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) > 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * CFG.type_weight_multipliers[t]
            past_aid_candidate = [k for k, v in aids_temp.most_common(n_past_aids_cand)]

        else:
            past_aid_candidate = list(unique_aids)[:n_past_aids_cand]

        # covisit candidate
        # USE "CLICKS" CO-VISITATION MATRIX
        aids2 = list(
            itertools.chain(
                *[covisit_click[aid] for aid in unique_aids if aid in covisit_click]
            )
        )
        # RERANK CANDIDATES
        top_aids2 = [
            aid2
            for aid2, cnt in Counter(aids2).most_common(n_covisit_cand)
            if aid2 not in unique_aids
        ]
        covisit_candidate = top_aids2[:n_covisit_cand]

        # final candidates
        candidate = past_aid_candidate + covisit_candidate

        # append to list result
        rank_list = [i for i in range(len(candidate))]
        sessions.append(session)
        candidates.append(candidate)
        ranks_list.append(rank_list)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"

    return result_series, ranks_list


def suggest_carts(
    n_past_aids_cand: int,
    n_covisit_cand: int,
    ses2aids: dict,
    ses2types: dict,
    covisit_click: dict,
    covisit_buys: dict,
):
    """
    covisit_click is dict of aid as key and list of suggested aid as value
    """

    sessions = []
    candidates = []
    ranks_list = []
    for session, aids in tqdm(ses2aids.items()):
        unique_buys = []
        # unique_aids = set(aids)
        unique_aids = list(dict.fromkeys(aids[::-1]))
        types = ses2types[session]
        for ix, aid in enumerate(aids):
            curr_type = types[ix]
            if (curr_type == 0) or (curr_type == 1):
                unique_buys.append(aid)

        # reverse the order
        unique_buys = list(dict.fromkeys(unique_buys[::-1]))

        # past aids
        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) > 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * CFG.type_weight_multipliers[t]
            past_aid_candidate = [k for k, v in aids_temp.most_common(n_past_aids_cand)]

        else:
            past_aid_candidate = list(unique_aids)[:n_past_aids_cand]

        # covisit candidates
        if len(unique_aids) >= 20:
            # using covisit matrix
            aids1 = list(
                itertools.chain(
                    *[covisit_click[aid] for aid in unique_aids if aid in covisit_click]
                )
            )

            # Rerank candidates using"top_20_carts" co-visitation matrix
            aids2 = list(
                itertools.chain(
                    *[covisit_buys[aid] for aid in unique_buys if aid in covisit_buys]
                )
            )

            # RERANK CANDIDATES
            top_aids2 = [
                aid2
                for aid2, cnt in Counter(aids1 + aids2).most_common(n_covisit_cand)
                if aid2 not in unique_aids
            ]
            covisit_candidate = top_aids2[:n_covisit_cand]

        else:
            # Use "cart order" and "clicks" co-visitation matrices
            aids1 = list(
                itertools.chain(
                    *[covisit_click[aid] for aid in unique_aids if aid in covisit_click]
                )
            )
            aids2 = list(
                itertools.chain(
                    *[covisit_buys[aid] for aid in unique_aids if aid in covisit_buys]
                )
            )

            # RERANK CANDIDATES
            top_aids2 = [
                aid2
                for aid2, cnt in Counter(aids1 + aids2).most_common(n_covisit_cand)
                if aid2 not in unique_aids
            ]
            covisit_candidate = top_aids2[:n_covisit_cand]

        # final candidates
        candidate = past_aid_candidate + covisit_candidate

        # append to list result
        rank_list = [i for i in range(len(candidate))]
        sessions.append(session)
        candidates.append(candidate)
        ranks_list.append(rank_list)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"

    return result_series, ranks_list


def suggest_buys(
    n_past_aids_cand: int,
    n_covisit_cand: int,
    ses2aids: dict,
    ses2types: dict,
    covisit_buys: dict,
    covisit_buy2buy: dict,
):
    """
    covisit_click is dict of aid as key and list of suggested aid as value
    """

    sessions = []
    candidates = []
    ranks_list = []
    for session, aids in tqdm(ses2aids.items()):
        unique_buys = []
        # unique_aids = set(aids)
        unique_aids = list(dict.fromkeys(aids[::-1]))
        types = ses2types[session]
        for ix, aid in enumerate(aids):
            curr_type = types[ix]
            if (curr_type == 1) or (curr_type == 2):
                unique_buys.append(aid)

        # reverse the order
        unique_buys = list(dict.fromkeys(unique_buys[::-1]))

        # past aids
        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) > 20:
            weights = np.logspace(0.1, 1, len(aids), base=2, endpoint=True) - 1
            aids_temp = Counter()
            # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
            for aid, w, t in zip(aids, weights, types):
                aids_temp[aid] += w * CFG.type_weight_multipliers[t]
            past_aid_candidate = [k for k, v in aids_temp.most_common(n_past_aids_cand)]

        else:
            past_aid_candidate = list(unique_aids)[:n_past_aids_cand]

        # covisit candidate
        # RERANK CANDIDATES USING WEIGHTS
        if len(unique_aids) >= 20:
            # USE "CART ORDER" CO-VISITATION MATRIX
            aids2 = list(
                itertools.chain(
                    *[covisit_buys[aid] for aid in unique_aids if aid in covisit_buys]
                )
            )

            # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
            aids3 = list(
                itertools.chain(
                    *[
                        covisit_buy2buy[aid]
                        for aid in unique_buys
                        if aid in covisit_buy2buy
                    ]
                )
            )

            # RERANK CANDIDATES
            top_aids2 = [
                aid2
                for aid2, cnt in Counter(aids2 + aids3).most_common(n_covisit_cand)
                if aid2 not in unique_aids
            ]
            covisit_candidate = top_aids2[:n_covisit_cand]

        else:
            # USE "CART ORDER" CO-VISITATION MATRIX
            aids2 = list(
                itertools.chain(
                    *[covisit_buys[aid] for aid in unique_aids if aid in covisit_buys]
                )
            )
            # USE "BUY2BUY" CO-VISITATION MATRIX
            aids3 = list(
                itertools.chain(
                    *[
                        covisit_buy2buy[aid]
                        for aid in unique_buys
                        if aid in covisit_buy2buy
                    ]
                )
            )
            # RERANK CANDIDATES
            top_aids2 = [
                aid2
                for aid2, cnt in Counter(aids2 + aids3).most_common(n_covisit_cand)
                if aid2 not in unique_aids
            ]
            covisit_candidate = top_aids2[:n_covisit_cand]

        # final candidates
        candidate = past_aid_candidate + covisit_candidate

        # append to list result
        rank_list = [i for i in range(len(candidate))]
        sessions.append(session)
        candidates.append(candidate)
        ranks_list.append(rank_list)

    # output series
    result_series = pd.Series(candidates, index=sessions)
    result_series.index.name = "session"

    return result_series, ranks_list


def generate_candidates_covisitation(
    name: str,
    mode: str,
    input_path: Path,
    output_path: Path,
    top_15_buys: dict,
    top_15_buy2buy: dict,
    top_20_clicks: dict,
):
    if mode == "training_train":
        n = CFG.N_train
    elif mode == "training_test":
        n = CFG.N_local_test
    else:
        n = CFG.N_test

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
        # logging.info("create ses2aids")
        ses2aids = df.groupby("session")["aid"].apply(list).to_dict()
        # logging.info("create ses2types")
        ses2types = df.groupby("session")["type"].apply(list).to_dict()

        logging.info("input type class proportion")
        logging.info(df["type"].value_counts(ascending=False))

        del df
        gc.collect()

        candidates_list = pd.Series()
        ranks_list = []
        for event in ["clicks", "carts", "orders"]:

            if (mode == "training_train") & (event == "clicks") & (ix > 6):
                logging.info("click ix > 6 continue")
                continue

            logging.info(f"start of suggesting {event}")
            if event == "clicks":
                candidates_list, ranks_list = suggest_clicks(
                    n_past_aids_cand=CFG.past_candidates,
                    n_covisit_cand=CFG.covisit_candidates,
                    ses2aids=ses2aids,
                    ses2types=ses2types,
                    covisit_click=top_20_clicks,
                )
            elif event == "carts":
                candidates_list, ranks_list = suggest_carts(
                    n_past_aids_cand=CFG.past_candidates,
                    n_covisit_cand=CFG.covisit_candidates,
                    ses2aids=ses2aids,
                    ses2types=ses2types,
                    covisit_click=top_20_clicks,
                    covisit_buys=top_15_buys,
                )
            elif event == "orders":
                candidates_list, ranks_list = suggest_buys(
                    n_past_aids_cand=CFG.past_candidates,
                    n_covisit_cand=CFG.covisit_candidates,
                    ses2aids=ses2aids,
                    ses2types=ses2types,
                    covisit_buy2buy=top_15_buy2buy,
                    covisit_buys=top_15_buys,
                )
            logging.info(f"end of suggesting {event}")

            logging.info("create candidates df")
            candidate_list_df = pd.DataFrame(
                candidates_list.add_suffix(f"_{event}"), columns=["labels"]
            ).reset_index()
            candidate_list_df["ranks"] = ranks_list

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

    if mode in ["training_train", "training_test"]:
        logging.info("read local covisitation buys")
        top_15_buys = get_top15_covisitation_buys()
        logging.info("read local covisitation buy2buy")
        top_15_buy2buy = get_top15_covisitation_buy2buy()
        logging.info("read local covisitation click")
        top_20_clicks = get_top20_covisitation_click()
    else:
        logging.info("read scoring covisitation buys")
        top_15_buys = get_top15_covisitation_buys(mode="scoring")
        logging.info("read scoring covisitation buy2buy")
        top_15_buy2buy = get_top15_covisitation_buy2buy(mode="scoring")
        logging.info("read scoring covisitation click")
        top_20_clicks = get_top20_covisitation_click(mode="scoring")

    if mode == "training_train":
        input_path = get_processed_training_train_splitted_dir()
        output_path = get_processed_training_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_covisitation(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            top_15_buys=top_15_buys,
            top_15_buy2buy=top_15_buy2buy,
            top_20_clicks=top_20_clicks,
        )

    elif mode == "training_test":
        input_path = get_processed_training_test_splitted_dir()
        output_path = get_processed_training_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_covisitation(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            top_15_buys=top_15_buys,
            top_15_buy2buy=top_15_buy2buy,
            top_20_clicks=top_20_clicks,
        )

    elif mode == "scoring_train":
        input_path = get_processed_scoring_train_splitted_dir()
        output_path = get_processed_scoring_train_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_covisitation(
            name="train",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            top_15_buys=top_15_buys,
            top_15_buy2buy=top_15_buy2buy,
            top_20_clicks=top_20_clicks,
        )

    elif mode == "scoring_test":
        input_path = get_processed_scoring_test_splitted_dir()
        output_path = get_processed_scoring_test_candidates_dir()
        logging.info(f"read input data from: {input_path}")
        logging.info(f"will save chunks data to: {output_path}")
        generate_candidates_covisitation(
            name="test",
            mode=mode,
            input_path=input_path,
            output_path=output_path,
            top_15_buys=top_15_buys,
            top_15_buy2buy=top_15_buy2buy,
            top_20_clicks=top_20_clicks,
        )


if __name__ == "__main__":
    main()
