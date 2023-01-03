import pandas as pd
import polars as pl
import joblib
from src.utils.constants import get_local_covisitation_dir, get_scoring_covisitation_dir
from src.utils.memory import freemem

DISK_PIECES = 4
VER = 10


def pqt_to_dict(df):
    return df.groupby("aid_x").aid_y.apply(list).to_dict()


def pqt_to_pair_dict(df):
    df["pair_aid"] = df["aid_x"].astype(str) + "_" + df["aid_y"].astype(str)
    dict_obj = df.groupby("pair_aid")["wgt"].apply(float).to_dict()
    return dict_obj


def convert_top20_covisitation_click(mode: str = "local"):

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    top_20_clicks = pqt_to_pair_dict(
        pd.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_0.pqt")
    )
    for k in range(1, DISK_PIECES):
        top_20_clicks.update(
            pqt_to_pair_dict(
                pd.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_{k}.pqt")
            )
        )

    filepath = f"{covisit_dir}/top_20_clicks_v{VER}.pkl"
    joblib.dump(top_20_clicks, filepath)


def convert_top15_covisitation_buy2buy(mode: str = "local"):

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    top_15_buy2buy = pqt_to_pair_dict(
        pd.read_parquet(covisit_dir / f"top_15_buy2buy_v{VER}_0.pqt")
    )

    filepath = f"{covisit_dir}/top_15_buy2buy_v{VER}.pkl"
    joblib.dump(top_15_buy2buy, filepath)


def convert_top15_covisitation_buys(mode: str = "local"):

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    top_15_buys = pqt_to_pair_dict(
        pd.read_parquet(covisit_dir / f"top_15_carts_orders_v{VER}_0.pqt")
    )
    for k in range(1, DISK_PIECES):
        top_15_buys.update(
            pqt_to_pair_dict(
                pd.read_parquet(covisit_dir / f"top_15_carts_orders_v{VER}_{k}.pqt")
            )
        )

    filepath = f"{covisit_dir}/top_15_carts_orders_v{VER}.pkl"
    joblib.dump(top_15_buys, filepath)


def get_top20_covisitation_click(mode: str = "local"):

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    top_20_clicks = pqt_to_dict(
        pd.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_0.pqt")
    )
    for k in range(1, DISK_PIECES):
        top_20_clicks.update(
            pqt_to_dict(pd.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_{k}.pqt"))
        )

    return top_20_clicks


def get_top15_covisitation_buy2buy(mode: str = "local"):

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    top_15_buy2buy = pqt_to_dict(
        pd.read_parquet(covisit_dir / f"top_15_buy2buy_v{VER}_0.pqt")
    )
    return top_15_buy2buy


def get_top15_covisitation_buys(mode: str = "local"):

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    top_15_buys = pqt_to_dict(
        pd.read_parquet(covisit_dir / f"top_15_carts_orders_v{VER}_0.pqt")
    )
    for k in range(1, DISK_PIECES):
        top_15_buys.update(
            pqt_to_dict(
                pd.read_parquet(covisit_dir / f"top_15_carts_orders_v{VER}_{k}.pqt")
            )
        )

    return top_15_buys


def get_top20_covisitation_click_df(mode: str = "local"):

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    top_20_clicks = pl.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_0.pqt")
    for k in range(1, DISK_PIECES):
        df_chunk = pl.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_{k}.pqt")
        top_20_clicks = pl.concat([top_20_clicks, df_chunk])

    top_20_clicks = top_20_clicks.select(["aid_x", "aid_y", "wgt"])
    top_20_clicks = freemem(top_20_clicks)
    return top_20_clicks


def get_top15_covisitation_buy2buy_df(mode: str = "local"):

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    top_15_buy2buy = pl.read_parquet(covisit_dir / f"top_15_buy2buy_v{VER}_0.pqt")
    top_15_buy2buy = top_15_buy2buy.select(["aid_x", "aid_y", "wgt"])
    top_15_buy2buy = freemem(top_15_buy2buy)
    return top_15_buy2buy


def get_top15_covisitation_buys_df(mode: str = "local"):

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    top_15_buys = pl.read_parquet(covisit_dir / f"top_15_carts_orders_v{VER}_0.pqt")
    for k in range(1, DISK_PIECES):
        df_chunk = pl.read_parquet(covisit_dir / f"top_15_carts_orders_v{VER}_{k}.pqt")
        top_15_buys = pl.concat([top_15_buys, df_chunk])

    top_15_buys = top_15_buys.select(["aid_x", "aid_y", "wgt"])
    top_15_buys = freemem(top_15_buys)
    return top_15_buys


def get_top20_covisitation_click_dict(mode: str = "local") -> dict:

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    filepath = f"{covisit_dir}/top_20_clicks_v{VER}.pkl"
    dict_obj = joblib.load(filepath)

    return dict_obj


def get_top15_covisitation_buy2buy_dict(mode: str = "local") -> dict:

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    filepath = f"{covisit_dir}/top_15_buy2buy_v{VER}.pkl"
    dict_obj = joblib.load(filepath)

    return dict_obj


def get_top15_covisitation_buys_dict(mode: str = "local") -> dict:

    if mode == "local":
        covisit_dir = get_local_covisitation_dir()
    else:
        covisit_dir = get_scoring_covisitation_dir()

    filepath = f"{covisit_dir}/top_15_carts_orders_v{VER}.pkl"
    dict_obj = joblib.load(filepath)

    return dict_obj
