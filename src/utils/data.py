import pandas as pd
import polars as pl
from src.utils.constants import get_covisitation_dir


DISK_PIECES = 4
VER = 6
covisit_dir = get_covisitation_dir()


def pqt_to_dict(df):
    return df.groupby("aid_x").aid_y.apply(list).to_dict()


def get_top20_covisitation_click():
    top_20_clicks = pqt_to_dict(
        pd.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_0.pqt")
    )
    for k in range(1, DISK_PIECES):
        top_20_clicks.update(
            pqt_to_dict(pd.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_{k}.pqt"))
        )

    return top_20_clicks


def get_top15_covisitation_buy2buy():
    top_15_buy2buy = pqt_to_dict(
        pd.read_parquet(covisit_dir / f"top_15_buy2buy_v{VER}_0.pqt")
    )
    return top_15_buy2buy


def get_top15_covisitation_buys():
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


def get_top20_covisitation_click_df():
    top_20_clicks = pl.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_0.pqt")
    for k in range(1, DISK_PIECES):
        df_chunk = pl.read_parquet(covisit_dir / f"top_20_clicks_v{VER}_{k}.pqt")
        top_20_clicks = pl.concat([top_20_clicks, df_chunk])

    top_20_clicks = top_20_clicks.select(["aid_x", "aid_y", "wgt"])
    return top_20_clicks


def get_top15_covisitation_buy2buy_df():
    top_15_buy2buy = pl.read_parquet(covisit_dir / f"top_15_buy2buy_v{VER}_0.pqt")
    top_15_buy2buy = top_15_buy2buy.select(["aid_x", "aid_y", "wgt"])
    return top_15_buy2buy


def get_top15_covisitation_buys_df():
    top_15_buys = pl.read_parquet(covisit_dir / f"top_15_carts_orders_v{VER}_0.pqt")
    for k in range(1, DISK_PIECES):
        df_chunk = pl.read_parquet(covisit_dir / f"top_15_carts_orders_v{VER}_{k}.pqt")
        top_15_buys = pl.concat([top_15_buys, df_chunk])

    top_15_buys = top_15_buys.select(["aid_x", "aid_y", "wgt"])
    return top_15_buys
