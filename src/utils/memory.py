import gc
import polars as pl


def freemem(df: pl.DataFrame):
    for col in df.columns:
        if df[col].dtype == pl.Int64:
            df = df.with_column(pl.col(col).cast(pl.Int32))
        elif df[col].dtype == pl.Float64:
            df = df.with_column(pl.col(col).cast(pl.Float32))
    gc.collect()
    return df
