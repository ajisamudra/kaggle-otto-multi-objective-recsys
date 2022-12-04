import polars as pl


def preprocess_events(data: pl.DataFrame):
    """
    df input
    session | type | ts | aid
    123 | 0 | 12313 | AID1
    123 | 1 | 12314 | AID1
    123 | 2 | 12345 | AID1
    """

    # START: event data preprocess
    oneday_cutoff = 1 * 60 * 60 * 24
    # sort session & ts ascendingly
    data = data.sort(["session", "ts"])

    # shift ts per session & get duration between event
    data = data.with_columns([pl.col("ts").shift().over("session").alias("prev_ts")])
    data = data.with_columns(
        [(pl.col("ts") - pl.col("prev_ts")).fill_null(0).alias("duration_second")]
    )
    data = data.with_columns(
        [
            pl.when(pl.col("duration_second") > oneday_cutoff)
            .then(1)
            .otherwise(0)
            .alias("oneday_session"),
            pl.when(pl.col("duration_second") > oneday_cutoff)
            .then(0)  # start of real-session will always have 0 duration_second
            .otherwise(pl.col("duration_second"))
            .alias("duration_second"),
        ]
    )
    # END: event data preprocess

    return data
