import polars as pl
from src.utils.date_function import get_hour_from_ts, get_weekday_from_ts


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
    # num reversed chrono + session_len
    data = data.with_columns(
        [
            pl.col("ts").shift().over("session").alias("prev_ts"),
            pl.col("session")
            .cumcount()
            .reverse()
            .over("session")
            .alias("action_num_reverse_chrono"),
            pl.col("session").count().over("session").alias("session_len"),
        ]
    )
    # add log_recency_score
    linear_interpolation = 0.1 + ((1 - 0.1) / (data["session_len"] - 1)) * (
        data["session_len"] - data["action_num_reverse_chrono"] - 1
    )
    data = data.with_columns(
        [
            (pl.col("ts") - pl.col("prev_ts")).fill_null(0).alias("duration_second"),
            pl.Series(2**linear_interpolation - 1)
            .alias("log_recency_score")
            .fill_nan(1),
        ]
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
    # add type_weighted_log_recency_score
    type_weights = {0: 1, 1: 6, 2: 3}
    type_weighted_log_recency_score = pl.Series(
        data["type"].apply(lambda x: type_weights[x]) * data["log_recency_score"]
    )
    data = data.with_columns(
        [type_weighted_log_recency_score.alias("type_weighted_log_recency_score")]
    )
    # add hour and weekday
    data = data.with_columns(
        [
            pl.col("ts").apply(lambda x: get_hour_from_ts(x)).alias("hour"),
            pl.col("ts").apply(lambda x: get_weekday_from_ts(x)).alias("weekday"),
        ],
    )
    # END: event data preprocess

    return data
