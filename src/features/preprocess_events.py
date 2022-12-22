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
            pl.col("ts").shift(periods=-1).over("session").alias("next_ts"),
            pl.col("session")
            .cumcount()
            .reverse()
            .over("session")
            .alias("action_num_reverse_chrono"),
            pl.col("session").count().over("session").alias("session_len"),
        ]
    )
    data = data.with_columns(
        [
            (pl.col("action_num_reverse_chrono") + 1).alias(
                "action_num_reverse_chrono"
            ),
        ]
    )
    # add log_recency_score
    linear_interpolation = 0.1 + ((1 - 0.1) / (data["session_len"] - 1)) * (
        data["session_len"] - data["action_num_reverse_chrono"]
    )
    data = data.with_columns(
        [
            (pl.col("next_ts") - pl.col("ts")).alias("duration_second"),
            pl.Series(2**linear_interpolation - 1)
            .alias("log_recency_score")
            .fill_nan(1),
        ]
    )
    data = data.with_columns(
        [
            pl.col("duration_second")
            .over("session")
            .shift()
            .alias("shifted_duration_second"),
        ]
    )
    data = data.with_columns(
        [
            # end of session will have duration second as the same as last 2 event
            pl.col("duration_second")
            .fill_null(pl.col("shifted_duration_second"))
            .alias("duration_second")
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
            pl.col("duration_second").fill_null(0).alias("duration_second"),
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

    # add second_duration & recency score
    data = data.with_columns(
        [
            (
                pl.col("type_weighted_log_recency_score") * pl.col("duration_second")
            ).alias("type_weighted_log_duration_second"),
            (pl.col("log_recency_score") * pl.col("duration_second")).alias(
                "log_duration_second"
            ),
        ]
    )

    # drop cols
    data = data.drop(
        columns=[
            "shifted_duration_second",
            "next_ts",
            "session_len",
        ]
    )

    # END: event data preprocess

    return data


def get_real_session_events(data: pl.DataFrame):
    """
    df input
    session | type | ts | aid
    123 | 0 | 12313 | AID1
    123 | 1 | 12314 | AID1
    123 | 2 | 12345 | AID1
    """

    # START: event data preprocess
    cutoff = 1 * 60 * 60 * 2
    # sort session & ts ascendingly
    data = data.sort(["session", "ts"])

    # shift ts per session & get duration between event
    # num reversed chrono + session_len
    data = data.with_columns(
        [
            pl.col("ts").shift(periods=-1).over("session").alias("next_ts"),
            pl.col("session").count().over("session").alias("session_len"),
        ]
    )
    data = data.with_columns(
        [
            (pl.col("next_ts") - pl.col("ts")).alias("duration_second"),
        ]
    )
    data = data.with_columns(
        [
            pl.col("duration_second")
            .over("session")
            .shift()
            .alias("shifted_duration_second"),
        ]
    )
    data = data.with_columns(
        [
            # end of session will have duration second as the same as last 2 event
            pl.col("duration_second")
            .fill_null(pl.col("shifted_duration_second"))
            .alias("duration_second")
        ]
    )
    data = data.with_columns(
        [
            pl.when(pl.col("duration_second") > cutoff)
            .then(1)
            .otherwise(0)
            .alias("real_session"),
        ]
    )
    data = data.with_columns(
        [pl.col("real_session").cumsum().over("session").alias("real_session_id")]
    )
    data = data.with_columns(
        [
            (pl.col("session") + pl.lit("_") + pl.col("real_session_id")).alias(
                "real_session_id"
            )
        ]
    )
    data = data.with_columns(
        [
            pl.col("real_session_id")
            .count()
            .over("real_session_id")
            .alias("real_session_len"),
        ]
    )

    # drop cols
    data = data.drop(
        columns=[
            "shifted_duration_second",
            "next_ts",
        ]
    )

    # END: event data preprocess

    return data
