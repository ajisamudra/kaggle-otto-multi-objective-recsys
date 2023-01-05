from datetime import datetime, timedelta


def get_hour_from_ts(x):
    return int((datetime.utcfromtimestamp(x) + timedelta(hours=2)).strftime("%H"))


def get_weekday_from_ts(x):
    """Return the day of the week as an integer, where Monday is 0 and Sunday is 6."""
    return (datetime.utcfromtimestamp(x) + timedelta(hours=2)).weekday()


def get_datetime_from_ts(x):
    return (datetime.utcfromtimestamp(x) + timedelta(hours=2)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def get_date_from_ts(x):
    return (datetime.utcfromtimestamp(x) + timedelta(hours=2)).strftime("%Y-%m-%d")


def get_datehour_from_ts(x):
    return (datetime.utcfromtimestamp(x) + timedelta(hours=2)).strftime(
        "%Y-%m-%d %H:00:00"
    )


def get_date_dt_from_ts(x):
    return datetime.utcfromtimestamp(x) + timedelta(hours=2)
