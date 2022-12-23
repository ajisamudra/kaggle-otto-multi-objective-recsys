from datetime import datetime


def get_hour_from_ts(x):
    return int(datetime.utcfromtimestamp(x).strftime("%H"))


def get_weekday_from_ts(x):
    """Return the day of the week as an integer, where Monday is 0 and Sunday is 6."""
    return datetime.utcfromtimestamp(x).weekday()


def get_datetime_from_ts(x):
    return datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S")


def get_date_from_ts(x):
    return datetime.utcfromtimestamp(x).strftime("%Y-%m-%d")


def get_datehour_from_ts(x):
    return datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:00:00")
