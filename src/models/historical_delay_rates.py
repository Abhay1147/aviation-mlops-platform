import html
import re

import numpy as np
import pandas as pd
import requests
from sqlalchemy import bindparam, text

from src.features.flight_features import AIRPORT_TIMEZONES


RATE_COLUMNS = [
    "origin_history_delay_rate",
    "carrier_origin_history_delay_rate",
]

ARCHIVE_INDEX = "https://transtats.bts.gov/PREZIP/"
AVAILABILITY_BUFFER_DAYS = 2
AIRPORT_PRIOR_FLIGHTS = 100

ARCHIVE_ROW = re.compile(
    r"(?P<date>\d{1,2}/\d{1,2}/\d{4})\b[^\r\n]*?"
    r"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
    r"(?P<year>\d{4})_(?P<month>\d{1,2})\.zip",
    re.IGNORECASE,
)

HISTORY_QUERY = text("""
    SELECT to_char(f.flight_date, 'YYYY-MM') AS month,
           f.origin,
           f.carrier,
           count(*) AS flights,
           count(*) FILTER (WHERE o.departure_delayed_15) AS delayed
    FROM flights AS f
    JOIN flight_outcomes AS o USING (flight_id)
    WHERE f.origin IN :airports
      AND f.flight_date >= DATE '2024-01-01'
      AND f.flight_date < DATE '2026-05-01'
      AND o.cancelled = FALSE
      AND o.departure_delayed_15 IS NOT NULL
    GROUP BY 1, 2, 3
""").bindparams(bindparam("airports", expanding=True))


def load_history(connection):
    history = pd.read_sql_query(
        HISTORY_QUERY,
        connection,
        params={"airports": list(AIRPORT_TIMEZONES)},
    )
    if history.empty:
        raise ValueError("No historical flight outcomes found.")

    response = requests.get(ARCHIVE_INDEX, timeout=(10, 60))
    response.raise_for_status()

    rows = re.sub(r"<br\s*/?>", "\n", response.text, flags=re.IGNORECASE)
    rows = html.unescape(re.sub(r"<[^>]+>", "", rows))

    available_at = {}
    for match in ARCHIVE_ROW.finditer(rows):
        month = f"{match['year']}-{int(match['month']):02d}"
        cutoff = pd.to_datetime(match["date"], format="%m/%d/%Y", utc=True)
        cutoff += pd.Timedelta(days=AVAILABILITY_BUFFER_DAYS)
        available_at[month] = max(available_at.get(month, cutoff), cutoff)

    history["available_at"] = history["month"].map(available_at)
    missing = sorted(history.loc[history["available_at"].isna(), "month"].unique())
    if missing:
        raise ValueError(f"BTS archive dates missing for: {missing}")

    return history


def historical_rates(flights, history):
    """Calculate rates using only the archives eligible at prediction time."""
    cutoffs = pd.DatetimeIndex(history["available_at"].unique()).sort_values()
    prediction_times = pd.to_datetime(flights["prediction_time_utc"], utc=True)
    if prediction_times.isna().any():
        raise ValueError("Missing prediction timestamp.")

    cutoff_index = cutoffs.searchsorted(prediction_times, side="right") - 1
    rates = np.full((len(flights), len(RATE_COLUMNS)), np.nan, dtype=np.float32)

    for index, cutoff in enumerate(cutoffs):
        positions = np.flatnonzero(cutoff_index == index)
        if positions.size == 0:
            continue

        known = history.loc[history["available_at"] <= cutoff]
        airport = known.groupby("origin")[["flights", "delayed"]].sum()
        airport_rate = airport["delayed"] / airport["flights"]

        airline_airport = (
            known.groupby(["origin", "carrier"])[["flights", "delayed"]].sum()
        )

        batch = flights.iloc[positions]
        prior_rate = batch["origin"].map(airport_rate).to_numpy(dtype=np.float64)
        keys = pd.MultiIndex.from_frame(batch[["origin", "carrier"]])
        counts = airline_airport.reindex(keys, fill_value=0)

        rates[positions, 0] = prior_rate
        rates[positions, 1] = (
            counts["delayed"].to_numpy() + AIRPORT_PRIOR_FLIGHTS * prior_rate
        ) / (counts["flights"].to_numpy() + AIRPORT_PRIOR_FLIGHTS)

    return rates
