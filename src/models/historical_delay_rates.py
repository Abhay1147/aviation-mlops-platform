import html
import re

import numpy as np
import pandas as pd
import requests
from sqlalchemy import bindparam, text

from src.features.flight_features import AIRPORT_TIMEZONES

RATE_COLUMNS = ["origin_history_delay_rate", "carrier_origin_history_delay_rate"]

ARCHIVE_INDEX = "https://transtats.bts.gov/PREZIP/"
ARCHIVE_ROW = re.compile(
    r"(?P<date>\d{1,2}/\d{1,2}/\d{4})\b[^\r\n]*?"
    r"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
    r"(?P<year>\d{4})_(?P<month>\d{1,2})\.zip",
    re.IGNORECASE,
)

HISTORY_QUERY = text("""
    SELECT to_char(f.flight_date, 'YYYY-MM') AS month,
           f.origin, f.carrier,
           count(*) AS flights,
           count(*) FILTER (WHERE o.departure_delayed_15) AS delayed
    FROM flights f
    JOIN flight_outcomes o USING (flight_id)
    WHERE f.origin IN :airports
      AND f.flight_date >= DATE '2024-01-01'
      AND f.flight_date < DATE '2026-05-01'
      AND o.cancelled = FALSE
      AND o.departure_delayed_15 IS NOT NULL
    GROUP BY 1, 2, 3
""").bindparams(bindparam("airports", expanding=True))


def load_history(connection):
    history = pd.read_sql_query(
        HISTORY_QUERY, connection, params={"airports": list(AIRPORT_TIMEZONES)}
    )
    if history.empty:
        raise ValueError("No historical outcomes found.")

    response = requests.get(ARCHIVE_INDEX, timeout=(10, 60))
    response.raise_for_status()
    rows = re.sub(r"<br\s*/?>", "\n", response.text, flags=re.IGNORECASE)
    rows = html.unescape(re.sub(r"<[^>]+>", "", rows))

    available = {}
    for match in ARCHIVE_ROW.finditer(rows):
        month = f"{match['year']}-{int(match['month']):02d}"
        cutoff = pd.to_datetime(match["date"], format="%m/%d/%Y", utc=True)
        cutoff += pd.Timedelta(days=2)
        available[month] = max(available.get(month, cutoff), cutoff)

    history["available_at"] = history["month"].map(available)
    missing = history.loc[history["available_at"].isna(), "month"].unique()
    if len(missing):
        raise ValueError(f"BTS archive dates missing for: {sorted(missing)}")

    return history


def historical_rates(flights, history):
    """Use the released totals available at each flight's prediction time."""
    releases = pd.DatetimeIndex(history["available_at"].unique()).sort_values()
    prediction_times = pd.to_datetime(flights["prediction_time_utc"], utc=True)
    if prediction_times.isna().any():
        raise ValueError("Missing prediction timestamp.")

    release_index = releases.searchsorted(prediction_times, side="right") - 1
    rates = np.full((len(flights), len(RATE_COLUMNS)), np.nan, dtype=np.float32)

    for index, cutoff in enumerate(releases):
        positions = np.flatnonzero(release_index == index)
        if positions.size == 0:
            continue

        known = history.loc[history["available_at"] <= cutoff]
        origin = known.groupby("origin")[["flights", "delayed"]].sum()
        origin_rate = origin["delayed"] / origin["flights"]
        carrier_origin = known.groupby(
            ["origin", "carrier"]
        )[["flights", "delayed"]].sum()

        batch = flights.iloc[positions]
        airport_rate = batch["origin"].map(origin_rate).to_numpy()
        keys = pd.MultiIndex.from_frame(batch[["origin", "carrier"]])
        counts = carrier_origin.reindex(keys, fill_value=0)

        # Give the airport rate the weight of 100 prior flights.
        rates[positions, 0] = airport_rate
        rates[positions, 1] = (
            counts["delayed"].to_numpy() + 100 * airport_rate
        ) / (counts["flights"].to_numpy() + 100)

    return rates
