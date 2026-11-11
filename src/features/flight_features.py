from psycopg2.errors import AmbiguousAlias
import pandas as pd 

AIRPORT_TIMEZONES = {
    "ATL": "America/New_York",
    "DEN": "America/Denver",
    "DFW": "America/Chicago",
    "JFK": "America/New_York",
    "LAX": "America/Los_Angeles",
    "MSP": "America/Chicago",
    "ORD": "America/Chicago",
    "PHX": "America/Phoenix",
    "SEA": "America/Los_Angeles",
    "SFO": "America/Los_Angeles"
}


def add_flight_time_features(flights, prediction_minutes=60):
    results = flights.copy()

    dates = pd.to_datetime(
        results["flight_date"],
        errors="raise"
    ).dt.normalize()

    times = results["scheduled_departure_hhmm"].astype("string")

    valid_times = times.str.fullmatch(r"(?:[01][0-9]|2[0-3])[0-5][0-9]|2400"
    ).fillna(False)


    if dates.isna().any() or not valid_times.all():
        raise ValueError("Missing flight dates or invalid departure times.")

    if not results["origin"].isin(AIRPORT_TIMEZONES).all():
        raise ValueError("An Origin outside our config airports")

    
    hours = times.str[:2].astype(int)

    minutes = times.str[2:].astype(int)

    local_departures = dates + pd.to_timedelta(
        hours * 60 + minutes,
        unit="min"
    )

    departure_utc = pd.Series(
        pd.NaT,
        index=results.index,
        dtype="datetime64[ns, UTC]"
    )
    for airport, timezone in AIRPORT_TIMEZONES.items():
        mask = results["origin"].eq(airport)

        departure_utc.loc[mask] = (
            local_departures.loc[mask].dt.tz_localize(
                timezone,
                ambiguous="NaT",
                nonexistent="NaT",

            )
            .dt.tz_convert("UTC")
        )

    results["scheduled_departure_utc"] = departure_utc
    results["prediction_time_utc"] = (
        departure_utc
        - pd.Timedelta(minutes=prediction_minutes)
    )

    results["scheduled_departure_hour"] = hours.mod(24)
    results["is_weekend"] = dates.dt.dayofweek.ge(5)
    results["unresolved_departure_time"] = departure_utc.isna()

    return results

    def clean_duration_features(flights):
        results = flights.copy()

        duration = pd.to_numeric(
            results["scheduled_elapsed_minutes"],
            errors="raise"
        )

        results["invalid_scheduled_duration"] = (
        duration.le(0).fillna(False)
        )

        results["scheduled_elapsed_minutes"] = duration.where(
            duration.gt(0)
        )

        return results
    
def clean_duration_features(flights):
    result = flights.copy()

    duration = pd.to_numeric(
        result["scheduled_elapsed_minutes"],
        errors="raise"
    )

    result["invalid_scheduled_duration"] = (
        duration.le(0).fillna(False)
    )

    result["scheduled_elapsed_minutes"] = duration.where(
        duration.gt(0)
    )

    return result