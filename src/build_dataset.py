from pathlib import Path
import json
import os
import argparse
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text, bindparam

from src.features.flight_features import (
    AIRPORT_TIMEZONES,
    add_flight_time_features,
    clean_duration_features
)
from src.features.weather_features import add_weather_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_AIRPORTS = list(AIRPORT_TIMEZONES)

START_DATE = "2026-01-01"
END_DATE = "2026-02-01"

PREDICTION_MINUTES = 60
MAX_WEATHER_AGE_MINUTES = 90

WEATHER_FEATURES = [
    "temperature_c",
    "dewpoint_c",
    "temperature_dewpoint_spread_c",
    "relative_humidity_pct",
    "wind_speed_kt",
    "wind_gust_kt",
    "visibility_mi",
    "reported_ceiling_ft",
    "precipitation_in",
    "precipitation_trace"
]


def load_data(start_date, end_date):
    load_dotenv(PROJECT_ROOT / ".env")

    database_url = URL.create(
        "postgresql+psycopg2",
        username="aviation",
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        database="aviation_mlops"
    )

    engine = create_engine(
        database_url,
        connect_args={
            "options": "-c default_transaction_read_only=on"
        }
    )

    flight_query = text("""
        SELECT
            f.flight_id,
            f.flight_date,
            f.day_of_week,
            f.carrier,
            f.origin,
            f.destination,
            f.scheduled_departure_hhmm,
            f.scheduled_elapsed_minutes,
            f.distance_miles,
            o.departure_delayed_15
        FROM flights AS f
        JOIN flight_outcomes AS o
            ON f.flight_id = o.flight_id
        WHERE f.origin IN :airports
          AND f.flight_date >= :start_date
          AND f.flight_date < :end_date
          AND o.cancelled = FALSE
          AND o.departure_delayed_15 IS NOT NULL
    """).bindparams(bindparam("airports", expanding=True))

    weather_query = text("""
        SELECT
            id AS weather_report_id,
            airport_code,
            observation_time,
            temperature_c,
            wind_speed_kt,
            visibility_mi,
            raw_data
        FROM historical_weather
        WHERE airport_code IN :airports
          AND observation_time >= :start_time
          AND observation_time < :end_time
    """).bindparams(bindparam("airports", expanding=True))

    weather_start = (
        pd.Timestamp(START_DATE, tz="UTC")
        - pd.Timedelta(days=1)
    )
    weather_end = (
        pd.Timestamp(END_DATE, tz="UTC")
        + pd.Timedelta(days=1)
    )

    try:
        with engine.connect().execution_options(
            isolation_level="REPEATABLE READ"
        ) as connection:
            with connection.begin():
                flights = pd.read_sql_query(
                    flight_query,
                    connection,
                    params={
                        "airports": PROJECT_AIRPORTS,
                        "start_date": START_DATE,
                        "end_date": END_DATE
                    },
                    parse_dates=["flight_date"]
                )

                weather = pd.read_sql_query(
                    weather_query,
                    connection,
                    params={
                        "airports": [
                            f"K{airport}"
                            for airport in PROJECT_AIRPORTS
                        ],
                        "start_time": weather_start.to_pydatetime(),
                        "end_time": weather_end.to_pydatetime()
                    }
                )
    finally:
        engine.dispose()

    if flights.empty or weather.empty:
        raise ValueError("Flight or weather data is empty.")

    return flights, weather


def build_dataset(flights, weather):
    flights = add_flight_time_features(
        flights,
        prediction_minutes=PREDICTION_MINUTES
    )
    flights = clean_duration_features(flights)

    flights["departure_delayed_15"] = (
        flights["departure_delayed_15"].astype("boolean")
    )

    # All configured project airports use these ICAO identifiers.
    flights["airport_code"] = flights["origin"].map({
        airport: f"K{airport}"
        for airport in PROJECT_AIRPORTS
    })

    weather = add_weather_features(weather)

    ambiguous_reports = weather.duplicated(
        ["airport_code", "observation_time"],
        keep=False
    )

    weather_columns = [
        "weather_report_id",
        "airport_code",
        "observation_time",
        *WEATHER_FEATURES
    ]

    eligible_weather = weather.loc[
        ~ambiguous_reports,
        weather_columns
    ].sort_values(
        ["observation_time", "airport_code"]
    )

    if eligible_weather.empty:
        raise ValueError("No unambiguous weather reports available.")

    valid_times = flights["prediction_time_utc"].notna()

    matchable_flights = flights.loc[valid_times].sort_values(
        ["prediction_time_utc", "airport_code"]
    )
    matchable_flights["prediction_time_utc"] = (
    matchable_flights["prediction_time_utc"]
    .astype("datetime64[ns, UTC]")
)

    eligible_weather["observation_time"] = (
        eligible_weather["observation_time"]
        .astype("datetime64[ns, UTC]")
    )

    matched = pd.merge_asof(
        matchable_flights,
        eligible_weather,
        left_on="prediction_time_utc",
        right_on="observation_time",
        by="airport_code",
        direction="backward",
        tolerance=pd.Timedelta(
            minutes=MAX_WEATHER_AGE_MINUTES
        ),
        allow_exact_matches=True
    )

    # Preserve flights whose departure timezone could not be resolved.
    unresolved = flights.loc[~valid_times]

    if not unresolved.empty:
        matched = pd.concat(
            [matched, unresolved],
            ignore_index=True
        )

    matched["observation_time"] = pd.to_datetime(
        matched["observation_time"],
        utc=True
    )

    matched["weather_report_id"] = (
        matched["weather_report_id"].astype("Int64")
    )

    matched["weather_missing"] = (
        matched["weather_report_id"].isna()
    )

    matched["weather_age_minutes"] = (
        matched["prediction_time_utc"]
        - matched["observation_time"]
    ).dt.total_seconds().div(60)

    if (
        len(matched) != len(flights)
        or not matched["flight_id"].is_unique
    ):
        raise ValueError("Matching changed flight counts or duplicated flights.")

    has_weather = ~matched["weather_missing"]

    if not has_weather.any():
        raise ValueError("No flights matched weather.")

    ages = matched.loc[has_weather, "weather_age_minutes"]

    if not ages.between(0, MAX_WEATHER_AGE_MINUTES).all():
        raise ValueError("A weather match violates the time rules.")

    audit = {
        "flights": len(matched),
        "matched_flights": int(has_weather.sum()),
        "unmatched_flights": int((~has_weather).sum()),
        "matched_percent": round(float(has_weather.mean() * 100), 2),
        "unresolved_departure_times": int((~valid_times).sum()),
        "ambiguous_weather_reports_excluded": int(
            ambiguous_reports.sum()
        ),
        "prediction_minutes_before_departure": PREDICTION_MINUTES,
        "maximum_weather_age_minutes": MAX_WEATHER_AGE_MINUTES,
        "oldest_matched_weather_minutes": float(ages.max())
    }

    return matched.sort_values("flight_id").reset_index(drop=True), audit

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the flight-weather development dataset."
    )

    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default="2026-01-01"
    )

    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default="2026-02-01",
        help="Exclusive end date."
    )

    args = parser.parse_args()

    if args.start_date >= args.end_date:
        parser.error("start-date must be earlier than end-date.")

    return args


def main():
    args = parse_args()

    flights, weather = load_data(
        args.start_date,
        args.end_date
    )

    dataset, audit = build_dataset(flights, weather)

    missing_rate = dataset["weather_missing"].mean()

    if missing_rate > 0.01:
        raise ValueError(
            f"Weather unmatched rate is {missing_rate:.2%}; "
            "exceeds the 1% limit."
        )

    audit["rows_dropped_no_weather"] = int(
        dataset["weather_missing"].sum()
    )

    dataset = dataset.loc[
        ~dataset["weather_missing"]
    ].copy()

    audit["output_flights"] = len(dataset)
    audit["start_date"] = args.start_date.isoformat()
    audit["end_date_exclusive"] = args.end_date.isoformat()

    period = (
        f"{args.start_date:%Y%m%d}_"
        f"{args.end_date:%Y%m%d}"
    )

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / f"flight_weather_{period}.parquet"
    dataset.to_parquet(dataset_path, index=False)

    reports_dir = PROJECT_ROOT / "reports" / "audits"
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_path = reports_dir / f"flight_weather_build_{period}.json"
    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(audit, indent=2))
    print(f"\nDataset saved: {dataset_path}")
    print(f"Audit saved: {audit_path}")


if __name__ == "__main__":
    main()