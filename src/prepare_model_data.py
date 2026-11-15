from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = SOURCE_DIR / "model"


CATEGORICAL_FEATURES= [
    "carrier",
    "origin",
    "destination",
]

NUMERIC_FEATURES = [
    "day_of_week",
    "flight_month",
    "scheduled_departure_hour",
    "scheduled_elapsed_minutes",
    "distance_miles",
    "temperature_c",
    "dewpoint_c",
    "temperature_dewpoint_spread_c",
    "relative_humidity_pct",
    "wind_speed_kt",
    "wind_gust_kt",
    "visibility_mi",
    "reported_ceiling_ft",
    "precipitation_in",
    "weather_age_minutes",
]

BOOLEAN_FEATURES = [
    "invalid_scheduled_duration",
    "precipitation_trace",
]

TARGET = "departure_delayed_15"

SOURCE_COLUMNS = [
    "flight_id",
    "flight_date",
    "prediction_time_utc",
    *CATEGORICAL_FEATURES,
    *[
        column for column in NUMERIC_FEATURES
        if column != "flight_month"
    ],
    *BOOLEAN_FEATURES,
    TARGET,
]

SCHEMA = pa.schema([
    pa.field("flight_id", pa.int64()),
    pa.field("flight_date", pa.timestamp("ns")),
    pa.field("prediction_time_utc", pa.timestamp("ns", tz="UTC")),
    *[
        pa.field(column, pa.string())
        for column in CATEGORICAL_FEATURES
    ],
    *[
        pa.field(column, pa.float64())
        for column in NUMERIC_FEATURES
    ],
    *[
        pa.field(column, pa.bool_())
        for column in BOOLEAN_FEATURES
    ],
    pa.field(TARGET, pa.bool_()),
])

SPLITS = {
    "train": ("2024-01-01", "2026-01-01"),
    "validation": ("2026-01-01", "2026-05-01"),
    "test": ("2026-05-01", "2026-09-01"),
}


def prepare_split(name, start_date, end_date):
    destination = OUTPUT_DIR / f"{name}.parquet"
    temporary = destination.with_suffix(".parquet.part")

    total_rows = 0
    delayed_rows = 0

    try:
        with pq.ParquetWriter(
            temporary,
            SCHEMA,
            compression="snappy"
        ) as writer:
            months = pd.date_range(
                start_date,
                end_date,
                freq="MS",
                inclusive="left"
            )

            for start in months:
                end = start + pd.offsets.MonthBegin(1)
                period = f"{start:%Y%m%d}_{end:%Y%m%d}"
                source = SOURCE_DIR / f"flight_weather_{period}.parquet"

                frame = pd.read_parquet(
                    source,
                    columns=SOURCE_COLUMNS
                )

                dates = pd.to_datetime(frame["flight_date"])
                valid_dates = (dates >= start) & (dates < end)

                if frame.empty or not valid_dates.all():
                    raise ValueError(f"Invalid flight dates in {source}")

                if frame["prediction_time_utc"].isna().any():
                    raise ValueError(f"Missing prediction times in {source}")

                if frame[TARGET].isna().any():
                    raise ValueError(f"Missing target labels in {source}")

                frame["flight_date"] = dates
                frame["flight_month"] = dates.dt.month

                table = pa.Table.from_pandas(
                    frame,
                    schema=SCHEMA,
                    preserve_index=False,
                    safe=True
                )
                writer.write_table(table)

                total_rows += len(frame)
                delayed_rows += int(frame[TARGET].sum())

                print(
                    f"{name}: {start:%Y-%m} | {len(frame):,} rows",
                    flush=True
                )

        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        f"{name} complete: {total_rows:,} rows | "
        f"delayed: {delayed_rows / total_rows:.2%} | "
        f"{destination}",
        flush=True
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, (start_date, end_date) in SPLITS.items():
        prepare_split(name, start_date, end_date)


if __name__ == "__main__":
    main()
