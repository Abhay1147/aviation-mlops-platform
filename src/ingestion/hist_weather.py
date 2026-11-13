from io import StringIO
import pandas as pd 
import requests
from pathlib import Path
import os
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json, execute_values
import argparse

def fetch_historical_weather(start_date, end_date):
    
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

    airports = [
        "ATL", "DEN", "DFW", "JFK", "LAX",
        "MSP", "ORD", "PHX", "SEA", "SFO"
    ]

    params = {
        "station": airports,
        "network": [
            "GA_ASOS", "CO_ASOS", "TX_ASOS",
            "NY_ASOS", "CA_ASOS", "MN_ASOS",
            "IL_ASOS", "AZ_ASOS", "WA_ASOS"
        ],
        "data": "all",
        "sts": f"{start_date}T00:00:00Z",
        "ets": f"{end_date}T00:00:00Z",
        "tz": "UTC",
        "format": "onlycomma",
        "report_type": [3, 4],
        "missing": "M",
        "trace": "T",
        "latlon": "no",
        "elev": "no"
    }

    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()

    weather_df = pd.read_csv(
        StringIO(response.text),
        comment="#",
        dtype="string",
        keep_default_na=False
    )

    required_columns = {
        "station", "valid", "tmpf", "sknt", "vsby", "metar"
    }

    if not required_columns.issubset(weather_df.columns):
        raise ValueError("Unexpected weather response columns.")

    missing_airports = set(airports) - set(weather_df["station"])

    if missing_airports:
        raise ValueError(f"No observations for: {missing_airports}")

    return weather_df

def prep_weather(weather_df):
    prepared = weather_df.copy()

    prepared["observation_time"] = pd.to_datetime(
        prepared["valid"],
        utc=True,
        errors="raise"
    )
    def numeric_cols(name):
        return pd.to_numeric(
            prepared[name].replace({"M":pd.NA,"": pd.NA}),
            errors='raise'
        )
    prepared["temperature_c"] =  (
        numeric_cols("tmpf") - 32) * 5/9
    
    prepared["wind_speed_kt"] = numeric_cols("sknt")
    prepared["visibility_mi"] = numeric_cols("vsby")

    return prepared


def save_historical_weather(weather_df, prepared):
    airport_codes = {
        "ATL": "KATL", "DEN": "KDEN", "DFW": "KDFW",
        "JFK": "KJFK", "LAX": "KLAX", "MSP": "KMSP",
        "ORD": "KORD", "PHX": "KPHX", "SEA": "KSEA",
        "SFO": "KSFO"
    }

    def db_number(value):
        return None if pd.isna(value) else float(value)

    if not weather_df.index.equals(prepared.index):
        raise ValueError("Source and prepared rows are not aligned.")

    rows = []

    for source, record in zip(
        weather_df.to_dict("records"),
        prepared.to_dict("records")
    ):
        if pd.isna(record["observation_time"]):
            raise ValueError("Missing observation timestamp.")

        if source["metar"].strip() in ("", "M"):
            raise ValueError("Missing METAR report text.")

        rows.append((
            airport_codes[source["station"]],
            record["observation_time"].to_pydatetime(),
            db_number(record["temperature_c"]),
            db_number(record["wind_speed_kt"]),
            db_number(record["visibility_mi"]),
            source["metar"],
            Json(source)
        ))

    if not rows:
        return 0

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    connection = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        dbname="aviation_mlops",
        user="aviation",
        password=os.environ["POSTGRES_PASSWORD"]
    )

    try:
        with connection:
            with connection.cursor() as cursor:
                inserted = execute_values(
                    cursor,
                    """
                    INSERT INTO historical_weather (
                        airport_code,
                        observation_time,
                        temperature_c,
                        wind_speed_kt,
                        visibility_mi,
                        metar,
                        raw_data
                    )
                    VALUES %s
                    ON CONFLICT ON CONSTRAINT
                        uq_historical_weather_report
                    DO NOTHING
                    RETURNING id;
                    """,
                    rows,
                    page_size=1000,
                    fetch=True
                )

        return len(inserted)

    finally:
        connection.close()

def main():
    parser = argparse.ArgumentParser(
        description="Load a historical-weather CSV into PostgreSQL."
    )
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    source_path = Path(args.file)

    weather_df = pd.read_csv(
        source_path,
        comment="#",
        dtype="string",
        keep_default_na=False
    )

    prepared = prep_weather(weather_df)
    inserted = save_historical_weather(weather_df, prepared)

    print(f"Source observations: {len(weather_df):,}")
    print(f"New observations saved: {inserted:,}")
    print(
        f"Existing reports skipped: "
        f"{len(weather_df) - inserted:,}"
    )
if __name__ == "__main__":
    main()
