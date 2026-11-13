import argparse
from datetime import date, timedelta
from pathlib import Path

from src.ingestion.hist_weather import fetch_historical_weather


def main():
    parser = argparse.ArgumentParser(
        description="Download historical weather for one flight month."
    )
    parser.add_argument("--month", required=True, help="YYYY-MM")
    args = parser.parse_args()

    try:
        start = date.fromisoformat(f"{args.month}-01")
    except ValueError:
        parser.error("--month must be a valid YYYY-MM")

    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)

    # Cover local-time departures and the weather matching window.
    weather_start = start - timedelta(days=1)
    weather_end = next_month + timedelta(days=1)

    project_root = Path(__file__).resolve().parents[2]
    directory = project_root / "data" / "raw" / "weather"
    directory.mkdir(parents=True, exist_ok=True)

    filename = (
        f"iem_{weather_start:%Y%m%d}_{weather_end:%Y%m%d}.csv"
    )
    destination = directory / filename
    temporary = destination.with_suffix(".csv.part")

    if destination.exists():
        print(f"Using existing weather CSV: {destination}")
        return

    weather = fetch_historical_weather(
        weather_start.isoformat(),
        weather_end.isoformat()
    )

    try:
        weather.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Weather observations: {len(weather):,}")
    print(f"Airports returned: {weather['station'].nunique()}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()