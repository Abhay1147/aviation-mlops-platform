import argparse
import csv
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import requests


COLUMN_MAP = {
    "DayOfWeek": "DAY_OF_WEEK",
    "FlightDate": "FL_DATE",
    "Reporting_Airline": "OP_UNIQUE_CARRIER",
    "Tail_Number": "TAIL_NUM",
    "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM",
    "OriginAirportID": "ORIGIN_AIRPORT_ID",
    "Origin": "ORIGIN",
    "DestAirportID": "DEST_AIRPORT_ID",
    "Dest": "DEST",
    "CRSDepTime": "CRS_DEP_TIME",
    "DepTime": "DEP_TIME",
    "DepDelay": "DEP_DELAY",
    "DepDelayMinutes": "DEP_DELAY_NEW",
    "DepDel15": "DEP_DEL15",
    "DepartureDelayGroups": "DEP_DELAY_GROUP",
    "CRSArrTime": "CRS_ARR_TIME",
    "ArrTime": "ARR_TIME",
    "ArrDelay": "ARR_DELAY",
    "ArrDelayMinutes": "ARR_DELAY_NEW",
    "ArrDel15": "ARR_DEL15",
    "ArrivalDelayGroups": "ARR_DELAY_GROUP",
    "Cancelled": "CANCELLED",
    "CancellationCode": "CANCELLATION_CODE",
    "Diverted": "DIVERTED",
    "CRSElapsedTime": "CRS_ELAPSED_TIME",
    "Distance": "DISTANCE",
}


def download_archive(year, month, destination):
    if destination.exists():
        print(f"Using existing archive: {destination}")
        return

    url = (
        "https://transtats.bts.gov/PREZIP/"
        "On_Time_Reporting_Carrier_On_Time_Performance"
        f"_1987_present_{year}_{month}.zip"
    )
    temporary = destination.with_suffix(".zip.part")

    try:
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()

            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)

        # Reject an HTML error page or an unreadable archive.
        with ZipFile(temporary) as archive:
            if not any(
                name.lower().endswith(".csv")
                for name in archive.namelist()
            ):
                raise ValueError("Downloaded archive contains no CSV.")

        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def convert_archive(archive_path, output_path, year, month):
    temporary = output_path.with_suffix(".csv.part")
    row_count = 0

    try:
        with ZipFile(archive_path) as archive:
            names = [
                name for name in archive.namelist()
                if name.lower().endswith(".csv")
            ]

            if len(names) != 1:
                raise ValueError(f"Expected one CSV, found: {names}")

            with (
                archive.open(names[0]) as source,
                temporary.open(
                    "w", newline="", encoding="utf-8"
                ) as output,
            ):
                reader = csv.DictReader(
                    TextIOWrapper(source, encoding="utf-8-sig")
                )

                missing = set(COLUMN_MAP) - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"Missing BTS columns: {sorted(missing)}")

                writer = csv.DictWriter(
                    output,
                    fieldnames=list(COLUMN_MAP.values())
                )
                writer.writeheader()

                for row in reader:
                    flight_date = datetime.strptime(
                        row["FlightDate"], "%Y-%m-%d"
                    )

                    if (flight_date.year, flight_date.month) != (year, month):
                        raise ValueError(
                            f"Unexpected flight date: {row['FlightDate']}"
                        )

                    converted = {
                        target: row[source]
                        for source, target in COLUMN_MAP.items()
                    }

                    # Match the existing flight loader's date format.
                    converted["FL_DATE"] = (
                        f"{flight_date:%m/%d/%Y} 12:00:00 AM"
                    )

                    writer.writerow(converted)
                    row_count += 1

        if row_count == 0:
            raise ValueError("Flight CSV contains no records.")

        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    return row_count


def main():
    parser = argparse.ArgumentParser(
        description="Download and adapt a BTS monthly flight archive."
    )
    parser.add_argument("--month", required=True, help="YYYY-MM")
    args = parser.parse_args()

    try:
        month_date = datetime.strptime(args.month, "%Y-%m")
    except ValueError:
        parser.error("--month must be a valid YYYY-MM")

    month_tag = month_date.strftime("%Y-%m")
    year = month_date.year
    month = month_date.month

    project_root = Path(__file__).resolve().parents[2]
    directory = project_root / "data" / "raw" / "flights" / month_tag
    directory.mkdir(parents=True, exist_ok=True)

    archive_path = directory / f"bts_{year}_{month:02d}.zip"
    output_path = directory / "Aviation_flight.csv"

    download_archive(year, month, archive_path)
    rows = convert_archive(archive_path, output_path, year, month)

    print(f"Month: {month_tag}")
    print(f"Flight rows: {rows:,}")
    print(f"Original archive: {archive_path}")
    print(f"Ingestion CSV: {output_path}")


if __name__ == "__main__":
    main()