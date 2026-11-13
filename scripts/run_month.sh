#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/run_month.sh YYYY-MM WEATHER_START WEATHER_END"
    echo "Example: bash scripts/run_month.sh 2026-01 2025-12-31 2026-02-02"
    exit 1
fi

MONTH="$1"
WEATHER_START="$2"
WEATHER_END="$3"

YEAR="${MONTH%-*}"
MONTH_NUMBER="${MONTH#*-}"

START_DATE="${MONTH}-01"
END_DATE=$(date -u -d "$START_DATE +1 month" +%F)

START_TAG="${START_DATE//-/}"
END_TAG="${END_DATE//-/}"
WEATHER_START_TAG="${WEATHER_START//-/}"
WEATHER_END_TAG="${WEATHER_END//-/}"

BUCKET="aviation-mlops-939377869323-us-east-2"
REGION="us-east-2"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

FLIGHT_FILE="data/raw/flights/${MONTH}/Aviation_flight.csv"
WEATHER_FILE="data/raw/weather/iem_${WEATHER_START_TAG}_${WEATHER_END_TAG}.csv"
OUTPUT_NAME="flight_weather_${START_TAG}_${END_TAG}"

mkdir -p "$(dirname "$FLIGHT_FILE")" data/raw/weather reports/audits

echo "Downloading raw files..."

aws s3 cp \
    "s3://${BUCKET}/raw/flights/year=${YEAR}/month=${MONTH_NUMBER}/Aviation_flight.csv" \
    "$FLIGHT_FILE" \
    --region "$REGION"

aws s3 cp \
    "s3://${BUCKET}/raw/weather/request_start=${WEATHER_START}/request_end=${WEATHER_END}/iem_${WEATHER_START_TAG}_${WEATHER_END_TAG}.csv" \
    "$WEATHER_FILE" \
    --region "$REGION"

echo "Loading flights..."

docker compose run --rm ingestion \
    python -m src.ingestion.flights \
    --file "$FLIGHT_FILE"

echo "Loading historical weather..."

docker compose run --rm ingestion \
    python -m src.ingestion.hist_weather \
    --file "$WEATHER_FILE"

echo "Building dataset..."

docker compose run --rm \
    -v "$PROJECT_ROOT/reports:/app/reports" \
    ingestion python -m src.build_dataset \
    --start-date "$START_DATE" \
    --end-date "$END_DATE"

echo "Uploading dataset and audit..."

aws s3 cp \
    "data/processed/${OUTPUT_NAME}.parquet" \
    "s3://${BUCKET}/processed/flight_weather/year=${YEAR}/month=${MONTH_NUMBER}/${OUTPUT_NAME}.parquet" \
    --region "$REGION"

aws s3 cp \
    "reports/audits/flight_weather_build_${START_TAG}_${END_TAG}.json" \
    "s3://${BUCKET}/reports/audits/year=${YEAR}/month=${MONTH_NUMBER}/flight_weather_build_${START_TAG}_${END_TAG}.json" \
    --region "$REGION"

echo "Monthly pipeline completed: ${MONTH}"