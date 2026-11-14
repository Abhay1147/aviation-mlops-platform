#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BUCKET="aviation-mlops-939377869323-us-east-2"
REGION="us-east-2"

CURRENT="2024-02-01"
END_EXCLUSIVE="2026-09-01"

mkdir -p data/backfill_state reports/logs

while [[ "$CURRENT" < "$END_EXCLUSIVE" ]]; do
    MONTH=$(date -u -d "$CURRENT" +%Y-%m)
    YEAR=$(date -u -d "$CURRENT" +%Y)
    MONTH_NUMBER=$(date -u -d "$CURRENT" +%m)
    NEXT=$(date -u -d "$CURRENT +1 month" +%F)

    MARKER="data/backfill_state/${MONTH}.done"

    if [[ -f "$MARKER" ]]; then
        echo "Skipping completed month: $MONTH"
        CURRENT="$NEXT"
        continue
    fi

    AVAILABLE_KB=$(df -Pk . | awk 'NR == 2 {print $4}')

    if (( AVAILABLE_KB < 6 * 1024 * 1024 )); then
        echo "Stopping: less than 6 GiB free before $MONTH."
        exit 1
    fi

    WEATHER_START=$(date -u -d "$CURRENT -1 day" +%F)
    WEATHER_END=$(date -u -d "$NEXT +1 day" +%F)

    WEATHER_START_TAG="${WEATHER_START//-/}"
    WEATHER_END_TAG="${WEATHER_END//-/}"

    FLIGHT_DIR="data/raw/flights/${MONTH}"
    ARCHIVE="${FLIGHT_DIR}/bts_${YEAR}_${MONTH_NUMBER}.zip"
    FLIGHT_CSV="${FLIGHT_DIR}/Aviation_flight.csv"
    WEATHER_CSV="data/raw/weather/iem_${WEATHER_START_TAG}_${WEATHER_END_TAG}.csv"

    FLIGHT_PREFIX="s3://${BUCKET}/raw/flights/year=${YEAR}/month=${MONTH_NUMBER}"
    WEATHER_PREFIX="s3://${BUCKET}/raw/weather/request_start=${WEATHER_START}/request_end=${WEATHER_END}"

    echo "Starting month: $MONTH"

    docker compose run --rm -T \
        --user "$(id -u):$(id -g)" \
        ingestion python -m src.ingestion.download_flights \
        --month "$MONTH"

    docker compose run --rm -T \
        --user "$(id -u):$(id -g)" \
        ingestion python -m src.ingestion.download_weather \
        --month "$MONTH"

    aws s3 cp "$ARCHIVE" \
        "${FLIGHT_PREFIX}/$(basename "$ARCHIVE")" \
        --region "$REGION" --only-show-errors

    aws s3 cp "$FLIGHT_CSV" \
        "${FLIGHT_PREFIX}/Aviation_flight.csv" \
        --region "$REGION" --only-show-errors

    aws s3 cp "$WEATHER_CSV" \
        "${WEATHER_PREFIX}/$(basename "$WEATHER_CSV")" \
        --region "$REGION" --only-show-errors

    bash scripts/run_month.sh \
        "$MONTH" "$WEATHER_START" "$WEATHER_END"

    # Only mark complete after the monthly runner and uploads succeed.
    date -u +%FT%TZ > "$MARKER"

    # Sources are now in S3; keep local Parquet files and audits.
    rm -f -- "$ARCHIVE" "$FLIGHT_CSV" "$WEATHER_CSV"

    echo "Completed month: $MONTH"
    CURRENT="$NEXT"
done

echo "Backfill completed through August 2026."