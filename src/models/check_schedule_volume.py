import os
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sqlalchemy import URL, bindparam, create_engine, text

from src.features.flight_features import (
    AIRPORT_TIMEZONES,
    add_flight_time_features,
)
from src.models.evaluation import evaluate, save_results
from src.models.historical_delay_rates import (
    AIRPORT_PRIOR_FLIGHTS,
    ARCHIVE_INDEX,
    AVAILABILITY_BUFFER_DAYS,
    RATE_COLUMNS,
    historical_rates,
    load_history,
)
from src.models.train_lightgbm import DATA_DIR, ROOT, load_split
from src.models.tune_threshold import prepare_features
from src.prepare_model_data import SPLITS


MODEL_RUN = "20261118T013557Z_lightgbm_448d18"
COMPARISON_RUN = "20261118T032845943376Z_schedule_volume"
VOLUME_COLUMNS = [
    "origin_departures_prev_60m",
    "origin_departures_near_60m",
]
HOUR_NS = pd.Timedelta(hours=1).value

SCHEDULE_QUERY = text("""
    SELECT flight_id, flight_date, day_of_week, origin,
           scheduled_departure_hhmm
    FROM flights
    WHERE origin IN :airports
      AND flight_date >= :start_date
      AND flight_date < :end_date
""").bindparams(bindparam("airports", expanding=True))


def monthly_counts(connection, start, end):
    schedule = pd.read_sql_query(
        SCHEDULE_QUERY,
        connection,
        params={
            "airports": list(AIRPORT_TIMEZONES),
            "start_date": (start - pd.Timedelta(days=2)).date(),
            "end_date": (end + pd.Timedelta(days=2)).date(),
        },
        parse_dates=["flight_date"],
    )
    schedule = add_flight_time_features(schedule, prediction_minutes=60)
    schedule = schedule.dropna(subset=["scheduled_departure_utc"])

    parts = []
    for _, group in schedule.groupby("origin"):
        times = (
            group["scheduled_departure_utc"]
            .astype("datetime64[ns, UTC]")
            .astype("int64")
            .to_numpy()
        )
        ordered = np.sort(times)
        left = np.searchsorted(ordered, times - HOUR_NS)
        before = np.searchsorted(ordered, times) - left

        # The wider window includes the current flight, so subtract one.
        nearby = np.searchsorted(ordered, times + HOUR_NS) - left - 1
        counts = pd.DataFrame(
            {VOLUME_COLUMNS[0]: before, VOLUME_COLUMNS[1]: nearby},
            index=group["flight_id"].to_numpy(),
            dtype="float32",
        )

        in_month = group["flight_date"].ge(start) & group["flight_date"].lt(end)
        parts.append(counts.loc[in_month.to_numpy()])

    return pd.concat(parts)


def load_experiment_split(connection, split, artifact, history):
    start, end = SPLITS[split]
    months = pd.date_range(start, end, freq="MS", inclusive="left")

    print(f"Building schedule counts and historical rates for {split}...", flush=True)
    counts = pd.concat([
        monthly_counts(connection, month, month + pd.offsets.MonthBegin(1))
        for month in months
    ])

    metadata = pd.read_parquet(
        DATA_DIR / f"{split}.parquet",
        columns=["flight_id", "origin", "carrier", "prediction_time_utc"],
    )
    aligned = counts.reindex(metadata["flight_id"].to_numpy())
    if aligned.isna().any().any():
        raise ValueError(f"Missing schedule counts for some {split} flights.")

    rates = historical_rates(metadata, history)
    print(
        f"{split}: {np.isfinite(rates[:, 0]).mean():.1%} "
        "of flights have an eligible airport history",
        flush=True,
    )

    frame, labels = load_split(split)
    features = prepare_features(frame, artifact, "lightgbm")
    features[VOLUME_COLUMNS] = aligned.to_numpy()
    features[RATE_COLUMNS] = rates

    return features, labels


def main():
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_history_rates"
    artifact = load(ROOT / "data" / "models" / MODEL_RUN / "lightgbm.joblib")

    engine = create_engine(URL.create(
        "postgresql+psycopg2",
        username=os.getenv("DB_USER", "aviation"),
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5433")),
        database=os.getenv("DB_NAME", "aviation_mlops"),
    ))

    try:
        with engine.connect() as connection:
            history = load_history(connection)
            train, y_train = load_experiment_split(
                connection, "train", artifact, history
            )

            model = clone(artifact["model"])
            print(f"Fitting on {len(y_train):,} flights...", flush=True)
            started = perf_counter()
            model.fit(train, y_train)
            fit_seconds = perf_counter() - started
            del train

            validation, labels = load_experiment_split(
                connection, "validation", artifact, history
            )
    finally:
        engine.dispose()

    scores = model.predict_proba(validation)[:, 1]
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    eligible = np.flatnonzero(precision[:-1] >= 0.40)
    metrics = {"history_rates_at_0_5": evaluate(labels, scores, 0.5)}

    if eligible.size:
        best = eligible[np.argmax(recall[eligible])]
        metrics["history_rates_at_min_40_precision"] = evaluate(
            labels, scores, float(thresholds[best])
        )

    availability = (
        history[["month", "available_at"]]
        .drop_duplicates()
        .sort_values("month")
    )
    report_dir = ROOT / "reports" / "experiments" / run_id
    save_results({
        "run_id": run_id,
        "comparison_run": COMPARISON_RUN,
        "added_features": RATE_COLUMNS,
        "parameters": model.get_params(),
        "fit_seconds": fit_seconds,
        "training_rows": len(y_train),
        "validation_rows": len(labels),
        "history_source": ARCHIVE_INDEX,
        "availability_buffer_days": AVAILABILITY_BUFFER_DAYS,
        "airport_prior_flights": AIRPORT_PRIOR_FLIGHTS,
        "available_at_by_month": {
            row.month: row.available_at.isoformat()
            for row in availability.itertuples(index=False)
        },
        "metrics": metrics,
    }, report_dir)

    print(f"Validation AP: {metrics['history_rates_at_0_5']['average_precision']:.5f}")
    operating_point = metrics.get("history_rates_at_min_40_precision")
    if operating_point:
        print(f"Recall at ≥40% precision: {operating_point['recall']:.2%}")
    else:
        print("No threshold reached 40% precision.")
    print(f"Results: {report_dir / 'results.txt'}")


if __name__ == "__main__":
    main()
