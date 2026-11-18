import os
from datetime import datetime, timezone
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sqlalchemy import URL, bindparam, create_engine, text

from src.features.flight_features import (
    AIRPORT_TIMEZONES,
    add_flight_time_features,
)
from src.models.evaluation import evaluate, save_results
from src.models.train_lightgbm import DATA_DIR, ROOT, load_split
from src.models.tune_threshold import prepare_features
from src.prepare_model_data import SPLITS


BASE_RUN = "20261118T013557Z_lightgbm_448d18"
VOLUME_COLUMNS = ["origin_departures_prev_60m", "origin_departures_near_60m"]
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
        # The wider window includes the flight itself, so subtract one.
        nearby = np.searchsorted(ordered, times + HOUR_NS) - left - 1

        counts = pd.DataFrame(
            {VOLUME_COLUMNS[0]: before, VOLUME_COLUMNS[1]: nearby},
            index=group["flight_id"].to_numpy(),
            dtype="float32",
        )
        # Neighboring dates contribute to counts but are emitted in their own month.
        in_month = group["flight_date"].ge(start) & group["flight_date"].lt(end)
        parts.append(counts.loc[in_month.to_numpy()])

    return pd.concat(parts)


def load_with_volume(connection, split, artifact):
    start, end = SPLITS[split]
    months = pd.date_range(start, end, freq="MS", inclusive="left")

    print(f"Building schedule counts for {split}...", flush=True)
    counts = pd.concat([
        monthly_counts(connection, month, month + pd.offsets.MonthBegin(1))
        for month in months
    ])

    frame, labels = load_split(split)
    ids = pd.read_parquet(
        DATA_DIR / f"{split}.parquet", columns=["flight_id"]
    )["flight_id"]
    aligned = counts.reindex(ids.to_numpy())

    if aligned.isna().any().any():
        raise ValueError(f"Missing schedule counts for some {split} flights.")

    features = prepare_features(frame, artifact, "lightgbm")
    for column in VOLUME_COLUMNS:
        features[column] = aligned[column].to_numpy()

    return features, labels


def main():
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_schedule_volume"
    artifact = joblib.load(
        ROOT / "data" / "models" / BASE_RUN / "lightgbm.joblib"
    )

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
            train, y_train = load_with_volume(connection, "train", artifact)
            model = clone(artifact["model"])

            print(f"Fitting on {len(y_train):,} flights...", flush=True)
            started = perf_counter()
            model.fit(train, y_train)
            fit_seconds = perf_counter() - started
            del train

            validation, labels = load_with_volume(
                connection, "validation", artifact
            )
    finally:
        engine.dispose()

    scores = model.predict_proba(validation)[:, 1]
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    eligible = np.flatnonzero(precision[:-1] >= 0.40)
    metrics = {"schedule_volume_at_0_5": evaluate(labels, scores, 0.5)}

    if eligible.size:
        best = eligible[np.argmax(recall[eligible])]
        metrics["schedule_volume_at_min_40_precision"] = evaluate(
            labels, scores, float(thresholds[best])
        )

    report_dir = ROOT / "reports" / "experiments" / run_id
    save_results({
        "run_id": run_id,
        "comparison_run": BASE_RUN,
        "added_features": VOLUME_COLUMNS,
        "parameters": model.get_params(),
        "fit_seconds": fit_seconds,
        "training_rows": len(y_train),
        "validation_rows": len(labels),
        "metrics": metrics,
    }, report_dir)

    print(f"Validation AP: {metrics['schedule_volume_at_0_5']['average_precision']:.5f}")
    operating_point = metrics.get("schedule_volume_at_min_40_precision")
    if operating_point:
        print(f"Recall at ≥40% precision: {operating_point['recall']:.2%}")
    else:
        print("No threshold reached 40% precision.")
    print(f"Results: {report_dir / 'results.txt'}")


if __name__ == "__main__":
    main()
