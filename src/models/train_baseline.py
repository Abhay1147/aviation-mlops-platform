import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.evaluation import evaluate, save_results
from src.prepare_model_data import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "model"

CATEGORY_COLUMNS = [
    *CATEGORICAL_FEATURES,
    "day_of_week",
    "flight_month",
    "scheduled_departure_hour",
]

NUMBER_COLUMNS = [
    column for column in NUMERIC_FEATURES
    if column not in CATEGORY_COLUMNS
] + BOOLEAN_FEATURES

COLUMNS = CATEGORY_COLUMNS + NUMBER_COLUMNS + [TARGET]


def read_batches(split, batch_size):
    source = pq.ParquetFile(DATA_DIR / f"{split}.parquet")

    for batch in source.iter_batches(
        batch_size=batch_size,
        columns=COLUMNS
    ):
        frame = batch.to_pandas()

        if frame[TARGET].isna().any():
            raise ValueError(f"Missing target in {split}.")

        yield frame


def category_values(frame):
    return (
        frame[CATEGORY_COLUMNS]
        .astype("string")
        .fillna("__MISSING__")
        .to_numpy(dtype=str)
    )


def numeric_values(frame):
    values = frame[NUMBER_COLUMNS].to_numpy(
        dtype=np.float64,
        na_value=np.nan
    )

    if np.isinf(values).any():
        raise ValueError("Infinite numeric feature encountered.")

    missing = np.isnan(values).astype(np.float64)
    filled = np.nan_to_num(values, nan=0.0)

    return filled, missing


def transform(frame, encoder, scaler):
    values, missing = numeric_values(frame)

    return sparse.hstack(
        [
            encoder.transform(category_values(frame)),
            sparse.csr_matrix(scaler.transform(values)),
            sparse.csr_matrix(missing),
        ],
        format="csr"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate aviation baseline models."
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if min(
        args.epochs,
        args.alpha,
        args.learning_rate,
        args.batch_size
    ) <= 0:
        parser.error("Epochs, alpha, learning rate and batch size must be positive.")

    return args


def main():
    args = parse_args()
    total_start = time.perf_counter()

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_baseline_"
        + uuid4().hex[:6]
    )

    print(f"Run: {run_id}", flush=True)

    # First pass: learn preprocessing from training data only.
    preprocessing_start = time.perf_counter()
    scaler = StandardScaler()
    vocabulary = [set() for _ in CATEGORY_COLUMNS]
    train_rows = 0
    train_delayed = 0

    print("Learning training preprocessing...", flush=True)

    for frame in read_batches("train", args.batch_size):
        values, _ = numeric_values(frame)
        scaler.partial_fit(values)

        categorical = category_values(frame)
        for index, known in enumerate(vocabulary):
            known.update(categorical[:, index])

        train_rows += len(frame)
        train_delayed += int(frame[TARGET].sum())

    if not train_rows:
        raise ValueError("Training data is empty.")

    known_categories = [
        sorted(values) for values in vocabulary
    ]

    encoder = OneHotEncoder(
        categories=known_categories,
        handle_unknown="ignore",
        sparse_output=True
    )

    # Explicit categories already contain every training category.
    encoder.fit(
        np.array(
            [[values[0] for values in known_categories]],
            dtype=str
        )
    )

    preprocessing_seconds = time.perf_counter() - preprocessing_start
    training_delay_rate = train_delayed / train_rows

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        learning_rate="constant",
        eta0=args.learning_rate,
        average=True,
        shuffle=True,
        random_state=args.seed
    )

    training_start = time.perf_counter()
    epoch_seconds = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        processed = 0

        for frame in read_batches("train", args.batch_size):
            features = transform(frame, encoder, scaler)
            labels = frame[TARGET].to_numpy(dtype=np.int8)

            model.partial_fit(
                features,
                labels,
                classes=np.array([0, 1])
            )
            processed += len(frame)

        elapsed = time.perf_counter() - epoch_start
        epoch_seconds.append(elapsed)

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"{processed:,} rows | {elapsed:.1f} seconds",
            flush=True
        )

    training_seconds = time.perf_counter() - training_start

    prediction_start = time.perf_counter()
    label_parts = []
    probability_parts = []

    for frame in read_batches("validation", args.batch_size):
        features = transform(frame, encoder, scaler)
        label_parts.append(
            frame[TARGET].to_numpy(dtype=np.int8)
        )
        probability_parts.append(
            model.predict_proba(features)[:, 1]
        )

    labels = np.concatenate(label_parts)
    probabilities = np.concatenate(probability_parts)
    prediction_seconds = time.perf_counter() - prediction_start

    evaluation_start = time.perf_counter()

    metrics = {
        "constant_probability": evaluate(
            labels,
            np.full(len(labels), training_delay_rate)
        ),
        "logistic_regression": evaluate(
            labels,
            probabilities
        ),
    }

    evaluation_seconds = time.perf_counter() - evaluation_start

    model_dir = ROOT / "data" / "models" / run_id
    model_dir.mkdir(parents=True, exist_ok=False)
    model_path = model_dir / "baseline.joblib"

    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "scaler": scaler,
            "category_columns": CATEGORY_COLUMNS,
            "number_columns": NUMBER_COLUMNS,
            "constant_probability": training_delay_rate,
            "threshold": 0.5,
            "numeric_missing_fill": 0.0,
            "numeric_missing_indicators": True,
        },
        model_path
    )

    results = {
        "run_id": run_id,
        "git_commit": os.getenv("GIT_COMMIT", "not supplied"),
        "model": "logistic regression trained using SGD",
        "parameters": model.get_params(),
        "run_arguments": vars(args),
        "training_period": "2024-01-01 to 2026-01-01 exclusive",
        "validation_period": "2026-01-01 to 2026-05-01 exclusive",
        "training_rows": train_rows,
        "validation_rows": len(labels),
        "training_delay_rate": training_delay_rate,
        "category_columns": CATEGORY_COLUMNS,
        "number_columns": NUMBER_COLUMNS,
        "preprocessing": {
            "categorical": "one-hot; unknown categories ignored",
            "numeric": "zero fill, standard scaling, missing indicators",
            "fitted_on": "training only",
        },
        "timings_seconds": {
            "preprocessing": preprocessing_seconds,
            "training_including_batch_transforms": training_seconds,
            "each_epoch": epoch_seconds,
            "validation_prediction_including_transforms": prediction_seconds,
            "metric_calculation": evaluation_seconds,
            "total_before_report_write": time.perf_counter() - total_start,
        },
        "environment": {
            "python": platform.python_version(),
            "visible_cpu_count": os.cpu_count(),
            "packages": {
                package: version(package)
                for package in [
                    "scikit-learn", "pandas", "numpy",
                    "scipy", "pyarrow", "joblib"
                ]
            },
        },
        "metrics": metrics,
        "model_path": str(model_path),
        "test_set_used": False,
    }

    report_dir = ROOT / "reports" / "experiments" / run_id
    save_results(results, report_dir)

    print(json.dumps(metrics, indent=2))
    print(f"Model saved: {model_path}")
    print(f"Readable results: {report_dir / 'results.txt'}")


if __name__ == "__main__":
    main()
