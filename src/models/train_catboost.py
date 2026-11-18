import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve

from src.models.evaluation import evaluate, save_results
from src.prepare_model_data import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "model"
NUMERIC_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES
FEATURES = CATEGORICAL_FEATURES + NUMERIC_COLUMNS

PARAMETERS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.1,
    "loss_function": "Logloss",
    "max_ctr_complexity": 1,
    "thread_count": 2,
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": 50,
}


def load_split(name):
    frame = pd.read_parquet(
        DATA_DIR / f"{name}.parquet",
        columns=FEATURES + [TARGET],
    )
    if frame.empty or frame[TARGET].isna().any():
        raise ValueError(f"Empty split or missing labels: {name}")

    labels = frame.pop(TARGET).to_numpy(dtype=np.int8)

    for column in CATEGORICAL_FEATURES:
        frame[column] = (
            frame[column].astype("string").fillna("__MISSING__").astype(object)
        )

    for column in NUMERIC_COLUMNS:
        frame[column] = frame[column].astype("float32")

    return frame[FEATURES], labels


def main():
    started = perf_counter()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_catboost_{uuid4().hex[:6]}"
    print(f"Run: {run_id}", flush=True)

    train, y_train = load_split("train")
    model = CatBoostClassifier(**PARAMETERS)

    fit_started = perf_counter()
    model.fit(train, y_train, cat_features=CATEGORICAL_FEATURES)
    fit_seconds = perf_counter() - fit_started
    del train

    validation, labels = load_split("validation")
    probabilities = model.predict_proba(validation, thread_count=2)[:, 1]

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    precision, recall = precision[:-1], recall[:-1]
    feasible = np.flatnonzero((precision >= 0.40) & (recall >= 0.50))

    if feasible.size:
        chosen = feasible[np.argmax(recall[feasible])]
    else:
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        chosen = np.argmax(f1)

    threshold = float(thresholds[chosen])
    metrics = {
        "catboost_at_0_5": evaluate(labels, probabilities, threshold=0.5),
        "validation_selected_threshold": evaluate(
            labels, probabilities, threshold=threshold
        ),
    }

    model_dir = ROOT / "data" / "models" / run_id
    model_dir.mkdir(parents=True, exist_ok=False)
    model_path = model_dir / "catboost.joblib"
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "category_columns": CATEGORICAL_FEATURES,
            "numeric_dtype": "float32",
            "category_missing_fill": "__MISSING__",
            "threshold": threshold,
        },
        model_path,
    )

    report_dir = ROOT / "reports" / "experiments" / run_id
    selected = metrics["validation_selected_threshold"]
    save_results(
        {
            "run_id": run_id,
            "git_commit": os.getenv("GIT_COMMIT", "not supplied"),
            "model": "CatBoost",
            "catboost_version": version("catboost"),
            "parameters": PARAMETERS,
            "training_rows": len(y_train),
            "validation_rows": len(labels),
            "fit_seconds": fit_seconds,
            "total_seconds_before_report": perf_counter() - started,
            "threshold_selection": (
                "Highest recall meeting both targets"
                if feasible.size else "Exploratory maximum validation F1"
            ),
            "targets_met": bool(feasible.size),
            "metrics": metrics,
            "model_path": str(model_path),
            "test_set_used": False,
        },
        report_dir,
    )

    print(f"Validation AP: {selected['average_precision']:.5f}")
    print(f"Targets met: {bool(feasible.size)}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {selected['precision']:.2%} | Recall: {selected['recall']:.2%}")
    print(f"Fit time: {fit_seconds:.1f}s")
    print(f"Results: {report_dir / 'results.txt'}")


if __name__ == "__main__":
    main()
