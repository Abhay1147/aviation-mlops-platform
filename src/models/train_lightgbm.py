import gc
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
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from src.models.evaluation import evaluate, save_results
from src.prepare_model_data import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "model"

CATEGORY_COLUMNS = CATEGORICAL_FEATURES
NUMBER_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES
FEATURES = CATEGORY_COLUMNS + NUMBER_COLUMNS

PARAMETER_GRID = {
    "n_estimators": [300, 600],
    "num_leaves": [15, 31],
    "learning_rate": [0.05, 0.10],
}


def load_split(name):
    frame = pd.read_parquet(
        DATA_DIR / f"{name}.parquet",
        columns=FEATURES + [TARGET]
    )

    if frame.empty or frame[TARGET].isna().any():
        raise ValueError(f"Empty data or missing labels in {name}.")

    labels = frame.pop(TARGET).to_numpy(dtype=np.int8)

    for column in NUMBER_COLUMNS:
        frame[column] = frame[column].astype("float32")

        if np.isinf(frame[column].to_numpy()).any():
            raise ValueError(f"Infinite values in {name}: {column}")

    return frame, labels


def main():
    total_start = time.perf_counter()
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_lightgbm_"
        + uuid4().hex[:6]
    )

    report_dir = ROOT / "reports" / "experiments" / run_id
    model_dir = ROOT / "data" / "models" / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)

    print(f"Run: {run_id}", flush=True)
    print("Loading training and validation features...", flush=True)

    preprocessing_start = time.perf_counter()

    train, y_train = load_split("train")
    validation, y_validation = load_split("validation")

    # Learn category vocabularies from training only.
    category_levels = {}

    for column in CATEGORY_COLUMNS:
        train_values = (
            train[column].astype("string").fillna("__MISSING__")
        )
        levels = sorted(train_values.unique().tolist())
        category_levels[column] = levels

        train[column] = pd.Categorical(
            train_values,
            categories=levels
        )
        validation[column] = pd.Categorical(
            validation[column]
            .astype("string")
            .fillna("__MISSING__"),
            categories=levels
        )

    train_rows = len(train)
    validation_rows = len(validation)

    features = pd.concat(
        [train, validation],
        ignore_index=True
    )
    labels = np.concatenate([y_train, y_validation])

    del train, validation
    gc.collect()

    # -1 = always training; 0 = validation for the single split.
    fold_assignment = np.concatenate([
        np.full(train_rows, -1, dtype=np.int8),
        np.zeros(validation_rows, dtype=np.int8),
    ])
    split = PredefinedSplit(fold_assignment)

    preprocessing_seconds = time.perf_counter() - preprocessing_start

    base_model = LGBMClassifier(
        objective="binary",
        random_state=42,
        n_jobs=2,
        min_child_samples=100,
        reg_lambda=1.0,
        max_bin=63,
        force_col_wise=True,
        deterministic=True,
        verbosity=-1
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAMETER_GRID,
        scoring={
            "average_precision": "average_precision",
            "roc_auc": "roc_auc",
            "neg_log_loss": "neg_log_loss",
        },
        cv=split,
        refit=False,
        n_jobs=1,
        pre_dispatch=1,
        return_train_score=False,
        error_score="raise",
        verbose=2
    )

    print("Starting eight sequential configurations...", flush=True)
    search_start = time.perf_counter()
    search.fit(features, labels)
    search_seconds = time.perf_counter() - search_start

    cv_results = search.cv_results_
    candidates = []

    for index, parameters in enumerate(cv_results["params"]):
        candidates.append({
            "configuration": index + 1,
            "parameters": parameters,
            "validation_average_precision": float(
                cv_results["mean_test_average_precision"][index]
            ),
            "validation_roc_auc": float(
                cv_results["mean_test_roc_auc"][index]
            ),
            "validation_log_loss": float(
                -cv_results["mean_test_neg_log_loss"][index]
            ),
            "fit_seconds": float(
                cv_results["mean_fit_time"][index]
            ),
            "score_seconds": float(
                cv_results["mean_score_time"][index]
            ),
        })

    # Primary: highest AP. Tie-breaker: lowest log loss.
    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["validation_average_precision"],
            row["validation_log_loss"]
        )
    )
    best_parameters = ranked[0]["parameters"]

    pd.DataFrame([
        {
            **row["parameters"],
            **{
                key: value
                for key, value in row.items()
                if key != "parameters"
            },
        }
        for row in ranked
    ]).to_csv(report_dir / "grid_results.csv", index=False)

    print(f"Best parameters: {best_parameters}", flush=True)

    # GridSearchCV with refit=False doesn't retain a fitted winner.
    # Fit the selected configuration again, on training rows only.
    best_model = clone(base_model).set_params(**best_parameters)

    final_fit_start = time.perf_counter()
    best_model.fit(
        features.iloc[:train_rows],
        y_train
    )
    final_fit_seconds = time.perf_counter() - final_fit_start

    prediction_start = time.perf_counter()
    probabilities = best_model.predict_proba(
        features.iloc[train_rows:]
    )[:, 1]
    prediction_seconds = time.perf_counter() - prediction_start

    metrics = {
        "lightgbm_at_0_5": evaluate(
            y_validation,
            probabilities,
            threshold=0.5
        ),
    }

    model_path = model_dir / "lightgbm.joblib"
    joblib.dump(
        {
            "model": best_model,
            "features": FEATURES,
            "category_columns": CATEGORY_COLUMNS,
            "category_levels": category_levels,
            "numeric_dtype": "float32",
            "threshold": 0.5,
        },
        model_path
    )

    results = {
        "run_id": run_id,
        "git_commit": os.getenv("GIT_COMMIT", "not supplied"),
        "model": "LightGBM",
        "selection_rule": (
            "Highest validation average precision; "
            "lowest log loss breaks ties."
        ),
        "parameter_grid": PARAMETER_GRID,
        "selected_parameters": best_model.get_params(),
        "configurations_evaluated": len(candidates),
        "candidate_results": ranked,
        "training_rows": train_rows,
        "validation_rows": validation_rows,
        "training_period": "2024-01-01 to 2026-01-01 exclusive",
        "validation_period": "2026-01-01 to 2026-05-01 exclusive",
        "features": FEATURES,
        "preprocessing": (
            "Training-only category vocabulary; float32 numeric values; "
            "LightGBM handles missing numeric and unknown category values."
        ),
        "timings_seconds": {
            "loading_and_preprocessing": preprocessing_seconds,
            "grid_search": search_seconds,
            "selected_model_refit": final_fit_seconds,
            "validation_prediction": prediction_seconds,
            "total_before_report_write": time.perf_counter() - total_start,
        },
        "environment": {
            "python": platform.python_version(),
            "visible_cpu_count": os.cpu_count(),
            "packages": {
                package: version(package)
                for package in [
                    "lightgbm", "scikit-learn", "pandas",
                    "numpy", "pyarrow", "joblib"
                ]
            },
        },
        "metrics": metrics,
        "model_path": str(model_path),
        "test_set_used": False,
    }

    save_results(results, report_dir)

    print(json.dumps(metrics, indent=2))
    print(f"Model saved: {model_path}")
    print(f"Results: {report_dir / 'results.txt'}")
    print(f"Grid comparison: {report_dir / 'grid_results.csv'}")


if __name__ == "__main__":
    main()
