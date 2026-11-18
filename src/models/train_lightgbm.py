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
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

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

SEARCH_SPACE = {
    "n_estimators": [200, 300, 600],
    "learning_rate": [0.025, 0.05, 0.10],
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 6, 10],
    "min_child_samples": [50, 100, 250, 500],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 10.0],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
}

SEARCH_ITERATIONS = 20
SEED = 42


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
    started = time.perf_counter()

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_lightgbm_random_"
        + uuid4().hex[:6]
    )

    report_dir = ROOT / "reports" / "experiments" / run_id
    model_dir = ROOT / "data" / "models" / run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)

    print(f"Run: {run_id}", flush=True)
    print("Loading training and validation data...", flush=True)

    preprocessing_start = time.perf_counter()

    train, y_train = load_split("train")
    validation, y_validation = load_split("validation")

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

        validation_values = (
            validation[column]
            .astype("string")
            .fillna("__MISSING__")
        )
        validation_values = validation_values.where(
            validation_values.isin(levels)
        )

        validation[column] = pd.Categorical(
            validation_values,
            categories=levels
        )

    train_rows = len(train)
    validation_rows = len(validation)

    features = pd.concat([train, validation], ignore_index=True)
    labels = np.concatenate([y_train, y_validation])

    del train, validation

    split = PredefinedSplit(
        np.concatenate([
            np.full(train_rows, -1, dtype=np.int8),
            np.zeros(validation_rows, dtype=np.int8),
        ])
    )

    preprocessing_seconds = time.perf_counter() - preprocessing_start

    base_model = LGBMClassifier(
        objective="binary",
        random_state=SEED,
        n_jobs=2,
        subsample_freq=1,
        max_bin=63,
        force_col_wise=True,
        deterministic=True,
        verbosity=-1
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=SEARCH_SPACE,
        n_iter=SEARCH_ITERATIONS,
        random_state=SEED,
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

    print(
        f"Searching {SEARCH_ITERATIONS} configurations sequentially...",
        flush=True
    )

    search_start = time.perf_counter()
    search.fit(features, labels)
    search_seconds = time.perf_counter() - search_start

    candidates = []

    for index, parameters in enumerate(search.cv_results_["params"]):
        candidates.append({
            "configuration": index + 1,
            "parameters": parameters,
            "validation_average_precision": float(
                search.cv_results_[
                    "mean_test_average_precision"
                ][index]
            ),
            "validation_roc_auc": float(
                search.cv_results_["mean_test_roc_auc"][index]
            ),
            "validation_log_loss": float(
                -search.cv_results_["mean_test_neg_log_loss"][index]
            ),
            "fit_seconds": float(
                search.cv_results_["mean_fit_time"][index]
            ),
            "score_seconds": float(
                search.cv_results_["mean_score_time"][index]
            ),
        })

    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["validation_average_precision"],
            row["validation_log_loss"]
        )
    )

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
    ]).to_csv(report_dir / "search_results.csv", index=False)

    best_parameters = ranked[0]["parameters"]
    print(f"Best parameters: {best_parameters}", flush=True)

    # Refit on training rows only, preserving validation separation.
    model = clone(base_model).set_params(**best_parameters)

    fit_start = time.perf_counter()
    model.fit(features.iloc[:train_rows], y_train)
    fit_seconds = time.perf_counter() - fit_start

    prediction_start = time.perf_counter()
    probabilities = model.predict_proba(
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
            "model": model,
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
        "search_method": "RandomizedSearchCV",
        "search_seed": SEED,
        "search_space": SEARCH_SPACE,
        "configurations_evaluated": len(candidates),
        "selection_rule": (
            "Highest validation AP; lowest log loss breaks ties."
        ),
        "selected_parameters": model.get_params(),
        "candidate_results": ranked,
        "training_rows": train_rows,
        "validation_rows": validation_rows,
        "training_period": "2024-01-01 to 2026-01-01 exclusive",
        "validation_period": "2026-01-01 to 2026-05-01 exclusive",
        "features": FEATURES,
        "preprocessing": (
            "Training-only categorical vocabulary; unknown categories "
            "become missing; float32 numeric values; native missing handling."
        ),
        "timings_seconds": {
            "loading_and_preprocessing": preprocessing_seconds,
            "randomized_search": search_seconds,
            "selected_model_refit": fit_seconds,
            "validation_prediction": prediction_seconds,
            "total_before_report_write": time.perf_counter() - started,
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
    print(f"Search comparison: {report_dir / 'search_results.csv'}")


if __name__ == "__main__":
    main()
