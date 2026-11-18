import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

from src.models.evaluation import evaluate, save_results
from src.models.train_lightgbm import (
    CATEGORY_COLUMNS,
    FEATURES,
    ROOT,
    load_split,
)

PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "enable_categorical": True,
    "max_bin": 64,
    "n_jobs": 2,
    "random_state": 42,
}


def main():
    started = perf_counter()
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_xgboost_{uuid4().hex[:6]}"
    print(f"Run: {run_id}", flush=True)

    train, y_train = load_split("train")
    category_levels = {}

    for column in CATEGORY_COLUMNS:
        train[column] = train[column].astype("category")
        category_levels[column] = train[column].cat.categories.tolist()

    model = XGBClassifier(**PARAMETERS)
    print(f"Fitting on {len(y_train):,} rows...", flush=True)
    fit_started = perf_counter()
    model.fit(train, y_train)
    fit_seconds = perf_counter() - fit_started
    del train

    validation, labels = load_split("validation")
    for column, levels in category_levels.items():
        values = validation[column]
        validation[column] = pd.Categorical(
            values.where(values.isin(levels)),
            categories=levels,
        )

    probabilities = model.predict_proba(validation)[:, 1]
    metrics = {
        "xgboost_at_0_5": evaluate(labels, probabilities, threshold=0.5)
    }

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = np.flatnonzero(precision[:-1] >= 0.40)
    recall_at_40_precision = None

    if eligible.size:
        best = eligible[np.argmax(recall[eligible])]
        recall_at_40_precision = float(recall[best])
        metrics["xgboost_at_min_40_precision"] = evaluate(
            labels, probabilities, threshold=float(thresholds[best])
        )

    model_dir = ROOT / "data" / "models" / run_id
    model_dir.mkdir(parents=True, exist_ok=False)
    model_path = model_dir / "xgboost.json"
    model.save_model(str(model_path))

    report_dir = ROOT / "reports" / "experiments" / run_id
    results = {
        "run_id": run_id,
        "git_commit": os.getenv("GIT_COMMIT", "not supplied"),
        "model": "XGBoost",
        "xgboost_version": version("xgboost"),
        "parameters": PARAMETERS,
        "features": FEATURES,
        "category_levels": category_levels,
        "training_rows": len(y_train),
        "validation_rows": len(labels),
        "fit_seconds": fit_seconds,
        "total_seconds_before_report": perf_counter() - started,
        "recall_at_min_40_precision": recall_at_40_precision,
        "targets_met": (
            recall_at_40_precision is not None and recall_at_40_precision >= 0.50
        ),
        "metrics": metrics,
        "model_path": str(model_path),
        "test_set_used": False,
    }
    save_results(results, report_dir)

    print(json.dumps(metrics, indent=2))
    print(f"Targets met: {results['targets_met']}")
    print(f"Fit time: {fit_seconds:.1f}s")
    print(f"Results: {report_dir / 'results.txt'}")


if __name__ == "__main__":
    main()
