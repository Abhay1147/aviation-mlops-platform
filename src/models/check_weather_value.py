import json
from datetime import datetime, timezone
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve

from src.models.evaluation import evaluate
from src.models.train_lightgbm import ROOT, TARGET, load_split
from src.models.tune_threshold import prepare_features

FULL_RUN = "20261118T013557Z_lightgbm_448d18"

SCHEDULE_FEATURES = [
    "carrier",
    "origin",
    "destination",
    "day_of_week",
    "flight_month",
    "scheduled_departure_hour",
    "scheduled_elapsed_minutes",
    "distance_miles",
    "invalid_scheduled_duration",
]


def main():
    artifact = joblib.load(
        ROOT / "data" / "models" / FULL_RUN / "lightgbm.joblib"
    )
    train, y_train = load_split("train")

    # Sample training rows for an exploratory correlation matrix.
    numeric = train.select_dtypes(include="number")
    sample = numeric.sample(n=min(200_000, len(numeric)), random_state=42)
    sample[TARGET] = y_train[sample.index.to_numpy()]
    correlations = sample.corr()
    del numeric, sample

    schedule_artifact = {**artifact, "features": SCHEDULE_FEATURES}
    schedule_train = prepare_features(train, schedule_artifact, "lightgbm")
    del train

    model = clone(artifact["model"])
    started = perf_counter()
    model.fit(schedule_train, y_train)
    fit_seconds = perf_counter() - started
    del schedule_train

    validation, labels = load_split("validation")
    features = prepare_features(validation, schedule_artifact, "lightgbm")
    scores = model.predict_proba(features)[:, 1]

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    eligible = np.flatnonzero(precision[:-1] >= 0.40)
    at_precision_floor = None
    if eligible.size:
        best = eligible[np.argmax(recall[eligible])]
        at_precision_floor = evaluate(labels, scores, float(thresholds[best]))

    results = {
        "full_feature_run": FULL_RUN,
        "schedule_features": SCHEDULE_FEATURES,
        "parameters": model.get_params(),
        "fit_seconds": fit_seconds,
        "schedule_only_at_0_5": evaluate(labels, scores, 0.5),
        "schedule_only_at_min_40_precision": at_precision_floor,
        "full_feature_validation_ap": 0.38938190881080603,
        "full_feature_recall_at_min_40_precision_rounded": 0.4021,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "reports" / "experiments" / f"{stamp}_weather_ablation"
    output.mkdir(parents=True, exist_ok=False)
    correlations.to_csv(output / "training_correlations.csv")
    summary = json.dumps(results, indent=2)
    (output / "results.txt").write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
