import json
from datetime import datetime, timezone
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sqlalchemy import URL, create_engine
from xgboost import XGBClassifier

from src.models.evaluation import evaluate
from src.models.train_lightgbm import ROOT
from src.models.volume_features import BASE_RUN, load_features

import os


def load_data():
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
            train, y_train, _ = load_features(connection, "train", artifact)
            validation, y_val, dates = load_features(
                connection, "validation", artifact
            )
    finally:
        engine.dispose()

    months = pd.to_datetime(dates).dt.strftime("%Y-%m").to_numpy()
    return train, y_train, validation, y_val, months, artifact["category_columns"]


def make_model(name, depth, trees, rate):
    if name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            enable_categorical=True,
            max_depth=depth,
            n_estimators=trees,
            learning_rate=rate,
            max_bin=63,
            n_jobs=2,
            random_state=42,
        )

    return CatBoostClassifier(
        loss_function="Logloss",
        depth=depth,
        iterations=trees,
        learning_rate=rate,
        max_ctr_complexity=1,
        thread_count=2,
        random_seed=42,
        allow_writing_files=False,
        verbose=False,
    )


def score_validation(labels, scores, months):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    eligible = np.flatnonzero(precision[:-1] >= 0.40)
    if not eligible.size:
        return evaluate(labels, scores, 0.5), None, {}

    best = eligible[np.argmax(recall[eligible])]
    threshold = float(thresholds[best])
    monthly = {}

    for month in sorted(set(months)):
        mask = months == month
        metrics = evaluate(labels[mask], scores[mask], threshold)
        monthly[month] = {
            "rows": int(mask.sum()),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "average_precision": metrics["average_precision"],
        }

    return evaluate(labels, scores, threshold), threshold, monthly


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "reports" / "experiments" / f"{stamp}_challenger_search"
    output.mkdir(parents=True, exist_ok=False)

    train, y_train, validation, y_val, months, categories = load_data()
    sample = np.random.default_rng(42).choice(
        len(y_train), size=min(200_000, len(y_train)), replace=False
    )

    for name in ("xgboost", "catboost"):
        if name == "catboost":
            for frame in (train, validation):
                for column in categories:
                    frame[column] = (
                        frame[column].astype("string").fillna("__MISSING__")
                        .astype(object)
                    )

        for depth in (4, 6, 8):
            for trees, rate in ((300, 0.10), (600, 0.05)):
                trial = f"{name}_depth{depth}_trees{trees}"
                print(f"Fitting {trial}...", flush=True)
                model = make_model(name, depth, trees, rate)

                started = perf_counter()
                if name == "catboost":
                    model.fit(train, y_train, cat_features=categories)
                else:
                    model.fit(train, y_train)
                fit_seconds = perf_counter() - started

                train_scores = model.predict_proba(train.iloc[sample])[:, 1]
                scores = model.predict_proba(validation)[:, 1].astype(np.float64)
                metrics, threshold, monthly = score_validation(y_val, scores, months)
                recall_at_floor = metrics["recall"] if threshold is not None else None

                result = {
                    "trial": trial,
                    "parameters": model.get_params(),
                    "fit_seconds": fit_seconds,
                    "train_sample_ap": float(
                        average_precision_score(y_train[sample], train_scores)
                    ),
                    "validation": metrics,
                    "threshold_at_40_precision": threshold,
                    "recall_at_40_precision": recall_at_floor,
                    "validation_by_month": monthly,
                }
                (output / f"{trial}.json").write_text(
                    json.dumps(result, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )

                recall_text = (
                    f"{recall_at_floor:.2%}"
                    if recall_at_floor is not None else "unavailable"
                )
                print(
                    f"{trial} | validation AP={metrics['average_precision']:.5f} "
                    f"| recall at 40% precision={recall_text} "
                    f"| fit={fit_seconds:.1f}s",
                    flush=True,
                )
                del model

    print(f"Reports: {output}", flush=True)


if __name__ == "__main__":
    main()