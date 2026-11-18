import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.models.evaluation import evaluate, save_results
from src.models.train_baseline import (
    ROOT,
    TARGET,
    read_batches,
    transform,
)


def prepare_features(frame, artifact, model_type):
    if model_type == "baseline":
        return transform(
            frame,
            artifact["encoder"],
            artifact["scaler"]
        )

    features = frame[artifact["features"]].copy()

    for column in artifact["category_columns"]:
        levels = artifact["category_levels"][column]

        values = (
            features[column]
            .astype("string")
            .fillna("__MISSING__")
        )

        # Unknown categories become missing without extending
        # the category vocabulary learned during training.
        values = values.where(values.isin(levels))

        features[column] = pd.Categorical(
            values,
            categories=levels
        )

    for column in artifact["features"]:
        if column not in artifact["category_columns"]:
            features[column] = features[column].astype("float32")

    return features


def threshold_counts(labels, probabilities, threshold):
    actual = labels == 1
    predicted = probabilities >= threshold

    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flights_flagged": tp + fp,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select a threshold using validation data only."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--model",
        choices=["baseline", "lightgbm"],
        default="baseline"
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.40
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.50
    )

    args = parser.parse_args()

    if not (
        0 < args.min_precision <= 1
        and 0 < args.min_recall <= 1
    ):
        parser.error(
            "Precision and recall targets must be in (0, 1]."
        )

    return args


def main():
    args = parse_args()
    started = time.perf_counter()

    model_dir = ROOT / "data" / "models" / args.run_id
    original_path = model_dir / f"{args.model}.joblib"

    artifact = joblib.load(original_path)

    label_parts = []
    probability_parts = []

    print("Predicting validation flights...", flush=True)

    for frame in read_batches("validation", batch_size=50000):
        features = prepare_features(
            frame,
            artifact,
            args.model
        )

        label_parts.append(
            frame[TARGET].to_numpy(dtype=np.int8)
        )
        probability_parts.append(
            artifact["model"].predict_proba(features)[:, 1]
        )

    if not label_parts:
        raise ValueError("Validation data is empty.")

    labels = np.concatenate(label_parts)
    probabilities = np.concatenate(probability_parts)

    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced invalid probabilities.")

    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities
    )

    # The last precision/recall pair has no corresponding threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    feasible = np.flatnonzero(
        (precision >= args.min_precision)
        & (recall >= args.min_recall)
    )

    target_met = len(feasible) > 0

    if target_met:
        chosen_index = feasible[np.argmax(recall[feasible])]
        selection_rule = (
            "Highest recall satisfying both validation targets."
        )
    else:
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0
        )
        chosen_index = int(np.argmax(f1))
        selection_rule = (
            "Targets not achievable; exploratory maximum validation F1."
        )

    chosen_threshold = float(thresholds[chosen_index])

    comparison_thresholds = sorted(set([
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        chosen_threshold,
    ]))

    comparison = [
        threshold_counts(labels, probabilities, threshold)
        for threshold in comparison_thresholds
    ]

    print("\nThreshold  Precision  Recall     F1       Flagged")

    for row in comparison:
        print(
            f"{row['threshold']:9.4f}  "
            f"{row['precision']:9.2%}  "
            f"{row['recall']:8.2%}  "
            f"{row['f1']:8.4f}  "
            f"{row['flights_flagged']:10,}"
        )

    metrics = {
        "original_threshold": evaluate(
            labels,
            probabilities,
            threshold=0.5
        ),
        "validation_selected_threshold": evaluate(
            labels,
            probabilities,
            threshold=chosen_threshold
        ),
    }

    artifact["threshold"] = chosen_threshold
    artifact["threshold_selection"] = {
        "split": "validation",
        "rule": selection_rule,
        "target_met": bool(target_met),
        "minimum_precision": args.min_precision,
        "minimum_recall": args.min_recall,
    }

    adjusted_path = (
        model_dir / f"{args.model}_thresholded.joblib"
    )
    joblib.dump(artifact, adjusted_path)

    results = {
        "source_run_id": args.run_id,
        "model_type": args.model,
        "source_model": str(original_path),
        "adjusted_model": str(adjusted_path),
        "model_retrained": False,
        "validation_rows": len(labels),
        "selected_threshold": chosen_threshold,
        "selection_rule": selection_rule,
        "target_met": bool(target_met),
        "minimum_precision": args.min_precision,
        "minimum_recall": args.min_recall,
        "threshold_comparison": comparison,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "test_set_used": False,
    }

    directory = (
        ROOT / "reports" / "experiments"
        / args.run_id / "threshold_analysis"
    )
    save_results(results, directory)

    print(f"\nTargets met: {target_met}")
    print(f"Selected threshold: {chosen_threshold:.6f}")
    print(selection_rule)
    print(json.dumps(metrics, indent=2))
    print(f"Results: {directory / 'results.txt'}")


if __name__ == "__main__":
    main()
