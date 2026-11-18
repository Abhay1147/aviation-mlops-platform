import json

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(labels, probabilities, threshold=0.5):
    predictions = probabilities >= threshold

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "log_loss": float(
            log_loss(labels, probabilities, labels=[0, 1])
        ),
        "brier_score": float(
            brier_score_loss(labels, probabilities)
        ),
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(labels, predictions, zero_division=0)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_results(results, directory):
    directory.mkdir(parents=True, exist_ok=True)

    formatted = json.dumps(
        results,
        indent=2,
        allow_nan=False
    )

    (directory / "results.json").write_text(
        formatted,
        encoding="utf-8"
    )

    lines = [
        "AVIATION MODEL EXPERIMENT",
        "Positive class: departure delayed at least 15 minutes",
        "Evaluation split: validation",
        "Confusion matrix: rows=actual, columns=predicted",
        "Class order: [not delayed, delayed]",
        "",
    ]

    for name, metrics in results["metrics"].items():
        lines.extend([
            name,
            "                       Predicted no   Predicted delay",
            (
                "Actual no delay        "
                f"{metrics['true_negatives']:>12,} "
                f"{metrics['false_positives']:>17,}"
            ),
            (
                "Actual delay           "
                f"{metrics['false_negatives']:>12,} "
                f"{metrics['true_positives']:>17,}"
            ),
            "",
        ])

    lines.extend(["FULL RUN DETAILS", formatted])

    (directory / "results.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )
