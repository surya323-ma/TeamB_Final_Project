from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import pandas as pd
import numpy as np


def evaluate(true_labels: list, predicted_labels: list) -> dict:
    """
    Evaluate NLU predictions against ground truth labels.
    Returns accuracy, per-class report, and confusion matrix.
    """
    accuracy = accuracy_score(true_labels, predicted_labels)

    report = classification_report(
        true_labels,
        predicted_labels,
        output_dict=True,
        zero_division=0,
    )

    labels = sorted(set(true_labels + predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    return {
        "accuracy": round(accuracy * 100, 2),
        "report": report,
        "confusion_matrix": cm,
        "labels": labels,
    }
