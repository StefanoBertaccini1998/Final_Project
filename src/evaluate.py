"""
evaluate.py — Performance Evaluation
Computes metrics appropriate for the chosen task (classification/detection/segmentation).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: list = None) -> dict:
    """Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: Optional list of class name strings.

    Returns:
        Dictionary with accuracy, precision, recall, f1, confusion_matrix.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, target_names=class_names)
    }
    return metrics


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute Intersection over Union for binary masks.

    Args:
        pred_mask: Predicted binary mask (0/1).
        true_mask: Ground truth binary mask (0/1).

    Returns:
        IoU value in [0, 1].
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_dice(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Compute Dice Coefficient for binary masks.

    Args:
        pred_mask: Predicted binary mask (0/1).
        true_mask: Ground truth binary mask (0/1).

    Returns:
        Dice coefficient in [0, 1].
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    return float(2 * intersection / total) if total > 0 else 0.0


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print("=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"\nConfusion Matrix:\n{value}")
        elif key == "report":
            print(f"\nClassification Report:\n{value}")
        else:
            print(f"{key.capitalize()}: {value:.4f}")
    print("=" * 40)
