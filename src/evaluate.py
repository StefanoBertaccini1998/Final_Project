"""
evaluate.py — Pipeline Evaluation Module
=========================================
Responsibility: compute and report performance metrics for both the classical
(HOG+SVM) and deep learning (EfficientNet) pipelines.

Design decisions:
  - Two averaging strategies are reported for binary classification:
      * 'binary': metrics for the defective class specifically. This is the
        primary metric because our goal is defect detection, not overall accuracy.
        A model that always predicts 'good' gets 72% accuracy but 0% recall on
        defects — binary recall exposes this failure mode immediately.
      * 'weighted': accounts for class imbalance; useful for overall model quality.
  - Recall on the defective class is flagged as the safety-critical metric.
    In industrial inspection, a False Negative (missed defect) has higher cost
    than a False Positive (good part rejected). Optimizing for recall on defects
    is the industrially correct objective.
  - IoU and Dice are implemented for future segmentation extension but are not
    used in the binary classification pipeline.

Usage:
    from src.evaluate import evaluate_classification, print_results

    metrics = evaluate_classification(y_true, y_pred, model_name='HOG+SVM')
    print_results(metrics)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
    class_names: list = None,
) -> dict:
    """Compute classification metrics for binary good/defective classification.

    Returns both binary (defect-class) and weighted metrics so callers can
    report the number most relevant to their context (safety vs. overall quality).

    Args:
        y_true: Ground truth labels — 0=good, 1=defective.
        y_pred: Predicted labels.
        model_name: Label used in printed output (e.g. 'HOG+SVM').
        class_names: Display names for classes. Defaults to ['good', 'defective'].

    Returns:
        Dict with keys: model_name, accuracy, f1_binary, f1_weighted,
        precision_defect, recall_defect, confusion_matrix, report.
    """
    if class_names is None:
        class_names = ["good", "defective"]

    cm = confusion_matrix(y_true, y_pred)

    return {
        "model_name":        model_name,
        "accuracy":          accuracy_score(y_true, y_pred),
        # Binary: metrics computed on the defective class (pos_label=1)
        "f1_binary":         f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0),
        "precision_defect":  precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0),
        "recall_defect":     recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0),
        # Weighted: accounts for class imbalance across all classes
        "f1_weighted":       f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix":  cm,
        "report":            classification_report(y_true, y_pred, target_names=class_names),
        "class_names":       class_names,
    }


def print_results(metrics: dict) -> None:
    """Print a formatted summary of classification metrics.

    Highlights recall_defect as the safety-critical metric.

    Args:
        metrics: Dict returned by evaluate_classification().
    """
    name = metrics.get("model_name", "model")
    sep  = "=" * 48

    print(sep)
    print(f"  Results — {name}")
    print(sep)
    print(f"  Accuracy           : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  F1 (binary/defect) : {metrics['f1_binary']:.4f}  <-- primary metric")
    print(f"  F1 (weighted)      : {metrics['f1_weighted']:.4f}")
    print(f"  Precision (defect) : {metrics['precision_defect']:.4f}")
    print(f"  Recall (defect)    : {metrics['recall_defect']:.4f}  <-- safety-critical")
    print()

    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:")
    print(f"    TN (good -> good)      : {tn:4d}")
    print(f"    FP (good -> defective) : {fp:4d}")
    print(f"    FN (defect missed)    : {fn:4d}  <-- safety-critical")
    print(f"    TP (defect detected)  : {tp:4d}")
    print(sep)


def results_row(metrics: dict) -> dict:
    """Return a flat dict suitable for building a comparison table across models.

    Used in notebooks/05_model_comparison.ipynb to assemble the final results table.

    Args:
        metrics: Dict returned by evaluate_classification().

    Returns:
        Flat dict with model_name, accuracy, f1_binary, recall_defect, precision_defect.
    """
    return {
        "Model":             metrics["model_name"],
        "Accuracy":          round(metrics["accuracy"],         4),
        "F1 (defect)":       round(metrics["f1_binary"],        4),
        "Recall (defect)":   round(metrics["recall_defect"],    4),
        "Precision (defect)":round(metrics["precision_defect"], 4),
    }


# ---------------------------------------------------------------------------
# Segmentation metrics (for future ROI / mask evaluation)
# ---------------------------------------------------------------------------

def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Intersection over Union for binary segmentation masks.

    Args:
        pred_mask: Predicted binary mask (bool or 0/1).
        true_mask: Ground truth binary mask (bool or 0/1).

    Returns:
        IoU in [0.0, 1.0]. Returns 0.0 when both masks are empty.
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union        = np.logical_or(pred_mask, true_mask).sum()
    return float(intersection / union) if union > 0 else 0.0


def compute_dice(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """Dice coefficient (F1 for segmentation masks).

    Args:
        pred_mask: Predicted binary mask (bool or 0/1).
        true_mask: Ground truth binary mask (bool or 0/1).

    Returns:
        Dice in [0.0, 1.0]. Returns 0.0 when both masks are empty.
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    total        = pred_mask.sum() + true_mask.sum()
    return float(2 * intersection / total) if total > 0 else 0.0
