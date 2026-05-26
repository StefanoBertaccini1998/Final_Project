"""
models/classical.py — Pipeline Stage 3a: Classical Classifier
==============================================================
Responsibility: train and evaluate an SVM classifier on HOG feature vectors.

Design decisions:
  - SVM with RBF kernel is the standard baseline for texture classification.
    It maps the HOG feature space into a higher-dimensional space where a
    linear separator exists, without explicitly computing the transformation.
  - StandardScaler is mandatory before SVM: the SVM optimization is sensitive
    to feature scale. HOG values across different blocks can differ by orders
    of magnitude; unscaled input causes the RBF kernel to be dominated by
    high-variance dimensions.
  - C=1.0 is the sklearn default and a reasonable starting point. A higher C
    reduces regularization and fits training data more closely at the risk of
    overfitting; a lower C increases regularization. Tune with cross-validation
    if the baseline F1 is below 0.80.

Why SVM over Random Forest as the primary baseline:
  SVMs with RBF kernels consistently outperform Random Forests on
  high-dimensional, dense feature vectors like HOG (typically 30k+ dims).
  Random Forests partition the feature space with axis-aligned splits, which
  is inefficient in high dimensions. SVM is the documented choice in the
  MVTec AD comparison literature for classical methods.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


# ---------------------------------------------------------------------------
# SVM constants
# ---------------------------------------------------------------------------

# C=1.0: default regularization. Balances margin maximization vs training
# error. Start here and tune only if validation F1 < 0.80.
SVM_C: float = 1.0

# RBF kernel: maps input to infinite-dimensional feature space via the
# Gaussian similarity function. Appropriate when the decision boundary
# is non-linear in the original HOG space (which it typically is).
SVM_KERNEL: str = "rbf"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ClassicalClassifier:
    """SVM classifier with StandardScaler preprocessing for HOG features.

    Wraps sklearn SVC + StandardScaler into a single object so that
    scaling is always applied consistently — callers cannot accidentally
    forget to scale at inference time.
    """

    def __init__(self, C: float = SVM_C, kernel: str = SVM_KERNEL):
        """
        Args:
            C: SVM regularization parameter. Lower = stronger regularization.
            kernel: SVM kernel type. 'rbf' for non-linear, 'linear' for baseline.
        """
        self.scaler = StandardScaler()
        self.model = SVC(C=C, kernel=kernel, probability=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalClassifier":
        """Fit scaler and train SVM.

        Args:
            X: HOG feature matrix (n_samples, n_features).
            y: Binary labels — 0=good, 1=defective (n_samples,).

        Returns:
            self, to allow chaining: clf.fit(X, y).predict(X_test)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: HOG feature matrix (n_samples, n_features).

        Returns:
            Predicted binary labels (n_samples,).
        """
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (requires probability=True at init).

        Useful for ROC curves and threshold tuning.

        Args:
            X: HOG feature matrix (n_samples, n_features).

        Returns:
            Array (n_samples, 2) — columns are P(good), P(defective).
        """
        return self.model.predict_proba(self.scaler.transform(X))

    def save(self, path: str) -> None:
        """Persist model and scaler together to avoid scale mismatch at load.

        Args:
            path: File path (e.g., 'outputs/checkpoints/svm_metal_nut.pkl').
        """
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: str) -> "ClassicalClassifier":
        """Load model and scaler from disk.

        Args:
            path: Path to a .pkl file saved by save().

        Returns:
            self with model and scaler restored.
        """
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self
