"""
models/classical.py — Classical ML Model (Core Logic)
SVM or Random Forest trained on handcrafted features (HOG).
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib


class ClassicalClassifier:
    """Classical ML classifier using SVM or Random Forest on HOG features."""

    def __init__(self, model_type: str = "svm", **kwargs):
        """
        Args:
            model_type: 'svm' or 'random_forest'.
            **kwargs: Additional args passed to the sklearn estimator.
        """
        self.scaler = StandardScaler()
        if model_type == "svm":
            self.model = SVC(probability=True, **kwargs)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Labels (n_samples,).
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Predicted labels.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: str):
        """Save model and scaler to disk."""
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: str):
        """Load model and scaler from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        return self
