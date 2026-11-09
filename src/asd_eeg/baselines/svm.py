"""Owner: Dodiya Mahavirsinh. Baseline SVM with the shared bounded input."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


def make_svm(kernel: str, c: float, random_state: int = SEED) -> SVC:
    return SVC(C=c, kernel=kernel, gamma="scale", class_weight="balanced",
               random_state=random_state)


def fit_scaler_phi(X_train: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """Fit StandardScaler on training subjects only, then bound angles.

    phi = (pi/2) * tanh(z/2). Never fit on held-out subjects.
    """
    scaler = StandardScaler().fit(X_train)
    return scaler, (np.pi / 2) * np.tanh(scaler.transform(X_train) / 2)


def apply_phi(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Apply a training-fitted scaler and the shared angle bound."""
    return (np.pi / 2) * np.tanh(scaler.transform(X) / 2)
