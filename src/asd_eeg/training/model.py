from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

from asd_eeg.baselines.svm import make_svm
from asd_eeg.config import Config
from asd_eeg.models.kernels import amplitude_kernel, fidelity_kernel
from asd_eeg.models.quantum_circuit import encode_states

MODELS = {
    "rbf": "RBF-SVM",
    "amplitude": "Amplitude kernel",
    "quantum": "Connectivity quantum kernel",
    "reassigned": "Reassigned connections",
}


@dataclass
class FittedModel:
    name: str
    config: Config
    scaler: StandardScaler
    estimator: object
    train_angles: np.ndarray
    train_states: np.ndarray | None
    feature_reference: np.ndarray

    def angles(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.config.n_features or not np.isfinite(X).all():
            raise ValueError("Prediction features do not match the fitted model.")
        return (np.pi / 2) * np.tanh(self.scaler.transform(X) / 2)

    def model_input(self, X: np.ndarray) -> np.ndarray:
        values = self.angles(X)
        if self.name == "rbf":
            return values
        if self.name == "amplitude":
            return amplitude_kernel(values, self.train_angles)
        states = encode_states(values, self.config, reassign=self.name == "reassigned")
        return fidelity_kernel(states, self.train_states)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = self.model_input(X)
        return self.estimator.predict(values), self.estimator.decision_function(values)


def fit_model(X: np.ndarray, y: np.ndarray, config: Config, name: str) -> FittedModel:
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training requires both Control and ASD subjects.")
    scaler = StandardScaler().fit(X)
    values = (np.pi / 2) * np.tanh(scaler.transform(X) / 2)
    states = None
    if name == "rbf":
        training = values
    elif name == "amplitude":
        training = amplitude_kernel(values, values)
    else:
        states = encode_states(values, config, reassign=name == "reassigned")
        training = fidelity_kernel(states, states)
    estimator = make_svm("rbf" if name == "rbf" else "precomputed", config.evaluation.svm_c)
    estimator.fit(training, y)
    return FittedModel(name, config, scaler, estimator, values, states, np.median(X, axis=0))
