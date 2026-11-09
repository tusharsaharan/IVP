"""A data-dependent, re-uploading EEG graph feature map."""

import numpy as np
import pennylane as qml

from asd_eeg.config import Config


def apply_feature_map(angles: np.ndarray, config: Config, reassign: bool = False) -> None:
    q = len(config.nodes)
    edges = config.edges
    values = angles * config.circuit.bandwidth
    order = np.arange(len(edges))
    if reassign:
        order = np.random.default_rng(config.circuit.edge_seed).permutation(len(edges))
    for wire in range(q):
        qml.Hadamard(wires=wire)
    for _ in range(config.circuit.passes):
        for wire in range(q):
            qml.RY(values[4 * wire], wires=wire)
            qml.RZ(values[4 * wire + 1], wires=wire)
        for edge, source in zip(edges, order, strict=True):
            qml.IsingZZ(values[4 * q + source], wires=edge)
        for wire in range(q):
            qml.RY(values[4 * wire + 2], wires=wire)
            qml.RZ(values[4 * wire + 3], wires=wire)
        for edge, source in zip(edges, order, strict=True):
            qml.IsingZZ(values[4 * q + len(edges) + source], wires=edge)


def make_qnode(config: Config, reassign: bool = False):
    device = qml.device("default.qubit", wires=len(config.nodes), shots=None)

    @qml.qnode(device, interface=None)
    def circuit(angles):
        apply_feature_map(angles, config, reassign)
        return qml.state()

    return circuit


def encode_states(angles: np.ndarray, config: Config, reassign: bool = False) -> np.ndarray:
    values = np.asarray(angles, dtype=float)
    if values.ndim != 2 or values.shape[1] != config.n_features or not np.isfinite(values).all():
        raise ValueError(f"Quantum input must be finite with shape [subjects, {config.n_features}].")
    circuit = make_qnode(config, reassign)
    return np.asarray([circuit(row) for row in values], dtype=complex)


def circuit_text(angles: np.ndarray, config: Config) -> str:
    return qml.draw(make_qnode(config), decimals=2)(angles)
