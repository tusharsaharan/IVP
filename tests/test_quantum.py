import numpy as np
import pennylane as qml

from asd_eeg.models.kernels import amplitude_kernel, amplitude_states, fidelity_kernel
from asd_eeg.models.quantum_circuit import encode_states


def test_fidelity_is_normalized_symmetric_psd_and_edge_sensitive(config):
    angles = np.random.default_rng(5).uniform(-1, 1, size=(8, config.n_features))
    states = encode_states(angles, config)
    kernel = fidelity_kernel(states, states)
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(kernel), 1, atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min() > -1e-10
    changed = angles.copy()
    changed[0, 4 * len(config.nodes)] += .8
    altered = encode_states(changed[:1], config)
    assert fidelity_kernel(altered, states[:1])[0, 0] < .999
    reassigned = encode_states(angles, config, reassign=True)
    assert not np.allclose(fidelity_kernel(reassigned, reassigned), kernel)


def test_amplitude_equivalence_including_zero_vector(config):
    values = np.random.default_rng(7).normal(size=(5, config.n_features))
    values[0] = 0
    classical = amplitude_kernel(values, values)
    states = amplitude_states(values)
    wires = int(np.log2(states.shape[1]))
    device = qml.device("default.qubit", wires=wires)

    @qml.qnode(device)
    def prepare(state):
        qml.AmplitudeEmbedding(state, wires=range(wires))
        return qml.state()

    actual = np.array([prepare(row) for row in states])
    np.testing.assert_allclose(classical, fidelity_kernel(actual, actual), atol=1e-12)


def test_scaler_is_training_only(config):
    from asd_eeg.training.model import fit_model

    X = np.random.default_rng(1).normal(size=(6, config.n_features))
    model = fit_model(X, np.array([0, 1, 0, 1, 0, 1]), config, "quantum")
    model.predict(np.full((1, config.n_features), 1000.0))
    np.testing.assert_allclose(model.scaler.mean_, X.mean(axis=0))
