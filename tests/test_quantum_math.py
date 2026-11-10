"""Quantum kernel mathematics: unit states, PSD kernel, sensitivity, formula match."""

import numpy as np
import pennylane as qml

from asd_eeg.models.kernels import amplitude_kernel, amplitude_states, fidelity_kernel
from asd_eeg.models.quantum_circuit import encode_states


def test_states_unit_norm_and_kernel_psd(config):
    rng = np.random.default_rng(11)
    angles = rng.uniform(-1, 1, size=(6, config.n_features))
    states = encode_states(angles, config)
    assert states.shape == (6, 2 ** len(config.nodes))
    assert states.dtype == complex
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-12)
    kernel = fidelity_kernel(states, states)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-8


def test_node_and_edge_inputs_change_state_and_kernel(config):
    rng = np.random.default_rng(12)
    angles = rng.uniform(-1, 1, size=(4, config.n_features))
    states = encode_states(angles, config)
    base = fidelity_kernel(states, states)
    node_shifted = angles.copy()
    node_shifted[0, 0] += 0.5
    edge_shifted = angles.copy()
    edge_shifted[0, 4 * len(config.nodes)] += 0.5
    assert not np.allclose(encode_states(node_shifted[:1], config), states[:1])
    assert not np.allclose(encode_states(edge_shifted[:1], config), states[:1])
    assert fidelity_kernel(encode_states(node_shifted[:1], config), states[:1])[0, 0] < 1.0
    assert fidelity_kernel(encode_states(edge_shifted[:1], config), states[:1])[0, 0] < 1.0
    reassigned = encode_states(angles, config, reassign=True)
    assert not np.allclose(fidelity_kernel(reassigned, reassigned), base)


def test_reassigned_control_equals_original_circuit_on_permuted_edge_inputs(config):
    """The control only relabels edge inputs; node features and gate layout are unchanged."""
    rng = np.random.default_rng(16)
    angles = rng.uniform(-1, 1, size=(5, config.n_features))
    q, n_edges = len(config.nodes), len(config.edges)
    order = np.random.default_rng(config.circuit.edge_seed).permutation(n_edges)
    permuted = angles.copy()
    permuted[:, 4 * q:4 * q + n_edges] = angles[:, 4 * q + order]
    permuted[:, 4 * q + n_edges:] = angles[:, 4 * q + n_edges + order]
    np.testing.assert_array_equal(permuted[:, :4 * q], angles[:, :4 * q])
    for start in (4 * q, 4 * q + n_edges):
        block = slice(start, start + n_edges)
        np.testing.assert_array_equal(np.sort(permuted[:, block]), np.sort(angles[:, block]))
    reassigned = encode_states(angles, config, reassign=True)
    expected = encode_states(permuted, config)
    np.testing.assert_allclose(fidelity_kernel(reassigned, expected).diagonal(), 1.0, atol=1e-12)


def test_reassignment_permutation_is_pinned_for_saved_bundles(config):
    """Frozen bundles depend on this exact permutation; seed 1729 leaves three edges fixed."""
    assert config.circuit.edge_seed == 1729
    order = np.random.default_rng(1729).permutation(6)
    assert order.tolist() == [0, 1, 2, 4, 5, 3]
    moved = [config.edges[k] for k in range(6) if order[k] != k]
    assert moved == [(1, 2), (1, 3), (2, 3)]


def test_amplitude_kernel_matches_documented_formula(config):
    rng = np.random.default_rng(13)
    left = rng.normal(size=(4, config.n_features))
    right = rng.normal(size=(3, config.n_features))
    left[0] = 0.0  # explicit zero-norm handling: must stay finite, no NaN
    got = amplitude_kernel(left, right)
    expected = (left @ right.T + 1.0) ** 2 / (
        (np.sum(left ** 2, axis=1) + 1)[:, None]
        * (np.sum(right ** 2, axis=1) + 1)[None, :]
    )
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(np.diag(amplitude_kernel(left, left)), 1.0, atol=1e-12)


def test_amplitude_kernel_matches_state_fidelity(config):
    rng = np.random.default_rng(14)
    values = rng.normal(size=(4, config.n_features))
    classical = amplitude_kernel(values, values)
    states = amplitude_states(values)
    wires = int(np.log2(states.shape[1]))

    @qml.qnode(qml.device("default.qubit", wires=wires))
    def prepare(state):
        qml.AmplitudeEmbedding(state, wires=range(wires))
        return qml.state()

    quantum = fidelity_kernel(
        np.array([prepare(row) for row in states]),
        np.array([prepare(row) for row in states]),
    )
    np.testing.assert_allclose(classical, quantum, atol=1e-12)


def test_bundle_save_reload_gives_same_prediction(config, tmp_path):
    import joblib

    from asd_eeg.training.model import fit_model

    rng = np.random.default_rng(15)
    X = rng.normal(size=(8, config.n_features))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    for name in ("rbf", "amplitude", "quantum", "reassigned"):
        model = fit_model(X, y, config, name)
        assert list(model.estimator.classes_) == [0, 1]
        path = tmp_path / f"{name}.joblib"
        joblib.dump({"schema_version": 1, "model": model}, path)
        reloaded = joblib.load(path)["model"]
        probe = rng.normal(size=(3, config.n_features))
        first = model.predict(probe)
        second = reloaded.predict(probe)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1], atol=0)
