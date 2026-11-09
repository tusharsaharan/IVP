import numpy as np


def fidelity_kernel(left_states: np.ndarray, right_states: np.ndarray) -> np.ndarray:
    """Squared overlap. Columns always preserve right_states reference order."""
    return np.abs(left_states.conj() @ right_states.T) ** 2


def amplitude_states(features: np.ndarray) -> np.ndarray:
    """Append 1 to handle zero vectors, then zero-pad to a power of two and normalize."""
    values = np.column_stack([features, np.ones(len(features))])
    width = 1 << (values.shape[1] - 1).bit_length()
    values = np.pad(values, ((0, 0), (0, width - values.shape[1])))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def amplitude_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Classically evaluated amplitude-fidelity kernel; this is not a speedup claim."""
    numerator = (left @ right.T + 1.0) ** 2
    denominator = (np.sum(left ** 2, axis=1) + 1)[:, None] * (np.sum(right ** 2, axis=1) + 1)[None, :]
    return numerator / denominator
