"""Tests for asd_eeg.visualization.kernels — synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from asd_eeg.visualization import kernels as kern_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _symmetric_kernel(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.1, 1.0, (n, n))
    K = A @ A.T
    # Normalize diagonal to 1 (approximate quantum kernel property)
    d = np.sqrt(np.diag(K))
    K = K / np.outer(d, d)
    return K


@pytest.fixture
def train_kernel():
    return _symmetric_kernel(20)


@pytest.fixture
def test_kernel():
    rng = np.random.default_rng(99)
    return rng.uniform(0.0, 1.0, (8, 20))


# ---------------------------------------------------------------------------
# _validate_kernel_matrix
# ---------------------------------------------------------------------------

class TestValidation:
    def test_non_finite_raises(self):
        mat = np.eye(5)
        mat[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            kern_module._validate_kernel_matrix(mat, check_square=True)

    def test_non_square_raises_when_required(self):
        with pytest.raises(ValueError, match="square"):
            kern_module._validate_kernel_matrix(np.ones((3, 5)), check_square=True)

    def test_valid_square_passes(self, train_kernel):
        mat = kern_module._validate_kernel_matrix(
            train_kernel, check_square=True, check_symmetric=True
        )
        assert mat.shape[0] == mat.shape[1]


# ---------------------------------------------------------------------------
# plot_kernel_matrix
# ---------------------------------------------------------------------------

class TestPlotKernelMatrix:
    def test_valid_training_kernel(self, train_kernel):
        fig = kern_module.plot_kernel_matrix(
            train_kernel, is_training_kernel=True, close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_valid_test_kernel(self, test_kernel):
        fig = kern_module.plot_kernel_matrix(
            test_kernel, is_training_kernel=False, close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, train_kernel, tmp_path):
        kern_module.plot_kernel_matrix(
            train_kernel, output_dir=tmp_path,
            filename_stem="km", formats=["png"], close=True
        )
        assert (tmp_path / "km.png").exists()

    def test_non_square_training_kernel_raises(self):
        with pytest.raises(ValueError, match="square"):
            kern_module.plot_kernel_matrix(
                np.ones((4, 6)), is_training_kernel=True, close=False
            )


# ---------------------------------------------------------------------------
# plot_kernel_comparison
# ---------------------------------------------------------------------------

class TestPlotKernelComparison:
    def test_two_kernels(self, train_kernel):
        K2 = _symmetric_kernel(20, seed=7)
        fig = kern_module.plot_kernel_comparison(
            {"RBF": train_kernel, "Quantum": K2}, close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            kern_module.plot_kernel_comparison({}, close=False)

    def test_saves_file(self, train_kernel, tmp_path):
        kern_module.plot_kernel_comparison(
            {"K": train_kernel},
            output_dir=tmp_path, filename_stem="comp",
            formats=["png"], close=True
        )
        assert (tmp_path / "comp.png").exists()


# ---------------------------------------------------------------------------
# plot_kernel_distribution
# ---------------------------------------------------------------------------

class TestPlotKernelDistribution:
    def test_valid_input(self, train_kernel):
        fig = kern_module.plot_kernel_distribution(
            {"K": train_kernel}, close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            kern_module.plot_kernel_distribution({}, close=False)


# ---------------------------------------------------------------------------
# plot_train_test_kernel
# ---------------------------------------------------------------------------

class TestPlotTrainTestKernel:
    def test_valid_pair(self, train_kernel, test_kernel):
        fig = kern_module.plot_train_test_kernel(
            train_kernel, test_kernel, close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_mismatched_columns_raises(self, train_kernel):
        bad_test = np.ones((5, 15))  # 15 cols != 20 train
        with pytest.raises(ValueError, match="columns"):
            kern_module.plot_train_test_kernel(train_kernel, bad_test, close=False)
