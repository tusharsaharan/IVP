"""Tests for asd_eeg.visualization.results — synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from asd_eeg.visualization import results as res_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_metrics():
    return {
        "RBF-SVM": {"accuracy": 0.72, "precision": 0.70, "recall": 0.74, "f1_score": 0.72},
        "QK-SVM":  {"accuracy": 0.78, "precision": 0.76, "recall": 0.80, "f1_score": 0.78},
    }


@pytest.fixture
def binary_labels_and_scores():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=30)
    scores = rng.standard_normal(30)
    return y_true, scores


# ---------------------------------------------------------------------------
# plot_model_comparison
# ---------------------------------------------------------------------------

class TestPlotModelComparison:
    def test_valid_input(self, model_metrics):
        fig = res_module.plot_model_comparison(model_metrics, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, model_metrics, tmp_path):
        res_module.plot_model_comparison(
            model_metrics, output_dir=tmp_path,
            filename_stem="mc", formats=["png"], close=True
        )
        assert (tmp_path / "mc.png").exists()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            res_module.plot_model_comparison({}, close=False)

    def test_subset_metrics(self, model_metrics):
        fig = res_module.plot_model_comparison(
            model_metrics, metric_keys=["accuracy", "f1_score"], close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix:
    def test_valid_input(self, binary_labels_and_scores):
        y_true, _ = binary_labels_and_scores
        y_pred = (y_true + np.random.default_rng(1).integers(0, 2, size=len(y_true))) % 2
        fig = res_module.plot_confusion_matrix(y_true, y_pred, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, binary_labels_and_scores, tmp_path):
        y_true, _ = binary_labels_and_scores
        y_pred = y_true.copy()
        res_module.plot_confusion_matrix(
            y_true, y_pred,
            output_dir=tmp_path, filename_stem="cm",
            formats=["png"], close=True
        )
        assert (tmp_path / "cm.png").exists()

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            res_module.plot_confusion_matrix(
                np.array([0, 1, 0]), np.array([1, 0]), close=False
            )


# ---------------------------------------------------------------------------
# plot_roc_curve
# ---------------------------------------------------------------------------

class TestPlotROCCurve:
    def test_valid_input(self, binary_labels_and_scores):
        y_true, scores = binary_labels_and_scores
        fig = res_module.plot_roc_curve(y_true, scores, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_single_class_raises(self):
        y_true = np.zeros(10)
        scores = np.ones(10)
        with pytest.raises(ValueError, match="two distinct"):
            res_module.plot_roc_curve(y_true, scores, close=False)

    def test_saves_png(self, binary_labels_and_scores, tmp_path):
        y_true, scores = binary_labels_and_scores
        res_module.plot_roc_curve(
            y_true, scores,
            output_dir=tmp_path, filename_stem="roc",
            formats=["png"], close=True
        )
        assert (tmp_path / "roc.png").exists()


# ---------------------------------------------------------------------------
# plot_fold_performance
# ---------------------------------------------------------------------------

class TestPlotFoldPerformance:
    def test_valid_input(self):
        fold_metrics = [{"accuracy": 0.7}, {"accuracy": 0.75}, {"accuracy": 0.68}]
        fig = res_module.plot_fold_performance(fold_metrics, metric_key="accuracy", close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            res_module.plot_fold_performance([], close=False)

    def test_saves_png(self, tmp_path):
        fold_metrics = [{"accuracy": 0.7}, {"accuracy": 0.75}]
        res_module.plot_fold_performance(
            fold_metrics, output_dir=tmp_path,
            filename_stem="fold", formats=["png"], close=True
        )
        assert (tmp_path / "fold.png").exists()


# ---------------------------------------------------------------------------
# plot_metric_heatmap
# ---------------------------------------------------------------------------

class TestPlotMetricHeatmap:
    def test_valid_input(self, model_metrics):
        fig = res_module.plot_metric_heatmap(model_metrics, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, model_metrics, tmp_path):
        res_module.plot_metric_heatmap(
            model_metrics, output_dir=tmp_path,
            filename_stem="heatmap", formats=["png"], close=True
        )
        assert (tmp_path / "heatmap.png").exists()
