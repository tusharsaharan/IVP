"""Tests for asd_eeg.explainability.perturbation — synthetic data only."""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from asd_eeg.explainability.perturbation import (
    EDGES,
    IQR_CONN_START,
    MEDIAN_CONN_START,
    N_CONNECTIONS,
    N_FEATURES,
    compute_training_references,
    plot_case_study,
    plot_perturbation_sensitivity,
    run_batch_perturbation,
    run_perturbation_analysis,
    save_perturbation_results,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def training_data(rng):
    return rng.standard_normal((30, N_FEATURES))


@pytest.fixture
def subject_vec(rng):
    return rng.standard_normal(N_FEATURES)


@pytest.fixture
def training_refs(training_data):
    return compute_training_references(training_data)


@pytest.fixture
def dummy_scorer():
    """Linear scorer: sum of connectivity features minus sum of IQR features."""
    def _score(x: np.ndarray) -> float:
        return float(x[MEDIAN_CONN_START:MEDIAN_CONN_START + N_CONNECTIONS].sum() -
                     x[IQR_CONN_START:IQR_CONN_START + N_CONNECTIONS].sum())
    return _score


# ---------------------------------------------------------------------------
# Constants / Layout
# ---------------------------------------------------------------------------

class TestConstants:
    def test_n_features(self):
        assert N_FEATURES == 28

    def test_n_connections(self):
        assert N_CONNECTIONS == 6

    def test_edges(self):
        assert EDGES == [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

    def test_median_start(self):
        assert MEDIAN_CONN_START == 16

    def test_iqr_start(self):
        assert IQR_CONN_START == 22

    def test_no_overlap(self):
        band_idx = set(range(16))
        med_idx = set(range(MEDIAN_CONN_START, MEDIAN_CONN_START + N_CONNECTIONS))
        iqr_idx = set(range(IQR_CONN_START, IQR_CONN_START + N_CONNECTIONS))
        assert not band_idx & med_idx
        assert not band_idx & iqr_idx
        assert not med_idx & iqr_idx
        assert len(band_idx | med_idx | iqr_idx) == N_FEATURES


# ---------------------------------------------------------------------------
# compute_training_references
# ---------------------------------------------------------------------------

class TestComputeTrainingReferences:
    def test_mean_strategy(self, training_data):
        refs = compute_training_references(training_data, strategy="training_mean")
        assert refs["median_conn"].shape == (N_CONNECTIONS,)
        assert refs["iqr_conn"].shape == (N_CONNECTIONS,)
        expected_med = training_data[:, MEDIAN_CONN_START:MEDIAN_CONN_START + N_CONNECTIONS].mean(axis=0)
        np.testing.assert_allclose(refs["median_conn"], expected_med)

    def test_median_strategy(self, training_data):
        refs = compute_training_references(training_data, strategy="training_median")
        assert refs["median_conn"].shape == (N_CONNECTIONS,)

    def test_invalid_strategy_raises(self, training_data):
        with pytest.raises(ValueError, match="strategy"):
            compute_training_references(training_data, strategy="invalid")

    def test_wrong_feature_count_raises(self):
        with pytest.raises(ValueError):
            compute_training_references(np.ones((10, 20)))


# ---------------------------------------------------------------------------
# run_perturbation_analysis
# ---------------------------------------------------------------------------

class TestRunPerturbationAnalysis:
    def test_returns_six_results(self, subject_vec, dummy_scorer, training_refs):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        assert len(results) == N_CONNECTIONS

    def test_result_keys(self, subject_vec, dummy_scorer, training_refs):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        for r in results:
            assert "edge_index" in r
            assert "edge_label" in r
            assert "original_score" in r
            assert "perturbed_score" in r
            assert "score_change" in r

    def test_score_change_calculation(self, subject_vec, dummy_scorer, training_refs):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        for r in results:
            assert pytest.approx(r["score_change"], abs=1e-9) == r["perturbed_score"] - r["original_score"]

    def test_original_score_consistent(self, subject_vec, dummy_scorer, training_refs):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        # All results share the same original score
        orig = results[0]["original_score"]
        for r in results:
            assert pytest.approx(r["original_score"], abs=1e-9) == orig

    def test_unperturbed_features_unchanged(self, subject_vec, dummy_scorer, training_refs):
        """Band-power features (indices 0-15) must not change during perturbation."""
        perturbed_vecs = []

        def recording_scorer(x):
            perturbed_vecs.append(x.copy())
            return float(x.sum())

        run_perturbation_analysis(subject_vec, recording_scorer, training_refs)
        for pv in perturbed_vecs[1:]:  # Skip original call
            np.testing.assert_array_equal(pv[:16], subject_vec[:16])

    def test_wrong_feature_length_raises(self, dummy_scorer, training_refs):
        with pytest.raises(ValueError, match="28"):
            run_perturbation_analysis(np.ones(10), dummy_scorer, training_refs)

    def test_broken_scorer_raises_runtime_error(self, subject_vec, training_refs):
        def bad_scorer(x):
            raise RuntimeError("scorer failed")
        with pytest.raises(RuntimeError, match="scorer"):
            run_perturbation_analysis(subject_vec, bad_scorer, training_refs)

    def test_non_numeric_features_raises(self, dummy_scorer, training_refs):
        with pytest.raises(TypeError):
            run_perturbation_analysis("not_array", dummy_scorer, training_refs)

    def test_subject_id_in_results(self, subject_vec, dummy_scorer, training_refs):
        results = run_perturbation_analysis(
            subject_vec, dummy_scorer, training_refs, subject_id="test_sub"
        )
        for r in results:
            assert r["subject_id"] == "test_sub"

    def test_edges_match_canonical_order(self, subject_vec, dummy_scorer, training_refs):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        for i, r in enumerate(results):
            assert r["edge_index"] == i


# ---------------------------------------------------------------------------
# run_batch_perturbation
# ---------------------------------------------------------------------------

class TestRunBatchPerturbation:
    def test_returns_dataframe(self, training_data, dummy_scorer, training_refs):
        df = run_batch_perturbation(
            training_data[:5], dummy_scorer, training_refs
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5 * N_CONNECTIONS

    def test_subject_ids(self, training_data, dummy_scorer, training_refs):
        ids = ["A", "B", "C"]
        df = run_batch_perturbation(
            training_data[:3], dummy_scorer, training_refs, subject_ids=ids
        )
        assert set(df["subject_id"].unique()) == set(ids)


# ---------------------------------------------------------------------------
# save_perturbation_results
# ---------------------------------------------------------------------------

class TestSavePerturbationResults:
    def test_saves_csv(self, subject_vec, dummy_scorer, training_refs, tmp_path):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        paths = save_perturbation_results(results, tmp_path, formats=("csv",))
        assert any(p.suffix == ".csv" for p in paths)
        df = pd.read_csv(paths[0])
        assert len(df) == N_CONNECTIONS

    def test_saves_json(self, subject_vec, dummy_scorer, training_refs, tmp_path):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        paths = save_perturbation_results(results, tmp_path, formats=("json",))
        assert any(p.suffix == ".json" for p in paths)
        data = json.loads(paths[0].read_text())
        assert len(data) == N_CONNECTIONS


# ---------------------------------------------------------------------------
# plot_perturbation_sensitivity
# ---------------------------------------------------------------------------

class TestPlotPerturbationSensitivity:
    def test_valid_input(self, subject_vec, dummy_scorer, training_refs):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        fig = plot_perturbation_sensitivity(results, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, subject_vec, dummy_scorer, training_refs, tmp_path):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        plot_perturbation_sensitivity(
            results, output_dir=tmp_path,
            filename_stem="pert", formats=["png"], close=True
        )
        assert (tmp_path / "pert.png").exists()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            plot_perturbation_sensitivity([], close=False)

    def test_dataframe_input(self, subject_vec, dummy_scorer, training_refs):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        df = pd.DataFrame(results)
        fig = plot_perturbation_sensitivity(df, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_case_study
# ---------------------------------------------------------------------------

class TestPlotCaseStudy:
    def test_valid_no_perturbation(self, subject_vec):
        fig = plot_case_study(
            subject_features=subject_vec,
            predicted_label=1,
            decision_score=0.42,
            is_synthetic=True,
            close=False,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_with_perturbation(self, subject_vec, dummy_scorer, training_refs):
        results = run_perturbation_analysis(subject_vec, dummy_scorer, training_refs)
        fig = plot_case_study(
            subject_features=subject_vec,
            predicted_label=0,
            decision_score=-0.3,
            perturbation_results=results,
            true_label=1,
            is_synthetic=True,
            close=False,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, subject_vec, tmp_path):
        plot_case_study(
            subject_features=subject_vec,
            predicted_label=1,
            decision_score=0.5,
            is_synthetic=True,
            output_dir=tmp_path,
            filename_stem="cs",
            formats=["png"],
            close=True,
        )
        assert (tmp_path / "cs.png").exists()

    def test_wrong_feature_length_raises(self):
        with pytest.raises(ValueError, match="28"):
            plot_case_study(
                subject_features=np.ones(10),
                predicted_label=0,
                decision_score=0.0,
                close=False,
            )
