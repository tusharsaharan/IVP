"""Tests for asd_eeg.visualization.connectivity — synthetic data only."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from asd_eeg.visualization import connectivity as conn_module
from asd_eeg.visualization.connectivity import EDGES, N_EDGES, N_NODES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_weights():
    rng = np.random.default_rng(7)
    return rng.uniform(0.1, 1.0, size=N_EDGES)


@pytest.fixture
def synthetic_matrix():
    rng = np.random.default_rng(7)
    m = rng.uniform(0.0, 1.0, size=(N_NODES, N_NODES))
    return (m + m.T) / 2  # symmetric


@pytest.fixture
def node_names():
    return ["F3", "F4", "P3", "P4"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_edges_order(self):
        assert EDGES == [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

    def test_n_edges(self):
        assert N_EDGES == 6

    def test_n_nodes(self):
        assert N_NODES == 4


# ---------------------------------------------------------------------------
# plot_connectivity_graph
# ---------------------------------------------------------------------------

class TestPlotConnectivityGraph:
    def test_valid_input(self, synthetic_weights):
        fig = conn_module.plot_connectivity_graph(synthetic_weights, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_custom_node_names(self, synthetic_weights, node_names):
        fig = conn_module.plot_connectivity_graph(
            synthetic_weights, node_names=node_names, close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, synthetic_weights, tmp_path):
        conn_module.plot_connectivity_graph(
            synthetic_weights, output_dir=tmp_path,
            filename_stem="graph", formats=["png"], close=True
        )
        assert (tmp_path / "graph.png").exists()

    def test_wrong_edge_count_raises(self):
        with pytest.raises(ValueError, match="6"):
            conn_module.plot_connectivity_graph(np.ones(5), close=False)

    def test_wrong_node_name_count_raises(self, synthetic_weights):
        with pytest.raises(ValueError):
            conn_module.plot_connectivity_graph(
                synthetic_weights, node_names=["A", "B"], close=False
            )

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            conn_module.plot_connectivity_graph("not_array", close=False)


# ---------------------------------------------------------------------------
# plot_connectivity_matrix
# ---------------------------------------------------------------------------

class TestPlotConnectivityMatrix:
    def test_valid_matrix(self, synthetic_matrix):
        fig = conn_module.plot_connectivity_matrix(synthetic_matrix, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="4, 4"):
            conn_module.plot_connectivity_matrix(np.ones((3, 3)), close=False)

    def test_saves_file(self, synthetic_matrix, tmp_path):
        conn_module.plot_connectivity_matrix(
            synthetic_matrix, output_dir=tmp_path,
            filename_stem="mat", formats=["png"], close=True
        )
        assert (tmp_path / "mat.png").exists()


# ---------------------------------------------------------------------------
# plot_edge_weights_bar
# ---------------------------------------------------------------------------

class TestPlotEdgeWeightsBar:
    def test_valid_input(self, synthetic_weights):
        fig = conn_module.plot_edge_weights_bar(synthetic_weights, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_wrong_count_raises(self):
        with pytest.raises(ValueError):
            conn_module.plot_edge_weights_bar(np.ones(4), close=False)

    def test_saves_file(self, synthetic_weights, tmp_path):
        conn_module.plot_edge_weights_bar(
            synthetic_weights, output_dir=tmp_path,
            filename_stem="bar", formats=["png"], close=True
        )
        assert (tmp_path / "bar.png").exists()
