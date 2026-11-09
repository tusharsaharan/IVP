"""Tests for asd_eeg.visualization.eeg — synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # headless backend for CI

from asd_eeg.visualization import eeg as eeg_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_eeg():
    """Return a (4, 512) synthetic EEG array."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((4, 512))


@pytest.fixture
def channel_names_4():
    return ["Fp1", "Fp2", "F3", "F4"]


# ---------------------------------------------------------------------------
# plot_eeg_traces
# ---------------------------------------------------------------------------

class TestPlotEEGTraces:
    def test_valid_input_returns_figure(self, synthetic_eeg):
        fig = eeg_module.plot_eeg_traces(synthetic_eeg, sfreq=256.0, close=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_custom_channel_names(self, synthetic_eeg, channel_names_4):
        fig = eeg_module.plot_eeg_traces(
            synthetic_eeg, channel_names=channel_names_4, close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_png(self, synthetic_eeg, tmp_path):
        eeg_module.plot_eeg_traces(
            synthetic_eeg, output_dir=tmp_path, filename_stem="test_eeg",
            formats=["png"], close=True
        )
        assert (tmp_path / "test_eeg.png").exists()

    def test_saves_pdf(self, synthetic_eeg, tmp_path):
        eeg_module.plot_eeg_traces(
            synthetic_eeg, output_dir=tmp_path, filename_stem="test_eeg",
            formats=["pdf"], close=True
        )
        assert (tmp_path / "test_eeg.pdf").exists()

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError):
            eeg_module.plot_eeg_traces(np.ones((4, 5, 6)), close=False)

    def test_mismatched_channel_names_raises(self, synthetic_eeg):
        with pytest.raises(ValueError):
            eeg_module.plot_eeg_traces(
                synthetic_eeg, channel_names=["A", "B"], close=False
            )

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError):
            eeg_module.plot_eeg_traces("not_an_array", close=False)

    def test_single_channel(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((1, 100))
        fig = eeg_module.plot_eeg_traces(data, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_flat_channel_no_division_error(self):
        """Flat channel (std=0) should not cause division-by-zero."""
        data = np.zeros((2, 100))
        data[1, :] = np.sin(np.linspace(0, 10, 100))
        fig = eeg_module.plot_eeg_traces(data, close=False)
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_selected_channels
# ---------------------------------------------------------------------------

class TestPlotSelectedChannels:
    def test_valid_subset(self, synthetic_eeg, channel_names_4):
        fig = eeg_module.plot_selected_channels(
            synthetic_eeg, channel_indices=[0, 2],
            channel_names=channel_names_4, close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_out_of_range_index_raises(self, synthetic_eeg):
        with pytest.raises(ValueError, match="out of range"):
            eeg_module.plot_selected_channels(
                synthetic_eeg, channel_indices=[0, 99], close=False
            )

    def test_saves_file(self, synthetic_eeg, tmp_path):
        eeg_module.plot_selected_channels(
            synthetic_eeg, channel_indices=[1, 3],
            output_dir=tmp_path, filename_stem="sel",
            formats=["png"], close=True
        )
        assert (tmp_path / "sel.png").exists()


# ---------------------------------------------------------------------------
# plot_eeg_time_window
# ---------------------------------------------------------------------------

class TestPlotEEGTimeWindow:
    def test_valid_window(self, synthetic_eeg):
        # 512 samples @ 256 Hz = 2 s
        fig = eeg_module.plot_eeg_time_window(
            synthetic_eeg, t_start=0.0, t_end=1.0, sfreq=256.0, close=False
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_negative_t_start_raises(self, synthetic_eeg):
        with pytest.raises(ValueError, match="t_start"):
            eeg_module.plot_eeg_time_window(
                synthetic_eeg, t_start=-0.1, t_end=1.0, close=False
            )

    def test_t_end_exceeds_duration_raises(self, synthetic_eeg):
        with pytest.raises(ValueError, match="t_end"):
            eeg_module.plot_eeg_time_window(
                synthetic_eeg, t_start=0.0, t_end=10.0, sfreq=256.0, close=False
            )

    def test_t_start_ge_t_end_raises(self, synthetic_eeg):
        with pytest.raises(ValueError):
            eeg_module.plot_eeg_time_window(
                synthetic_eeg, t_start=1.0, t_end=0.5, sfreq=256.0, close=False
            )

    def test_saves_file(self, synthetic_eeg, tmp_path):
        eeg_module.plot_eeg_time_window(
            synthetic_eeg, t_start=0.0, t_end=1.0, sfreq=256.0,
            output_dir=tmp_path, filename_stem="window",
            formats=["png"], close=True
        )
        assert (tmp_path / "window.png").exists()
