import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, BLUE, PAPER = "#243342", "#335e83", "#f8f9f7"


def connectivity_figure(matrix, names):
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=PAPER)
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                    color=PAPER if abs(matrix[i][j]) > .55 else INK)
    fig.colorbar(image, ax=ax, label="Median alpha-envelope correlation", shrink=.8)
    fig.tight_layout()
    return fig


def signal_figure(recording, seconds=8):
    count = min(len(recording.channels), 6)
    samples = min(recording.data.shape[1], round(seconds * recording.sfreq))
    time = np.arange(samples) / recording.sfreq
    fig, axes = plt.subplots(count, 1, sharex=True, figsize=(8, max(3, count * .8)), facecolor=PAPER)
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.set_facecolor(PAPER)
        ax.plot(time, recording.data[i, :samples] * 1e6, color=BLUE, linewidth=.7)
        ax.set_ylabel(recording.channels[i], rotation=0, labelpad=22, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("Time (seconds); amplitudes in microvolts")
    fig.tight_layout()
    return fig


def confusion_figure(matrix):
    fig, ax = plt.subplots(figsize=(4, 3), facecolor=PAPER)
    ax.imshow(matrix, cmap="Blues")
    maximum = max(max(row) for row in matrix)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=18,
                    color=PAPER if matrix[i][j] > maximum / 2 else INK)
    ax.set_xticks([0, 1], ["Control", "ASD"])
    ax.set_yticks([0, 1], ["Control", "ASD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Verified label")
    fig.tight_layout()
    return fig
