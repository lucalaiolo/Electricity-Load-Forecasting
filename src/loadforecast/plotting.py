"""Shared chart style: quiet chrome, thin marks, one fixed colour per model."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Colour follows the model in every figure (categorical slots in fixed order).
COLORS = {
    "Ensemble": "#2a78d6",
    "TSO": "#eb6834",
    "SARIMA": "#1baf7a",
    "GBM": "#eda100",
    "MLP": "#e87ba4",
    "LSTM": "#008300",
    "Seasonal naive": MUTED,
    "actual": INK,
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # for non-model categories (max three per chart)
LINE = 1.4  # pt; about 2 px once GitHub scales the 200-dpi image down


def setup() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "ytick.left": False,
            "xtick.major.width": 0.6,
            "lines.linewidth": LINE,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_2,
            "legend.handlelength": 1.6,
        }
    )


def save(fig, path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
