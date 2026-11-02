"""Colours and the dark theme shared by all explainit plots."""

from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend import Legend

COLORS = {
    "deep_teal": "#008080",
    "forest_green": "#228b22",
    "moss_green": "#8a9a5b",
    "warm_brown": "#8b4513",
    "steel_gray": "#46505a",
    "dusty_blue": "#6b8e9b",
    "dirty_white": "#f0f0eb",
    "charcoal": "#2d2d2d",
    "light_gray": "#b4b4b4",
    "dark_background": "#0f0f0f",
    "black_background": "#000000",
    "plot_background": "#191919",
}

# Bars and histograms, by category / class index.
BAR_COLORS = (
    COLORS["deep_teal"],
    COLORS["forest_green"],
    COLORS["moss_green"],
    COLORS["warm_brown"],
    COLORS["steel_gray"],
    COLORS["dusty_blue"],
)

# Sample/exemplar markers, by pair index.
PAIR_COLORS = (
    "#FF6B35",
    "#FFD166",
    "#06D6A0",
    "#118AB2",
    "#9D4EDD",
    "#EF476F",
    "#26C485",
    "#F4A261",
)

# Lines marking where the priority crosses these levels.
LEVEL_COLORS = {
    1.0: "#90EE90",
    0.75: "#A0E7E5",
    0.5: "#FFD166",
    0.25: "#F4A261",
    0.0: "#E76F51",
}

TEXT = COLORS["dirty_white"]


def bar_color(index: int) -> str:
    return BAR_COLORS[index % len(BAR_COLORS)]


def pair_color(index: int) -> str:
    return PAIR_COLORS[index % len(PAIR_COLORS)]


def line_color(kind: str = "theoretical") -> str:
    """Colour of a priority curve (``"theoretical"``) or a sampled histogram (``"empirical"``)."""
    if kind == "theoretical":
        return COLORS["deep_teal"]
    if kind == "empirical":
        return COLORS["forest_green"]
    return COLORS["warm_brown"]


def apply_style() -> None:
    """Set the dark theme in Matplotlib's global settings (known issue PLOT-1)."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Bebas Neue", "Luckiest Guy", "Rockwell"]

    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 26
    plt.rcParams["xtick.labelsize"] = 24
    plt.rcParams["ytick.labelsize"] = 24
    plt.rcParams["legend.fontsize"] = 18
    # Ensure the minus sign renders on fonts lacking U+2212.
    plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams["axes.edgecolor"] = TEXT
    plt.rcParams["axes.linewidth"] = 2.0
    plt.rcParams["grid.color"] = COLORS["steel_gray"]
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["text.color"] = TEXT
    plt.rcParams["axes.labelcolor"] = TEXT
    plt.rcParams["xtick.color"] = TEXT
    plt.rcParams["ytick.color"] = TEXT

    plt.rcParams["figure.facecolor"] = COLORS["dark_background"]
    plt.rcParams["axes.facecolor"] = COLORS["plot_background"]


def style_axes(ax: Axes, grid: Literal["both", "x", "y"] = "both") -> None:
    """Theme one axes: open spines, dashed grid (``"both"`` or ``"y"``), large bold labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)
    ax.spines["left"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)

    ax.grid(True, axis=grid, alpha=0.4, color=COLORS["steel_gray"], linestyle="--", linewidth=1.2)
    ax.set_axisbelow(True)
    ax.set_facecolor(COLORS["plot_background"])

    ax.set_title(ax.get_title(), color=TEXT, fontweight="bold", fontsize=26, pad=30)
    ax.set_xlabel(ax.get_xlabel(), color=TEXT, fontweight="bold", fontsize=28)
    ax.set_ylabel(ax.get_ylabel(), color=TEXT, fontweight="bold", fontsize=28)

    ax.tick_params(axis="x", colors=TEXT, labelsize=24)
    ax.tick_params(axis="y", colors=TEXT, labelsize=24)


def styled_legend(
    ax: Axes, *handles_and_labels: Any, fontsize: float, anchor: tuple[float, float]
) -> Legend:
    """A themed legend to the right of ``ax``, anchored at ``anchor`` (axes coordinates)."""
    legend = ax.legend(
        *handles_and_labels,
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor=COLORS["dark_background"],
        edgecolor=TEXT,
        fontsize=fontsize,
        loc="center left",
        bbox_to_anchor=anchor,
        ncol=1,
    )
    legend.get_frame().set_alpha(0.9)
    for text in legend.get_texts():
        text.set_color(TEXT)
    return legend


VALUE_LABEL_BOX = {
    "boxstyle": "round,pad=0.4",
    "facecolor": COLORS["dark_background"],
    "alpha": 0.8,
    "edgecolor": TEXT,
    "linewidth": 1.5,
}
