from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .set_style import apply_style

apply_style()


def plot_heatmap(
    grid_df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | Path | None = None,
    cmap: str = "viridis",
    annot: bool = False,
    fmt: str = ".2f",
    vmin=None,
    vmax=None,
    center=None,
    cbar_label: str | None = None,
    linewidths: float = 0.3,
    linecolor: str = "white",
    square: bool = False,
    mask: pd.DataFrame | None = None,
):
    """
    Plot a labeled heatmap from a 2D pandas DataFrame.

    Args:
        grid_df: 2D table of values to plot. Index values become y tick labels,
            columns become x tick labels.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional output path for the figure.
        cmap: Matplotlib/seaborn colormap name.
        annot: Whether to draw cell values.
        fmt: Annotation number format.
        vmin: Optional lower color scale bound.
        vmax: Optional upper color scale bound.
        center: Optional colormap center, useful for diverging maps.
        cbar_label: Optional colorbar label.
        linewidths: Width of cell separators.
        linecolor: Color of cell separators.
        square: Whether to force square cells.
        mask: Optional boolean mask with the same shape as `grid_df`.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    hm = sns.heatmap(
        grid_df,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        vmin=vmin,
        vmax=vmax,
        center=center,
        linewidths=linewidths,
        linecolor=linecolor,
        square=square,
        cbar=True,
        mask=mask,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Keep row/column labels readable for small sweep grids.
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    if cbar_label is not None:
        colorbar = hm.collections[0].colorbar
        if colorbar is not None:
            colorbar.set_label(cbar_label)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        fig.show()

    return fig, ax
