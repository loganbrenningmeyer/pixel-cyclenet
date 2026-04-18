from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .set_style import apply_style

apply_style()


def plot_curve(
    x,
    y,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | Path | None = None,
    label: str | None = None,
    color: str | None = None,
    marker: str | None = "o",
    linewidth: float = 2.0,
    alpha: float = 1.0,
    grid: bool = True,
):
    """
    Plot a simple single-series curve.

    Args:
        x: Sequence of x values.
        y: Sequence of y values.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: Optional output path for the figure.
        label: Optional legend label for the curve.
        color: Optional line color.
        marker: Optional marker style.
        linewidth: Line width.
        alpha: Line alpha.
        grid: Whether to draw a light major grid.
    """
    df = pd.DataFrame({"x": x, "y": y})

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    sns.lineplot(
        data=df,
        x="x",
        y="y",
        ax=ax,
        label=label,
        color=color,
        marker=marker,
        linewidth=linewidth,
        alpha=alpha,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if grid:
        ax.grid(True, which="major", axis="both")
    else:
        ax.grid(False)

    if label is not None:
        ax.legend(frameon=False)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        fig.show()

    return fig, ax
