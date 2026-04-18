from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from .set_style import apply_style

apply_style()


def _sample_coords(xy: np.ndarray, max_points: int | None, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly samples max_points coordinates for plotting
    """
    if max_points is None or len(xy) <= max_points:
        return xy
    
    idx = rng.choice(len(xy), size=max_points, replace=False)
    return xy[idx]


def plot_proj_scatter(
    coords: list[np.ndarray], 
    labels: list[str],
    colors: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | Path | None = None, 
    max_points_per_group: int | None = 2000,
    seed: int = 42,
    alpha: float = 0.5,
    point_size: float = 10.0,
):
    """
    
    """
    # -------------------------
    # Randomly sample coords per group
    # -------------------------
    rng = np.random.default_rng(seed)

    sampled_coords = [
        _sample_coords(xy, max_points_per_group, rng)
        for xy in coords
    ]

    # -------------------------
    # Create UMAP scatter plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for xy, label, color in zip(sampled_coords, labels, colors):
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=point_size,
            alpha=alpha,
            c=color,
            label=label,
            linewidths=0,
            edgecolors="none",
            rasterized=len(xy) > 3000,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

    # -------------------------
    # Save figure
    # -------------------------
    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()

    plt.close(fig)


def plot_proj_density(
    coords: list[np.ndarray], 
    labels: list[str],
    colors: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | Path | None = None, 
    max_points_per_group: int | None = 500,
    seed: int = 42,
    fill: bool = True,
    alpha: float = 0.5,
    point_alpha: float = 0.1,
    point_size: float = 10.0,
    show_points: bool = False,
):
    rng = np.random.default_rng(seed)
    legend_handles = []

    # -------------------------
    # Create UMAP density plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for xy, label, color in zip(coords, labels, colors):
        sns.kdeplot(
            x=xy[:, 0],
            y=xy[:, 1],
            ax=ax,
            fill=fill,
            levels=6,
            color=color,
            alpha=alpha,
        )
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=label))

        # -- Optionally overlay points
        if show_points:
            sampled_xy = _sample_coords(xy, max_points_per_group, rng)
            ax.scatter(
                sampled_xy[:, 0],
                sampled_xy[:, 1],
                s=point_size,
                alpha=point_alpha,
                color=color,
                label=label,
                linewidths=0,
                edgecolors="none",
                rasterized=len(xy) > 3000,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_points:
        ax.legend()
    else:
        ax.legend(handles=legend_handles)

    # -------------------------
    # Save figure
    # -------------------------
    if save_path:
        fig.savefig(save_path)
    else:
        fig.show()
 
    plt.close(fig)   

def plot_proj_density_marginal(
    coords: list[np.ndarray], 
    labels: list[str],
    colors: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | Path | None = None, 
    max_points_per_group: int | None = 500,
    seed: int = 42,
    fill: bool = True,
    alpha: float = 0.5,
    point_alpha: float = 0.1,
    point_size: float = 10.0,
    show_points: bool = False,
):
    rng = np.random.default_rng(seed)
    rows = []
    for xy, label in zip(coords, labels):
        rows.extend(
            {
                xlabel: float(x),
                ylabel: float(y),
                "group": label,
            }
            for x, y in xy
        )

    df = pd.DataFrame(rows)
    palette = {label: color for label, color in zip(labels, colors)}

    g = sns.jointplot(
        data=df,
        x=xlabel,
        y=ylabel,
        hue="group",
        kind="kde",
        fill=fill,
        levels=6,
        alpha=alpha,
        palette=palette,
        height=6.5,
        space=0,
    )

    if show_points:
        for xy, label, color in zip(coords, labels, colors):
            sampled_xy = _sample_coords(xy, max_points_per_group, rng)
            g.ax_joint.scatter(
                sampled_xy[:, 0],
                sampled_xy[:, 1],
                s=point_size,
                alpha=point_alpha,
                color=color,
                linewidths=0,
                edgecolors="none",
                rasterized=len(xy) > 3000,
            )

    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    g.ax_joint.set_xticks([])
    g.ax_joint.set_yticks([])
    g.ax_marg_x.set_xticklabels([])
    g.ax_marg_x.set_yticks([])
    g.ax_marg_y.set_xticks([])
    g.ax_marg_y.set_yticklabels([])

    g.figure.suptitle(title, y=1.02)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.figure.savefig(save_path)
    else:
        g.figure.show()

    plt.close(g.figure)