import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .set_style import apply_style

apply_style()


HEATMAP_TITLE_FONTSIZE = 20
HEATMAP_AXIS_LABEL_FONTSIZE = 18
HEATMAP_TICK_LABEL_FONTSIZE = 15
HEATMAP_ANNOTATION_FONTSIZE = 15
HEATMAP_COLORBAR_LABEL_FONTSIZE = 18
HEATMAP_COLORBAR_TICK_FONTSIZE = 15
COMBINED_FIGURE_TITLE_FONTSIZE = 24
COMBINED_PANEL_SIZE_INCHES = 4.9
COMBINED_COLORBAR_WIDTH_RATIO = 0.14
COMBINED_GRID_WSPACE = 0.10
COMBINED_GRID_HSPACE = 0.14
COMBINED_FIG_LEFT = 0.07
COMBINED_FIG_RIGHT = 0.95
COMBINED_FIG_BOTTOM = 0.10
COMBINED_FIG_TOP = 0.90


REQUIRED_SWEEP_COLUMNS = {
    "step",
    "noise_strength",
    "cfg_weight",
}
REQUIRED_LPIPS_COLUMNS = REQUIRED_SWEEP_COLUMNS | {
    "lpips_mean",
    "lpips_std",
}
REQUIRED_FID_COLUMNS = REQUIRED_SWEEP_COLUMNS | {
    "fid",
}
REQUIRED_CLIP_FID_COLUMNS = REQUIRED_SWEEP_COLUMNS | {
    "clip_fid",
}
REQUIRED_DEEPLAB_FD_COLUMNS = REQUIRED_SWEEP_COLUMNS | {
    "deeplab_fd",
}


def load_and_validate_dataframe(
    csv_path: str | Path,
    required_columns: set[str],
    csv_label: str,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_label} CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_dataframe_columns(df, required_columns, csv_label=csv_label)
    return df


def default_out_dir(csv_path: str | Path, share_color_scale: bool) -> Path:
    csv_path = Path(csv_path)
    dir_name = "heatmaps_shared" if share_color_scale else "heatmaps"
    return csv_path.parent / dir_name


def resolve_out_dirs(
    csv_paths: list[str | Path],
    out_dirs: list[str | Path] | None,
    share_color_scale: bool,
) -> list[Path]:
    if out_dirs is None:
        return [
            default_out_dir(csv_path=csv_path, share_color_scale=share_color_scale)
            for csv_path in csv_paths
        ]

    if len(out_dirs) != len(csv_paths):
        raise ValueError(
            "If `out_dirs` is provided, it must contain one output directory per CSV path."
        )

    return [Path(out_dir) for out_dir in out_dirs]


def compute_shared_value_range(
    csv_paths: list[str | Path],
    value_col: str,
    required_columns: set[str],
    csv_label: str,
) -> tuple[float, float]:
    if not csv_paths:
        raise ValueError("Expected at least one CSV path to compute a shared color scale.")

    mins: list[float] = []
    maxs: list[float] = []
    for csv_path in csv_paths:
        df = load_and_validate_dataframe(
            csv_path=csv_path,
            required_columns=required_columns,
            csv_label=csv_label,
        )
        mins.append(float(df[value_col].min()))
        maxs.append(float(df[value_col].max()))

    return min(mins), max(maxs)


def default_combined_save_path(
    csv_paths: list[str | Path],
    metric_name: str,
    share_color_scale: bool,
) -> Path:
    if not csv_paths:
        raise ValueError("Expected at least one CSV path to resolve a combined output path.")
    return default_out_dir(
        csv_path=csv_paths[0],
        share_color_scale=share_color_scale,
    ) / f"{metric_name}_heatmap_grid.pdf"


def format_checkpoint_label(step: int) -> str:
    if step >= 1000:
        value = step / 1000.0
        if value.is_integer():
            return f"{int(value)}k"
        return f"{value:g}k"
    return str(step)


def resolve_panel_titles(
    panel_titles: list[str] | None,
    steps: list[int],
) -> list[str]:
    if panel_titles is not None:
        if len(panel_titles) != len(steps):
            raise ValueError(
                "If `panel_titles` is provided, it must contain one title per CSV path."
            )
        return panel_titles

    return [f"Checkpoint {format_checkpoint_label(step)}" for step in steps]


def load_single_step_panel_data(
    csv_paths: list[str | Path],
    value_col: str,
    required_columns: set[str],
    csv_label: str,
    annot_builder=None,
    panel_titles: list[str] | None = None,
) -> list[dict[str, object]]:
    panels: list[dict[str, object]] = []
    steps: list[int] = []
    dataframes: list[tuple[Path, pd.DataFrame, int]] = []

    for csv_path in csv_paths:
        path = Path(csv_path)
        df = load_and_validate_dataframe(
            csv_path=path,
            required_columns=required_columns,
            csv_label=csv_label,
        )
        unique_steps = sorted(int(step) for step in df["step"].unique())
        if len(unique_steps) != 1:
            raise ValueError(
                "Combined heatmap mode expects each CSV to contain exactly one unique "
                f"`step`. Found steps {unique_steps} in {path}."
            )

        step = unique_steps[0]
        steps.append(step)
        dataframes.append((path, df.loc[df["step"] == step].copy(), step))

    titles = resolve_panel_titles(panel_titles=panel_titles, steps=steps)

    for title, (path, step_df, step) in zip(titles, dataframes):
        panel = {
            "csv_path": path,
            "step": step,
            "grid_df": build_metric_grid(step_df, value_col=value_col),
            "annot_df": annot_builder(step_df) if annot_builder is not None else None,
            "title": title,
        }
        panels.append(panel)

    return panels


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
    cbar_labelpad: float = 8.0,
    linewidths: float = 0.3,
    linecolor: str = "white",
    square: bool = False,
    mask: pd.DataFrame | None = None,
    ax=None,
    cbar: bool = True,
    title_fontsize: float = HEATMAP_TITLE_FONTSIZE,
    axis_label_fontsize: float = HEATMAP_AXIS_LABEL_FONTSIZE,
    tick_label_fontsize: float = HEATMAP_TICK_LABEL_FONTSIZE,
    annotation_fontsize: float = HEATMAP_ANNOTATION_FONTSIZE,
    colorbar_label_fontsize: float = HEATMAP_COLORBAR_LABEL_FONTSIZE,
    colorbar_tick_fontsize: float = HEATMAP_COLORBAR_TICK_FONTSIZE,
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
        cbar_labelpad: Extra padding between the colorbar and its label.
        linewidths: Width of cell separators.
        linecolor: Color of cell separators.
        square: Whether to force square cells.
        mask: Optional boolean mask with the same shape as `grid_df`.
        ax: Optional existing matplotlib axis to draw into.
        cbar: Whether to draw a colorbar for this heatmap.
        title_fontsize: Font size for the panel title.
        axis_label_fontsize: Font size for x/y axis labels.
        tick_label_fontsize: Font size for axis tick labels.
        annotation_fontsize: Font size for cell annotations.
        colorbar_label_fontsize: Font size for the colorbar label.
        colorbar_tick_fontsize: Font size for the colorbar tick labels.
    """
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

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
        cbar=cbar,
        mask=mask,
        annot_kws={"fontsize": annotation_fontsize},
    )

    if square:
        ax.set_box_aspect(grid_df.shape[0] / grid_df.shape[1])

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)

    # Keep row/column labels readable for small sweep grids.
    ax.tick_params(axis="x", rotation=0, labelsize=tick_label_fontsize)
    ax.tick_params(axis="y", rotation=0, labelsize=tick_label_fontsize)

    if cbar_label is not None:
        colorbar = hm.collections[0].colorbar
        if colorbar is not None:
            colorbar.set_label(
                cbar_label,
                labelpad=cbar_labelpad,
                fontsize=colorbar_label_fontsize,
            )
            colorbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    if save_path is not None and created_figure:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    elif save_path is None and created_figure:
        fig.show()

    return fig, ax


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    csv_label: str,
) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            f"{csv_label} CSV is missing required columns: " + ", ".join(missing)
        )


def build_metric_grid(
    step_df: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    grid_df = step_df.pivot(index="noise_strength", columns="cfg_weight", values=value_col)
    grid_df = grid_df.sort_index(axis=0).sort_index(axis=1)
    grid_df.index.name = "noise_strength"
    grid_df.columns.name = "cfg_weight"
    return grid_df


def build_label_grid(
    step_df: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    return build_metric_grid(step_df, value_col=label_col)


def build_lpips_annotation_grid(
    step_df: pd.DataFrame,
    include_std: bool = True,
    mean_fmt: str = ".3f",
    std_fmt: str = ".3f",
) -> pd.DataFrame:
    label_df = step_df.copy()
    label_df["label"] = label_df["lpips_mean"].map(lambda value: format(value, mean_fmt))
    if include_std:
        label_df["label"] = (
            label_df["label"]
            + "\n"
            + r"$\pm$"
            + label_df["lpips_std"].map(lambda value: format(value, std_fmt))
        )
    return build_label_grid(label_df, label_col="label")


def build_fid_annotation_grid(
    step_df: pd.DataFrame,
    fid_fmt: str = ".2f",
) -> pd.DataFrame:
    label_df = step_df.copy()
    label_df["label"] = label_df["fid"].map(lambda value: format(value, fid_fmt))
    return build_label_grid(label_df, label_col="label")


def build_clip_fid_annotation_grid(
    step_df: pd.DataFrame,
    clip_fid_fmt: str = ".2f",
) -> pd.DataFrame:
    label_df = step_df.copy()
    label_df["label"] = label_df["clip_fid"].map(lambda value: format(value, clip_fid_fmt))
    return build_label_grid(label_df, label_col="label")


def build_deeplab_fd_annotation_grid(
    step_df: pd.DataFrame,
    deeplab_fd_fmt: str = ".2f",
) -> pd.DataFrame:
    label_df = step_df.copy()
    label_df["label"] = label_df["deeplab_fd"].map(lambda value: format(value, deeplab_fd_fmt))
    return build_label_grid(label_df, label_col="label")


def plot_lpips_heatmap_for_step(
    step_df: pd.DataFrame,
    step: int,
    save_path: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    include_std: bool = True,
    vmin=None,
    vmax=None,
):
    grid_df = build_metric_grid(step_df, value_col="lpips_mean")
    annot_df = build_lpips_annotation_grid(step_df, include_std=include_std) if annot else None

    title = f"LPIPS Across CFG Weight and Noise Strength (step {step})"
    fig, ax = plot_heatmap(
        grid_df=grid_df,
        title=title,
        xlabel="CFG Weight ($w$)",
        ylabel="Noise Strength ($\\gamma$)",
        save_path=save_path,
        cmap=cmap,
        annot=annot_df if annot else False,
        fmt="",
        vmin=vmin,
        vmax=vmax,
        cbar_label="LPIPS $\\downarrow$",
    )
    return fig, ax


def plot_fid_heatmap_for_step(
    step_df: pd.DataFrame,
    step: int,
    save_path: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    vmin=None,
    vmax=None,
):
    grid_df = build_metric_grid(step_df, value_col="fid")
    annot_df = build_fid_annotation_grid(step_df) if annot else None

    title = f"FID Across CFG Weight and Noise Strength (step {step})"
    fig, ax = plot_heatmap(
        grid_df=grid_df,
        title=title,
        xlabel="CFG Weight ($w$)",
        ylabel="Noise Strength ($\\gamma$)",
        save_path=save_path,
        cmap=cmap,
        annot=annot_df if annot else False,
        fmt="",
        vmin=vmin,
        vmax=vmax,
        cbar_label="FID $\\downarrow$",
    )
    return fig, ax


def plot_clip_fid_heatmap_for_step(
    step_df: pd.DataFrame,
    step: int,
    save_path: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    vmin=None,
    vmax=None,
):
    grid_df = build_metric_grid(step_df, value_col="clip_fid")
    annot_df = build_clip_fid_annotation_grid(step_df) if annot else None

    title = f"CLIP-FID Across CFG Weight and Noise Strength (step {step})"
    fig, ax = plot_heatmap(
        grid_df=grid_df,
        title=title,
        xlabel="CFG Weight ($w$)",
        ylabel="Noise Strength ($\\gamma$)",
        save_path=save_path,
        cmap=cmap,
        annot=annot_df if annot else False,
        fmt="",
        vmin=vmin,
        vmax=vmax,
        cbar_label="CLIP-FID $\\downarrow$",
    )
    return fig, ax


def plot_deeplab_fd_heatmap_for_step(
    step_df: pd.DataFrame,
    step: int,
    save_path: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    vmin=None,
    vmax=None,
):
    grid_df = build_metric_grid(step_df, value_col="deeplab_fd")
    annot_df = build_deeplab_fd_annotation_grid(step_df) if annot else None

    title = f"DeepLab FD Across CFG Weight and Noise Strength (step {step})"
    fig, ax = plot_heatmap(
        grid_df=grid_df,
        title=title,
        xlabel="CFG Weight ($w$)",
        ylabel="Noise Strength ($\\gamma$)",
        save_path=save_path,
        cmap=cmap,
        annot=annot_df if annot else False,
        fmt="",
        vmin=vmin,
        vmax=vmax,
        cbar_label="DeepLab FD $\\downarrow$",
    )
    return fig, ax


def save_combined_heatmap_grid(
    panels: list[dict[str, object]],
    save_path: str | Path,
    xlabel: str,
    ylabel: str,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    figure_title: str | None = None,
    cbar_labelpad: float = 8.0,
    linewidths: float = 0.3,
    linecolor: str = "white",
    square: bool = False,
) -> Path:
    if not panels:
        raise ValueError("Expected at least one panel to save a combined heatmap grid.")

    n_panels = len(panels)
    n_cols = min(2, max(1, math.ceil(math.sqrt(n_panels))))
    n_rows = math.ceil(n_panels / n_cols)

    fig_width = COMBINED_PANEL_SIZE_INCHES * n_cols + COMBINED_PANEL_SIZE_INCHES * COMBINED_COLORBAR_WIDTH_RATIO + 1.2
    fig_height = COMBINED_PANEL_SIZE_INCHES * n_rows + 1.3
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid_spec = fig.add_gridspec(
        n_rows,
        n_cols + 1,
        width_ratios=[1.0] * n_cols + [COMBINED_COLORBAR_WIDTH_RATIO],
        wspace=COMBINED_GRID_WSPACE,
        hspace=COMBINED_GRID_HSPACE,
    )

    flat_axes = np.empty(n_rows * n_cols, dtype=object)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            flat_axes[row_idx * n_cols + col_idx] = fig.add_subplot(grid_spec[row_idx, col_idx])
    used_axes = flat_axes[:n_panels]
    colorbar_ax = fig.add_subplot(grid_spec[:, -1])

    for idx, (ax, panel) in enumerate(zip(used_axes, panels)):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        show_xlabel = row_idx == n_rows - 1
        show_ylabel = col_idx == 0

        plot_heatmap(
            grid_df=panel["grid_df"],
            title=str(panel["title"]),
            xlabel=xlabel if show_xlabel else "",
            ylabel=ylabel if show_ylabel else "",
            cmap=cmap,
            annot=panel["annot_df"] if panel["annot_df"] is not None else False,
            fmt="",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            ax=ax,
            cbar_labelpad=cbar_labelpad,
            linewidths=linewidths,
            linecolor=linecolor,
            square=True,
        )

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=plt.get_cmap(cmap)),
        cax=colorbar_ax,
    )
    colorbar.set_label(
        cbar_label,
        labelpad=cbar_labelpad,
        fontsize=HEATMAP_COLORBAR_LABEL_FONTSIZE,
    )
    colorbar.ax.tick_params(labelsize=HEATMAP_COLORBAR_TICK_FONTSIZE)

    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=COMBINED_FIGURE_TITLE_FONTSIZE)

    fig.subplots_adjust(
        left=COMBINED_FIG_LEFT,
        right=COMBINED_FIG_RIGHT,
        bottom=COMBINED_FIG_BOTTOM,
        top=COMBINED_FIG_TOP,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def save_lpips_heatmaps_from_csv(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    include_std: bool = True,
    share_color_scale: bool = True,
    shared_vmin=None,
    shared_vmax=None,
) -> list[Path]:
    csv_path = Path(csv_path)
    df = load_and_validate_dataframe(
        csv_path=csv_path,
        required_columns=REQUIRED_LPIPS_COLUMNS,
        csv_label="LPIPS stats",
    )

    out_root = (
        Path(out_dir)
        if out_dir is not None
        else default_out_dir(csv_path=csv_path, share_color_scale=share_color_scale)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if shared_vmin is not None or shared_vmax is not None:
        vmin = shared_vmin
        vmax = shared_vmax
    else:
        vmin = float(df["lpips_mean"].min()) if share_color_scale else None
        vmax = float(df["lpips_mean"].max()) if share_color_scale else None
    saved_paths: list[Path] = []

    for step in sorted(df["step"].unique()):
        step_df = df.loc[df["step"] == step].copy()
        if step_df.empty:
            continue

        save_path = out_root / f"lpips_heatmap_step-{int(step)}.pdf"
        plot_lpips_heatmap_for_step(
            step_df=step_df,
            step=int(step),
            save_path=save_path,
            cmap=cmap,
            annot=annot,
            include_std=include_std,
            vmin=vmin,
            vmax=vmax,
        )
        saved_paths.append(save_path)

    if not saved_paths:
        raise ValueError(f"No LPIPS heatmaps were generated from {csv_path}")

    return saved_paths


def save_lpips_heatmap_grid_from_csvs(
    csv_paths: list[str | Path],
    save_path: str | Path | None = None,
    panel_titles: list[str] | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    include_std: bool = True,
    share_color_scale: bool = True,
) -> Path:
    if not share_color_scale:
        raise ValueError("Combined LPIPS heatmap mode requires `share_color_scale=True`.")

    panels = load_single_step_panel_data(
        csv_paths=csv_paths,
        value_col="lpips_mean",
        required_columns=REQUIRED_LPIPS_COLUMNS,
        csv_label="LPIPS stats",
        annot_builder=(
            lambda step_df: build_lpips_annotation_grid(step_df, include_std=include_std)
            if annot
            else None
        ),
        panel_titles=panel_titles,
    )
    vmin, vmax = compute_shared_value_range(
        csv_paths=csv_paths,
        value_col="lpips_mean",
        required_columns=REQUIRED_LPIPS_COLUMNS,
        csv_label="LPIPS stats",
    )

    resolved_save_path = (
        Path(save_path)
        if save_path is not None
        else default_combined_save_path(
            csv_paths=csv_paths,
            metric_name="lpips",
            share_color_scale=share_color_scale,
        )
    )
    return save_combined_heatmap_grid(
        panels=panels,
        save_path=resolved_save_path,
        xlabel="CFG Weight ($w$)",
        ylabel="Noise Strength ($\\gamma$)",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_label="LPIPS $\\downarrow$",
        figure_title="LPIPS Across CFG Weight and Noise Strength",
    )


def save_lpips_heatmaps_from_csvs(
    csv_paths: list[str | Path],
    out_dirs: list[str | Path] | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    include_std: bool = True,
    share_color_scale: bool = True,
) -> list[Path]:
    resolved_out_dirs = resolve_out_dirs(
        csv_paths=csv_paths,
        out_dirs=out_dirs,
        share_color_scale=share_color_scale,
    )
    shared_vmin = None
    shared_vmax = None
    if share_color_scale:
        shared_vmin, shared_vmax = compute_shared_value_range(
            csv_paths=csv_paths,
            value_col="lpips_mean",
            required_columns=REQUIRED_LPIPS_COLUMNS,
            csv_label="LPIPS stats",
        )

    saved_paths: list[Path] = []
    for csv_path, out_dir in zip(csv_paths, resolved_out_dirs):
        saved_paths.extend(
            save_lpips_heatmaps_from_csv(
                csv_path=csv_path,
                out_dir=out_dir,
                cmap=cmap,
                annot=annot,
                include_std=include_std,
                share_color_scale=share_color_scale,
                shared_vmin=shared_vmin,
                shared_vmax=shared_vmax,
            )
        )

    return saved_paths


def save_fid_heatmaps_from_csv(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
    shared_vmin=None,
    shared_vmax=None,
) -> list[Path]:
    csv_path = Path(csv_path)
    df = load_and_validate_dataframe(
        csv_path=csv_path,
        required_columns=REQUIRED_FID_COLUMNS,
        csv_label="FID stats",
    )

    out_root = (
        Path(out_dir)
        if out_dir is not None
        else default_out_dir(csv_path=csv_path, share_color_scale=share_color_scale)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if shared_vmin is not None or shared_vmax is not None:
        vmin = shared_vmin
        vmax = shared_vmax
    else:
        vmin = float(df["fid"].min()) if share_color_scale else None
        vmax = float(df["fid"].max()) if share_color_scale else None
    saved_paths: list[Path] = []

    for step in sorted(df["step"].unique()):
        step_df = df.loc[df["step"] == step].copy()
        if step_df.empty:
            continue

        save_path = out_root / f"fid_heatmap_step-{int(step)}.pdf"
        plot_fid_heatmap_for_step(
            step_df=step_df,
            step=int(step),
            save_path=save_path,
            cmap=cmap,
            annot=annot,
            vmin=vmin,
            vmax=vmax,
        )
        saved_paths.append(save_path)

    if not saved_paths:
        raise ValueError(f"No FID heatmaps were generated from {csv_path}")

    return saved_paths


def save_fid_heatmap_grid_from_csvs(
    csv_paths: list[str | Path],
    save_path: str | Path | None = None,
    panel_titles: list[str] | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
) -> Path:
    if not share_color_scale:
        raise ValueError("Combined FID heatmap mode requires `share_color_scale=True`.")

    panels = load_single_step_panel_data(
        csv_paths=csv_paths,
        value_col="fid",
        required_columns=REQUIRED_FID_COLUMNS,
        csv_label="FID stats",
        annot_builder=build_fid_annotation_grid if annot else None,
        panel_titles=panel_titles,
    )
    vmin, vmax = compute_shared_value_range(
        csv_paths=csv_paths,
        value_col="fid",
        required_columns=REQUIRED_FID_COLUMNS,
        csv_label="FID stats",
    )

    resolved_save_path = (
        Path(save_path)
        if save_path is not None
        else default_combined_save_path(
            csv_paths=csv_paths,
            metric_name="fid",
            share_color_scale=share_color_scale,
        )
    )
    return save_combined_heatmap_grid(
        panels=panels,
        save_path=resolved_save_path,
        xlabel="CFG Weight ($w$)",
        ylabel="Noise Strength ($\\gamma$)",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_label="FID $\\downarrow$",
        figure_title="FID Across CFG Weight and Noise Strength",
    )


def save_fid_heatmaps_from_csvs(
    csv_paths: list[str | Path],
    out_dirs: list[str | Path] | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
) -> list[Path]:
    resolved_out_dirs = resolve_out_dirs(
        csv_paths=csv_paths,
        out_dirs=out_dirs,
        share_color_scale=share_color_scale,
    )
    shared_vmin = None
    shared_vmax = None
    if share_color_scale:
        shared_vmin, shared_vmax = compute_shared_value_range(
            csv_paths=csv_paths,
            value_col="fid",
            required_columns=REQUIRED_FID_COLUMNS,
            csv_label="FID stats",
        )

    saved_paths: list[Path] = []
    for csv_path, out_dir in zip(csv_paths, resolved_out_dirs):
        saved_paths.extend(
            save_fid_heatmaps_from_csv(
                csv_path=csv_path,
                out_dir=out_dir,
                cmap=cmap,
                annot=annot,
                share_color_scale=share_color_scale,
                shared_vmin=shared_vmin,
                shared_vmax=shared_vmax,
            )
        )

    return saved_paths


def save_clip_fid_heatmaps_from_csv(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
    shared_vmin=None,
    shared_vmax=None,
) -> list[Path]:
    csv_path = Path(csv_path)
    df = load_and_validate_dataframe(
        csv_path=csv_path,
        required_columns=REQUIRED_CLIP_FID_COLUMNS,
        csv_label="CLIP-FID stats",
    )

    out_root = (
        Path(out_dir)
        if out_dir is not None
        else default_out_dir(csv_path=csv_path, share_color_scale=share_color_scale)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if shared_vmin is not None or shared_vmax is not None:
        vmin = shared_vmin
        vmax = shared_vmax
    else:
        vmin = float(df["clip_fid"].min()) if share_color_scale else None
        vmax = float(df["clip_fid"].max()) if share_color_scale else None
    saved_paths: list[Path] = []

    for step in sorted(df["step"].unique()):
        step_df = df.loc[df["step"] == step].copy()
        if step_df.empty:
            continue

        save_path = out_root / f"clip_fid_heatmap_step-{int(step)}.pdf"
        plot_clip_fid_heatmap_for_step(
            step_df=step_df,
            step=int(step),
            save_path=save_path,
            cmap=cmap,
            annot=annot,
            vmin=vmin,
            vmax=vmax,
        )
        saved_paths.append(save_path)

    if not saved_paths:
        raise ValueError(f"No CLIP-FID heatmaps were generated from {csv_path}")

    return saved_paths


def save_clip_fid_heatmap_grid_from_csvs(
    csv_paths: list[str | Path],
    save_path: str | Path | None = None,
    panel_titles: list[str] | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
) -> Path:
    if not share_color_scale:
        raise ValueError("Combined CLIP-FID heatmap mode requires `share_color_scale=True`.")

    panels = load_single_step_panel_data(
        csv_paths=csv_paths,
        value_col="clip_fid",
        required_columns=REQUIRED_CLIP_FID_COLUMNS,
        csv_label="CLIP-FID stats",
        annot_builder=build_clip_fid_annotation_grid if annot else None,
        panel_titles=panel_titles,
    )
    vmin, vmax = compute_shared_value_range(
        csv_paths=csv_paths,
        value_col="clip_fid",
        required_columns=REQUIRED_CLIP_FID_COLUMNS,
        csv_label="CLIP-FID stats",
    )

    resolved_save_path = (
        Path(save_path)
        if save_path is not None
        else default_combined_save_path(
            csv_paths=csv_paths,
            metric_name="clip_fid",
            share_color_scale=share_color_scale,
        )
    )
    return save_combined_heatmap_grid(
        panels=panels,
        save_path=resolved_save_path,
        xlabel="CFG Weight ($w$)",
        ylabel="Noise Strength ($\\gamma$)",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_label="CLIP-FID $\\downarrow$",
        figure_title="CLIP-FID Across CFG Weight and Noise Strength",
    )


def save_clip_fid_heatmaps_from_csvs(
    csv_paths: list[str | Path],
    out_dirs: list[str | Path] | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
) -> list[Path]:
    resolved_out_dirs = resolve_out_dirs(
        csv_paths=csv_paths,
        out_dirs=out_dirs,
        share_color_scale=share_color_scale,
    )
    shared_vmin = None
    shared_vmax = None
    if share_color_scale:
        shared_vmin, shared_vmax = compute_shared_value_range(
            csv_paths=csv_paths,
            value_col="clip_fid",
            required_columns=REQUIRED_CLIP_FID_COLUMNS,
            csv_label="CLIP-FID stats",
        )

    saved_paths: list[Path] = []
    for csv_path, out_dir in zip(csv_paths, resolved_out_dirs):
        saved_paths.extend(
            save_clip_fid_heatmaps_from_csv(
                csv_path=csv_path,
                out_dir=out_dir,
                cmap=cmap,
                annot=annot,
                share_color_scale=share_color_scale,
                shared_vmin=shared_vmin,
                shared_vmax=shared_vmax,
            )
        )

    return saved_paths


def save_deeplab_fd_heatmaps_from_csv(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
    shared_vmin=None,
    shared_vmax=None,
) -> list[Path]:
    csv_path = Path(csv_path)
    df = load_and_validate_dataframe(
        csv_path=csv_path,
        required_columns=REQUIRED_DEEPLAB_FD_COLUMNS,
        csv_label="DeepLab FD stats",
    )

    out_root = (
        Path(out_dir)
        if out_dir is not None
        else default_out_dir(csv_path=csv_path, share_color_scale=share_color_scale)
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if shared_vmin is not None or shared_vmax is not None:
        vmin = shared_vmin
        vmax = shared_vmax
    else:
        vmin = float(df["deeplab_fd"].min()) if share_color_scale else None
        vmax = float(df["deeplab_fd"].max()) if share_color_scale else None
    saved_paths: list[Path] = []

    for step in sorted(df["step"].unique()):
        step_df = df.loc[df["step"] == step].copy()
        if step_df.empty:
            continue

        save_path = out_root / f"deeplab_fd_heatmap_step-{int(step)}.pdf"
        plot_deeplab_fd_heatmap_for_step(
            step_df=step_df,
            step=int(step),
            save_path=save_path,
            cmap=cmap,
            annot=annot,
            vmin=vmin,
            vmax=vmax,
        )
        saved_paths.append(save_path)

    if not saved_paths:
        raise ValueError(f"No DeepLab FD heatmaps were generated from {csv_path}")

    return saved_paths


def save_deeplab_fd_heatmap_grid_from_csvs(
    csv_paths: list[str | Path],
    save_path: str | Path | None = None,
    panel_titles: list[str] | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
) -> Path:
    if not share_color_scale:
        raise ValueError("Combined DeepLab FD heatmap mode requires `share_color_scale=True`.")

    panels = load_single_step_panel_data(
        csv_paths=csv_paths,
        value_col="deeplab_fd",
        required_columns=REQUIRED_DEEPLAB_FD_COLUMNS,
        csv_label="DeepLab FD stats",
        annot_builder=build_deeplab_fd_annotation_grid if annot else None,
        panel_titles=panel_titles,
    )
    vmin, vmax = compute_shared_value_range(
        csv_paths=csv_paths,
        value_col="deeplab_fd",
        required_columns=REQUIRED_DEEPLAB_FD_COLUMNS,
        csv_label="DeepLab FD stats",
    )

    resolved_save_path = (
        Path(save_path)
        if save_path is not None
        else default_combined_save_path(
            csv_paths=csv_paths,
            metric_name="deeplab_fd",
            share_color_scale=share_color_scale,
        )
    )
    return save_combined_heatmap_grid(
        panels=panels,
        save_path=resolved_save_path,
        xlabel="CFG Weight ($w$)",
        ylabel="Noise Strength ($\\gamma$)",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_label="DeepLab FD $\\downarrow$",
        figure_title="DeepLab FD Across CFG Weight and Noise Strength",
    )


def save_deeplab_fd_heatmaps_from_csvs(
    csv_paths: list[str | Path],
    out_dirs: list[str | Path] | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
) -> list[Path]:
    resolved_out_dirs = resolve_out_dirs(
        csv_paths=csv_paths,
        out_dirs=out_dirs,
        share_color_scale=share_color_scale,
    )
    shared_vmin = None
    shared_vmax = None
    if share_color_scale:
        shared_vmin, shared_vmax = compute_shared_value_range(
            csv_paths=csv_paths,
            value_col="deeplab_fd",
            required_columns=REQUIRED_DEEPLAB_FD_COLUMNS,
            csv_label="DeepLab FD stats",
        )

    saved_paths: list[Path] = []
    for csv_path, out_dir in zip(csv_paths, resolved_out_dirs):
        saved_paths.extend(
            save_deeplab_fd_heatmaps_from_csv(
                csv_path=csv_path,
                out_dir=out_dir,
                cmap=cmap,
                annot=annot,
                share_color_scale=share_color_scale,
                shared_vmin=shared_vmin,
                shared_vmax=shared_vmax,
            )
        )

    return saved_paths


def main() -> None:
    # Metric type to visualize from a sweep CSV. Supported values: "lpips",
    # "fid", "clip_fid", or "deeplab_fd".
    metric = "lpips"

    # One or more summary CSVs generated by the corresponding metric script.
    # Add multiple paths here to share one colorbar range across all outputs.
    csv_paths = [
        Path("/develop/data/remote_sensing/tiled/projection/cyclenet_sim_proj/oem_only/ema/step-2500/lpips_stats.csv"),
        Path("/develop/data/remote_sensing/tiled/projection/cyclenet_sim_proj/all_real/pareto/step-10000/lpips_stats.csv"),
        Path("/develop/data/remote_sensing/tiled/projection/cyclenet_sim_proj/all_real/pareto/step-20000/lpips_stats.csv"),
        Path("/develop/data/remote_sensing/tiled/projection/cyclenet_sim_proj/all_real_ft_invar/step-30000/lpips_stats.csv"),
    ]

    # Optional output directories matching `csv_paths`. Leave as `None` to
    # write into `heatmaps_shared/` when sharing color scales, else `heatmaps/`.
    out_dirs = None

    # Output mode:
    # - "individual": save one heatmap file per CSV
    # - "combined": save one multi-panel figure with a shared colorbar
    save_mode = "individual"

    # Combined-mode only: optional panel titles matching `csv_paths`. Leave as
    # `None` to use the single `step` found in each CSV.
    panel_titles = None

    # Combined-mode only: destination for the single output figure. Leave as
    # `None` to use a default file under `heatmaps_shared/`.
    combined_save_path = None

    # Whether to annotate each heatmap cell with the metric value.
    annot = True

    # LPIPS-only: whether annotations should include `+- std` under the mean value.
    include_std = False

    # Whether all step heatmaps should share the same metric color scale.
    share_color_scale = False

    # Colormap for metric heatmaps; reversed so lower values appear more favorable.
    cmap = "viridis_r"

    if save_mode not in {"individual", "combined"}:
        raise ValueError("`save_mode` must be either 'individual' or 'combined'.")
    if save_mode == "combined" and out_dirs is not None:
        raise ValueError("`out_dirs` is only used when `save_mode='individual'`.")
    if save_mode == "combined" and not share_color_scale:
        raise ValueError("`save_mode='combined'` requires `share_color_scale=True`.")

    metric_name = metric.lower()
    if metric_name == "lpips":
        if save_mode == "combined":
            saved_paths = [
                save_lpips_heatmap_grid_from_csvs(
                    csv_paths=csv_paths,
                    save_path=combined_save_path,
                    panel_titles=panel_titles,
                    cmap=cmap,
                    annot=annot,
                    include_std=include_std,
                    share_color_scale=share_color_scale,
                )
            ]
        else:
            saved_paths = save_lpips_heatmaps_from_csvs(
                csv_paths=csv_paths,
                out_dirs=out_dirs,
                cmap=cmap,
                annot=annot,
                include_std=include_std,
                share_color_scale=share_color_scale,
            )
    elif metric_name == "fid":
        if save_mode == "combined":
            saved_paths = [
                save_fid_heatmap_grid_from_csvs(
                    csv_paths=csv_paths,
                    save_path=combined_save_path,
                    panel_titles=panel_titles,
                    cmap=cmap,
                    annot=annot,
                    share_color_scale=share_color_scale,
                )
            ]
        else:
            saved_paths = save_fid_heatmaps_from_csvs(
                csv_paths=csv_paths,
                out_dirs=out_dirs,
                cmap=cmap,
                annot=annot,
                share_color_scale=share_color_scale,
            )
    elif metric_name == "clip_fid":
        if save_mode == "combined":
            saved_paths = [
                save_clip_fid_heatmap_grid_from_csvs(
                    csv_paths=csv_paths,
                    save_path=combined_save_path,
                    panel_titles=panel_titles,
                    cmap=cmap,
                    annot=annot,
                    share_color_scale=share_color_scale,
                )
            ]
        else:
            saved_paths = save_clip_fid_heatmaps_from_csvs(
                csv_paths=csv_paths,
                out_dirs=out_dirs,
                cmap=cmap,
                annot=annot,
                share_color_scale=share_color_scale,
            )
    elif metric_name == "deeplab_fd":
        if save_mode == "combined":
            saved_paths = [
                save_deeplab_fd_heatmap_grid_from_csvs(
                    csv_paths=csv_paths,
                    save_path=combined_save_path,
                    panel_titles=panel_titles,
                    cmap=cmap,
                    annot=annot,
                    share_color_scale=share_color_scale,
                )
            ]
        else:
            saved_paths = save_deeplab_fd_heatmaps_from_csvs(
                csv_paths=csv_paths,
                out_dirs=out_dirs,
                cmap=cmap,
                annot=annot,
                share_color_scale=share_color_scale,
            )
    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. Expected 'lpips', 'fid', 'clip_fid', or 'deeplab_fd'."
        )

    for path in saved_paths:
        print(f"Saved {metric_name.upper()} heatmap to {path}")


if __name__ == "__main__":
    main()
