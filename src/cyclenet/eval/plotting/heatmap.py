from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .set_style import apply_style

apply_style()


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
            colorbar.set_label(cbar_label, labelpad=cbar_labelpad)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
    else:
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


def save_lpips_heatmaps_from_csv(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    include_std: bool = True,
    share_color_scale: bool = True,
) -> list[Path]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"LPIPS stats CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_dataframe_columns(df, REQUIRED_LPIPS_COLUMNS, csv_label="LPIPS stats")

    out_root = Path(out_dir) if out_dir is not None else csv_path.parent / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

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


def save_fid_heatmaps_from_csv(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
) -> list[Path]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"FID stats CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_dataframe_columns(df, REQUIRED_FID_COLUMNS, csv_label="FID stats")

    out_root = Path(out_dir) if out_dir is not None else csv_path.parent / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

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


def save_clip_fid_heatmaps_from_csv(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
) -> list[Path]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CLIP-FID stats CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_dataframe_columns(df, REQUIRED_CLIP_FID_COLUMNS, csv_label="CLIP-FID stats")

    out_root = Path(out_dir) if out_dir is not None else csv_path.parent / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

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


def save_deeplab_fd_heatmaps_from_csv(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    cmap: str = "viridis_r",
    annot: bool = True,
    share_color_scale: bool = True,
) -> list[Path]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"DeepLab FD stats CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_dataframe_columns(df, REQUIRED_DEEPLAB_FD_COLUMNS, csv_label="DeepLab FD stats")

    out_root = Path(out_dir) if out_dir is not None else csv_path.parent / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

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


def main() -> None:
    # Metric type to visualize from a sweep CSV. Supported values: "lpips",
    # "fid", "clip_fid", or "deeplab_fd".
    metric = "lpips"

    # Path to the summary CSV generated by either `src/cyclenet/eval/lpips.py`
    # or `src/cyclenet/eval/fid.py`.
    csv_path = Path("/develop/data/remote_sensing/tiled/projection/cyclenet_sim_proj/all_real_ft_invar/lpips_stats.csv")

    # Directory where per-step heatmaps will be written.
    out_dir = csv_path.parent / "heatmaps"

    # Whether to annotate each heatmap cell with the metric value.
    annot = True

    # LPIPS-only: whether annotations should include `+- std` under the mean value.
    include_std = False

    # Whether all step heatmaps should share the same metric color scale.
    share_color_scale = True

    # Colormap for metric heatmaps; reversed so lower values appear more favorable.
    cmap = "viridis_r"

    metric_name = metric.lower()
    if metric_name == "lpips":
        saved_paths = save_lpips_heatmaps_from_csv(
            csv_path=csv_path,
            out_dir=out_dir,
            cmap=cmap,
            annot=annot,
            include_std=include_std,
            share_color_scale=share_color_scale,
        )
    elif metric_name == "fid":
        saved_paths = save_fid_heatmaps_from_csv(
            csv_path=csv_path,
            out_dir=out_dir,
            cmap=cmap,
            annot=annot,
            share_color_scale=share_color_scale,
        )
    elif metric_name == "clip_fid":
        saved_paths = save_clip_fid_heatmaps_from_csv(
            csv_path=csv_path,
            out_dir=out_dir,
            cmap=cmap,
            annot=annot,
            share_color_scale=share_color_scale,
        )
    elif metric_name == "deeplab_fd":
        saved_paths = save_deeplab_fd_heatmaps_from_csv(
            csv_path=csv_path,
            out_dir=out_dir,
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
