from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .set_style import apply_style

apply_style()


REQUIRED_METRIC_COLUMNS = {
    "checkpoint_name",
    "step",
    "noise_strength",
    "cfg_weight",
    "fid",
    "lpips_mean",
    "deeplab_fd",
}


def validate_columns(df: pd.DataFrame, required_columns: set[str], csv_label: str) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{csv_label} CSV is missing required columns: {', '.join(missing)}")


def compute_pareto_mask(df: pd.DataFrame, metric_cols: list[str]) -> pd.Series:
    values = df[metric_cols].to_numpy(dtype=float)
    is_pareto = [True] * len(df)

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if (values[j] <= values[i]).all() and (values[j] < values[i]).any():
                is_pareto[i] = False
                break

    return pd.Series(is_pareto, index=df.index)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"})


def _ordered_checkpoint_names(df: pd.DataFrame) -> list[str]:
    ordered = (
        df.loc[:, ["checkpoint_name", "step"]]
        .drop_duplicates()
        .sort_values(["step", "checkpoint_name"], ascending=[True, True])
    )
    return ordered["checkpoint_name"].tolist()


def load_checkpoint_metrics(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Merged checkpoint metrics CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df, REQUIRED_METRIC_COLUMNS, csv_label="Merged checkpoint metrics")

    df = df.copy()
    if "is_pareto_deeplab_lpips" in df.columns:
        df["is_pareto_deeplab_lpips"] = _coerce_bool_series(df["is_pareto_deeplab_lpips"])
    else:
        df["is_pareto_deeplab_lpips"] = False

    if "is_pareto_fid_deeplab" in df.columns:
        df["is_pareto_fid_deeplab"] = _coerce_bool_series(df["is_pareto_fid_deeplab"])
    else:
        df["is_pareto_fid_deeplab"] = False

    if "is_pareto_3d" in df.columns:
        df["is_pareto_3d"] = _coerce_bool_series(df["is_pareto_3d"])
    else:
        df["is_pareto_3d"] = False

    if "is_selected" in df.columns:
        df["is_selected"] = _coerce_bool_series(df["is_selected"])
    else:
        df["is_selected"] = False

    # Recompute Pareto flags per checkpoint if they are absent in the CSV.
    if not df["is_pareto_deeplab_lpips"].any():
        for checkpoint_name in _ordered_checkpoint_names(df):
            mask = df["checkpoint_name"] == checkpoint_name
            df.loc[mask, "is_pareto_deeplab_lpips"] = compute_pareto_mask(
                df.loc[mask],
                ["deeplab_fd", "lpips_mean"],
            )

    if not df["is_pareto_fid_deeplab"].any():
        for checkpoint_name in _ordered_checkpoint_names(df):
            mask = df["checkpoint_name"] == checkpoint_name
            df.loc[mask, "is_pareto_fid_deeplab"] = compute_pareto_mask(
                df.loc[mask],
                ["fid", "deeplab_fd"],
            )

    if not df["is_pareto_3d"].any():
        for checkpoint_name in _ordered_checkpoint_names(df):
            mask = df["checkpoint_name"] == checkpoint_name
            df.loc[mask, "is_pareto_3d"] = compute_pareto_mask(
                df.loc[mask],
                ["fid", "lpips_mean", "deeplab_fd"],
            )

    return df


def compute_fid_deeplab_then_lpips_selection(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_selected_fid_deeplab_lpips"] = False

    for checkpoint_name in _ordered_checkpoint_names(df):
        checkpoint_mask = df["checkpoint_name"] == checkpoint_name
        candidates = df.loc[
            checkpoint_mask & df["is_pareto_fid_deeplab"]
        ].copy()
        if candidates.empty:
            continue

        selected_row = candidates.sort_values(
            ["lpips_mean", "deeplab_fd", "fid", "noise_strength", "cfg_weight"],
            ascending=[True, True, True, True, True],
        ).iloc[0]
        df.loc[selected_row.name, "is_selected_fid_deeplab_lpips"] = True

    return df


def _build_checkpoint_palette(checkpoint_names: list[str]) -> dict[str, tuple[float, float, float]]:
    palette = sns.color_palette("tab10", n_colors=max(len(checkpoint_names), 3))
    return {
        checkpoint_name: palette[idx]
        for idx, checkpoint_name in enumerate(checkpoint_names)
    }


def _build_selected_label(row: pd.Series) -> str:
    return (
        f"{row['checkpoint_name']} "
        f"($\\gamma={float(row['noise_strength']):g}$, $w={float(row['cfg_weight']):g}$)"
    )


def plot_pairwise_pareto_panel(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    pareto_col: str,
    palette: dict[str, tuple[float, float, float]],
    annotate_selected: bool = True,
    title: str | None = None,
    title_fontsize: float | None = None,
) -> None:
    checkpoint_names = _ordered_checkpoint_names(df)

    for checkpoint_name in checkpoint_names:
        checkpoint_df = df.loc[df["checkpoint_name"] == checkpoint_name].copy()
        color = palette[checkpoint_name]

        ax.scatter(
            checkpoint_df[x_col],
            checkpoint_df[y_col],
            s=34,
            alpha=0.35,
            color=color,
            edgecolors="none",
            label=checkpoint_name,
            zorder=1,
        )

        pareto_df = checkpoint_df.loc[checkpoint_df[pareto_col]].copy()
        if not pareto_df.empty:
            pareto_df = pareto_df.sort_values(x_col)
            ax.plot(
                pareto_df[x_col],
                pareto_df[y_col],
                color=color,
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                pareto_df[x_col],
                pareto_df[y_col],
                s=58,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                zorder=3,
            )

        selected_df = checkpoint_df.loc[checkpoint_df["is_selected"]].copy()
        if not selected_df.empty:
            ax.scatter(
                selected_df[x_col],
                selected_df[y_col],
                s=95,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.7,
                zorder=4,
            )
            if annotate_selected:
                row = selected_df.iloc[0]
                ax.annotate(
                    _build_selected_label(row),
                    (float(row[x_col]), float(row[y_col])),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)


def plot_pairwise_pareto_grid(
    df: pd.DataFrame,
    save_path: str | Path,
    annotate_selected: bool = True,
    title_fontsize: float | None = None,
) -> Path:
    checkpoint_names = _ordered_checkpoint_names(df)
    palette = _build_checkpoint_palette(checkpoint_names)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))

    plot_pairwise_pareto_panel(
        ax=axes[0],
        df=df,
        x_col="lpips_mean",
        y_col="deeplab_fd",
        x_label="LPIPS ($\\downarrow$)",
        y_label="DeepLab FD ($\\downarrow$)",
        pareto_col="is_pareto_deeplab_lpips",
        palette=palette,
        annotate_selected=annotate_selected,
        title="Preservation vs Task-Aware Alignment",
        title_fontsize=title_fontsize,
    )
    plot_pairwise_pareto_panel(
        ax=axes[1],
        df=df,
        x_col="lpips_mean",
        y_col="fid",
        x_label="LPIPS ($\\downarrow$)",
        y_label="FID ($\\downarrow$)",
        pareto_col="is_pareto_3d",
        palette=palette,
        annotate_selected=annotate_selected,
        title="Preservation vs Generic Realism",
        title_fontsize=title_fontsize,
    )
    plot_pairwise_pareto_panel(
        ax=axes[2],
        df=df,
        x_col="deeplab_fd",
        y_col="fid",
        x_label="DeepLab FD ($\\downarrow$)",
        y_label="FID ($\\downarrow$)",
        pareto_col="is_pareto_3d",
        palette=palette,
        annotate_selected=annotate_selected,
        title="Task-Aware vs Generic Realism",
        title_fontsize=title_fontsize,
    )

    handles = [
        mlines.Line2D(
            [],
            [],
            color=palette[checkpoint_name],
            marker="o",
            linestyle="None",
            markersize=6,
            label=f"Checkpoint {checkpoint_name}",
        )
        for checkpoint_name in checkpoint_names
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markersize=6,
            label="Pareto-optimal",
        )
    )
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="*",
            linestyle="None",
            markersize=8,
            label="Selected",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=1,
        frameon=True,
        bbox_to_anchor=(0.5, 1.03),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_fid_lpips_tradeoff(
    df: pd.DataFrame,
    save_path: str | Path,
    title: str = "FID vs. LPIPS",
    title_fontsize: float | None = None,
    pareto_col: str = "is_pareto_3d",
) -> Path:
    checkpoint_names = _ordered_checkpoint_names(df)
    palette = _build_checkpoint_palette(checkpoint_names)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    for checkpoint_name in checkpoint_names:
        checkpoint_df = df.loc[df["checkpoint_name"] == checkpoint_name].copy()
        color = palette[checkpoint_name]

        ax.scatter(
            checkpoint_df["lpips_mean"],
            checkpoint_df["fid"],
            s=34,
            alpha=0.35,
            color=color,
            edgecolors="none",
            zorder=1,
        )

        pareto_df = checkpoint_df.loc[checkpoint_df[pareto_col]].copy()
        if not pareto_df.empty:
            pareto_df = pareto_df.sort_values("lpips_mean")
            ax.plot(
                pareto_df["lpips_mean"],
                pareto_df["fid"],
                color=color,
                linewidth=1.0,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                pareto_df["lpips_mean"],
                pareto_df["fid"],
                s=58,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                zorder=3,
            )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("LPIPS ($\\downarrow$)")
    ax.set_ylabel("FID ($\\downarrow$)")
    ax.grid(alpha=0.25)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=palette[checkpoint_name],
            marker="o",
            linestyle="None",
            markersize=6,
            label=f"Checkpoint {checkpoint_name}",
        )
        for checkpoint_name in checkpoint_names
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markersize=6,
            label="Pareto-optimal",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper right",
        ncol=1,
        frameon=True,
        bbox_to_anchor=(0.98, 0.98),
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_3d_pareto_scatter(
    df: pd.DataFrame,
    save_path: str | Path,
    elev: float = 24.0,
    azim: float = -52.0,
    annotate_selected: bool = False,
    title_fontsize: float | None = None,
    title_pad: float | None = None,
) -> Path:
    checkpoint_names = _ordered_checkpoint_names(df)
    palette = _build_checkpoint_palette(checkpoint_names)

    fig = plt.figure(figsize=(8.6, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    for checkpoint_name in checkpoint_names:
        checkpoint_df = df.loc[df["checkpoint_name"] == checkpoint_name].copy()
        color = palette[checkpoint_name]

        ax.scatter(
            checkpoint_df["lpips_mean"],
            checkpoint_df["fid"],
            checkpoint_df["deeplab_fd"],
            s=26,
            alpha=0.28,
            color=color,
            depthshade=False,
            label=checkpoint_name,
        )

        pareto_df = checkpoint_df.loc[checkpoint_df["is_pareto_3d"]].copy()
        if not pareto_df.empty:
            ax.scatter(
                pareto_df["lpips_mean"],
                pareto_df["fid"],
                pareto_df["deeplab_fd"],
                s=58,
                facecolors="none",
                edgecolors=[color],
                linewidths=1.2,
                depthshade=False,
            )

        selected_df = checkpoint_df.loc[checkpoint_df["is_selected"]].copy()
        if not selected_df.empty:
            ax.scatter(
                selected_df["lpips_mean"],
                selected_df["fid"],
                selected_df["deeplab_fd"],
                s=120,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.7,
                depthshade=False,
            )
            if annotate_selected:
                row = selected_df.iloc[0]
                ax.text(
                    float(row["lpips_mean"]),
                    float(row["fid"]),
                    float(row["deeplab_fd"]),
                    _build_selected_label(row),
                    fontsize=8,
                )

    ax.set_xlabel("LPIPS ($\\downarrow$)")
    ax.set_ylabel("FID ($\\downarrow$)")
    ax.set_zlabel("DeepLab FD ($\\downarrow$)", labelpad=12)
    ax.set_title(
        "3D Tradeoff Across Checkpoints",
        fontsize=title_fontsize,
        pad=title_pad,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", frameon=True)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # `tight_layout()` plus the project-wide `savefig.bbox="tight"` setting
    # can clip 3D z-axis labels during export, so reserve margin explicitly.
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.02, top=0.94)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_fid_deeplab_then_lpips_story(
    df: pd.DataFrame,
    save_path: str | Path,
    elev: float = 24.0,
    azim: float = 118.0,
    annotate_selected: bool = True,
    panel_title_fontsize: float | None = None,
    suptitle_fontsize: float | None = None,
    legend_fontsize: float | None = None,
    story_3d_title_pad: float | None = None,
    story_3d_title_y: float | None = None,
) -> Path:
    df = compute_fid_deeplab_then_lpips_selection(df)
    checkpoint_names = _ordered_checkpoint_names(df)
    palette = _build_checkpoint_palette(checkpoint_names)

    fig = plt.figure(figsize=(13.4, 7.4))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.75, 1.0],
        height_ratios=[1.0, 0.82],
        wspace=0.18,
        hspace=0.34,
    )

    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    # Keep the same camera orientation while moving the vertical axis
    # to the opposite edge so its tick labels sit next to DeepLabv3-FID.
    ax3d.zaxis._axinfo["juggled"] = (1, 2, 0)
    ax_front = fig.add_subplot(grid[0, 1])
    ax_lpips = fig.add_subplot(grid[1, 1])

    for checkpoint_name in checkpoint_names:
        checkpoint_df = df.loc[df["checkpoint_name"] == checkpoint_name].copy()
        color = palette[checkpoint_name]
        front_df = checkpoint_df.loc[checkpoint_df["is_pareto_fid_deeplab"]].copy()
        selected_df = checkpoint_df.loc[
            checkpoint_df["is_selected_fid_deeplab_lpips"]
        ].copy()

        ax3d.scatter(
            checkpoint_df["deeplab_fd"],
            checkpoint_df["lpips_mean"],
            checkpoint_df["fid"],
            s=20,
            alpha=0.14,
            color=color,
            depthshade=False,
        )
        if not front_df.empty:
            front_3d_df = front_df.sort_values(["deeplab_fd", "lpips_mean", "fid"])
            ax3d.plot(
                front_3d_df["deeplab_fd"],
                front_3d_df["lpips_mean"],
                front_3d_df["fid"],
                color=color,
                linewidth=1.0,
                alpha=0.8,
            )
            ax3d.scatter(
                front_df["deeplab_fd"],
                front_df["lpips_mean"],
                front_df["fid"],
                s=64,
                facecolors="none",
                edgecolors=[color],
                linewidths=1.3,
                depthshade=False,
            )
        if not selected_df.empty:
            row = selected_df.iloc[0]
            ax3d.scatter(
                [float(row["deeplab_fd"])],
                [float(row["lpips_mean"])],
                [float(row["fid"])],
                s=135,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.7,
                depthshade=False,
            )

        ax_front.scatter(
            checkpoint_df["deeplab_fd"],
            checkpoint_df["fid"],
            s=28,
            alpha=0.18,
            color=color,
            edgecolors="none",
            zorder=1,
        )
        if not front_df.empty:
            front_2d_df = front_df.sort_values("deeplab_fd")
            ax_front.plot(
                front_2d_df["deeplab_fd"],
                front_2d_df["fid"],
                color=color,
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )
            ax_front.scatter(
                front_df["deeplab_fd"],
                front_df["fid"],
                s=62,
                facecolors="none",
                edgecolors=color,
                linewidths=1.3,
                zorder=3,
            )
        if not selected_df.empty:
            row = selected_df.iloc[0]
            ax_front.scatter(
                [float(row["deeplab_fd"])],
                [float(row["fid"])],
                s=110,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.7,
                zorder=4,
            )

    checkpoint_to_y = {
        checkpoint_name: idx for idx, checkpoint_name in enumerate(checkpoint_names)
    }
    for checkpoint_name in checkpoint_names:
        checkpoint_df = df.loc[df["checkpoint_name"] == checkpoint_name].copy()
        color = palette[checkpoint_name]
        front_df = checkpoint_df.loc[checkpoint_df["is_pareto_fid_deeplab"]].copy()
        if front_df.empty:
            continue

        front_df = front_df.sort_values(
            ["lpips_mean", "deeplab_fd", "fid", "noise_strength", "cfg_weight"],
            ascending=[True, True, True, True, True],
        )
        y = checkpoint_to_y[checkpoint_name]
        ax_lpips.plot(
            front_df["lpips_mean"],
            [y] * len(front_df),
            color=color,
            linewidth=1.0,
            alpha=0.6,
            zorder=1,
        )
        ax_lpips.scatter(
            front_df["lpips_mean"],
            [y] * len(front_df),
            s=40,
            color=color,
            alpha=0.7,
            zorder=2,
        )

        selected_df = front_df.loc[front_df["is_selected_fid_deeplab_lpips"]].copy()
        if not selected_df.empty:
            row = selected_df.iloc[0]
            ax_lpips.scatter(
                [float(row["lpips_mean"])],
                [y],
                s=120,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.7,
                zorder=3,
            )
            if annotate_selected:
                ax_lpips.annotate(
                    f"$\\gamma={float(row['noise_strength']):g}$, $w={float(row['cfg_weight']):g}$",
                    (float(row["lpips_mean"]), y),
                    xytext=(7, 6),
                    textcoords="offset points",
                    fontsize=8,
                    va="bottom",
                )

    ax3d.set_xlabel("DeepLabv3-FID ($\\downarrow$)")
    ax3d.set_ylabel("LPIPS ($\\downarrow$)")
    ax3d.set_zlabel("FID ($\\downarrow$)", labelpad=12)
    ax3d.zaxis.set_rotate_label(False)
    ax3d.zaxis.label.set_rotation(90)
    ax3d.set_title(
        "3D Tradeoff Overview",
        fontsize=panel_title_fontsize,
        pad=story_3d_title_pad,
    )
    if story_3d_title_y is not None:
        ax3d.title.set_y(story_3d_title_y)
    ax3d.invert_yaxis()
    ax3d.view_init(elev=elev, azim=azim)

    ax_front.set_title(
        "Pass 1: (FID, DeepLabv3-FID) Pareto Front",
        fontsize=panel_title_fontsize,
        pad=10.0,
    )
    ax_front.set_xlabel("DeepLabv3-FID ($\\downarrow$)")
    ax_front.set_ylabel("FID ($\\downarrow$)")
    ax_front.grid(alpha=0.25)

    ax_lpips.set_title("Pass 2: Lowest LPIPS", fontsize=panel_title_fontsize, pad=10.0)
    ax_lpips.set_xlabel("LPIPS ($\\downarrow$)")
    ax_lpips.set_ylabel("Checkpoint")
    ax_lpips.set_yticks(list(checkpoint_to_y.values()))
    ax_lpips.set_yticklabels(checkpoint_names)
    ax_lpips.grid(axis="x", alpha=0.25)
    ax_lpips.set_axisbelow(True)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=palette[checkpoint_name],
            marker="o",
            linestyle="None",
            markersize=6,
            label=checkpoint_name,
        )
        for checkpoint_name in checkpoint_names
    ]
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markersize=6,
            label="(FID, DeepLabv3-FID) Pareto front",
        )
    )
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="*",
            linestyle="None",
            markersize=8,
            label="Selected model configuration",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper left",
        ncol=1,
        frameon=True,
        bbox_to_anchor=(0.02, 0.985),
        fontsize=legend_fontsize,
    )
    fig.suptitle(
        "Multi-Objective Model Configuration Selection",
        y=0.995,
        fontsize=suptitle_fontsize,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Reserve margins explicitly because the 3D axis and figure-level legend
    # do not interact reliably with the project's global tight save bbox.
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.08, top=0.89)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(save_path)
    plt.close(fig)
    return save_path


def save_all_pareto_plots(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    annotate_selected_pairwise: bool = True,
    annotate_selected_3d: bool = False,
    pairwise_title_fontsize: float | None = None,
    scatter_3d_title_fontsize: float | None = None,
    scatter_3d_title_pad: float | None = None,
    story_panel_title_fontsize: float | None = None,
    story_suptitle_fontsize: float | None = None,
    story_legend_fontsize: float | None = None,
    story_3d_title_pad: float | None = None,
    story_3d_title_y: float | None = None,
) -> list[Path]:
    df = load_checkpoint_metrics(csv_path)
    out_root = Path(out_dir) if out_dir is not None else Path(csv_path).parent / "pareto_plots"
    out_root.mkdir(parents=True, exist_ok=True)

    saved_paths = [
        plot_pairwise_pareto_grid(
            df=df,
            save_path=out_root / "pareto_pairwise_grid.pdf",
            annotate_selected=annotate_selected_pairwise,
            title_fontsize=pairwise_title_fontsize,
        ),
        plot_fid_lpips_tradeoff(
            df=df,
            save_path=out_root / "pareto_fid_lpips_tradeoff.pdf",
            title_fontsize=pairwise_title_fontsize,
        ),
        plot_3d_pareto_scatter(
            df=df,
            save_path=out_root / "pareto_3d_scatter.pdf",
            annotate_selected=annotate_selected_3d,
            title_fontsize=scatter_3d_title_fontsize,
            title_pad=scatter_3d_title_pad,
        ),
        plot_fid_deeplab_then_lpips_story(
            df=df,
            save_path=out_root / "pareto_fid_deeplab_then_lpips_story.pdf",
            annotate_selected=annotate_selected_pairwise,
            panel_title_fontsize=story_panel_title_fontsize,
            suptitle_fontsize=story_suptitle_fontsize,
            legend_fontsize=story_legend_fontsize,
            story_3d_title_pad=story_3d_title_pad,
            story_3d_title_y=story_3d_title_y,
        ),
    ]
    return saved_paths


def main() -> None:
    # Path to the merged checkpoint-metrics CSV produced by
    # `scripts/analyze_checkpoint_metrics.py`.
    csv_path = Path("/cgi/data/nvesd/workspaces/logan/code/pixel-cyclenet/eval/checkpoints/fid_deeplab/checkpoint_metrics_merged.csv")

    # Directory where the Pareto tradeoff figures will be written.
    out_dir = csv_path.parent / "pareto_plots"

    # Whether to annotate the selected operating points in the pairwise grid.
    annotate_selected_pairwise = True

    # Whether to annotate the selected operating points in the 3D scatter.
    # This is usually noisier than the pairwise figure.
    annotate_selected_3d = False

    # Panel-title font size for the pairwise Pareto grid.
    pairwise_title_fontsize = 13.5

    # Title font size for the standalone 3D Pareto scatter.
    scatter_3d_title_fontsize = 14.0

    # Extra padding between the standalone 3D axes and its title.
    scatter_3d_title_pad = 8.0

    # Title font size for the story figure's 3D and 2D panels.
    story_panel_title_fontsize = 14.0

    # Font size for the story figure's overall title.
    story_suptitle_fontsize = 15.0

    # Font size for the story figure legend.
    story_legend_fontsize = 10.0

    # Padding between the story figure's 3D axes and its title.
    story_3d_title_pad = 0.0

    # Manual vertical position for the story figure's 3D title. Lower values
    # move the title closer to the plot when `pad` is no longer sufficient.
    story_3d_title_y = 0.965

    saved_paths = save_all_pareto_plots(
        csv_path=csv_path,
        out_dir=out_dir,
        annotate_selected_pairwise=annotate_selected_pairwise,
        annotate_selected_3d=annotate_selected_3d,
        pairwise_title_fontsize=pairwise_title_fontsize,
        scatter_3d_title_fontsize=scatter_3d_title_fontsize,
        scatter_3d_title_pad=scatter_3d_title_pad,
        story_panel_title_fontsize=story_panel_title_fontsize,
        story_suptitle_fontsize=story_suptitle_fontsize,
        story_legend_fontsize=story_legend_fontsize,
        story_3d_title_pad=story_3d_title_pad,
        story_3d_title_y=story_3d_title_y,
    )

    for path in saved_paths:
        print(f"Saved Pareto plot to {path}")


if __name__ == "__main__":
    main()
