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
        for checkpoint_name in sorted(df["checkpoint_name"].unique()):
            mask = df["checkpoint_name"] == checkpoint_name
            df.loc[mask, "is_pareto_deeplab_lpips"] = compute_pareto_mask(
                df.loc[mask],
                ["deeplab_fd", "lpips_mean"],
            )

    if not df["is_pareto_3d"].any():
        for checkpoint_name in sorted(df["checkpoint_name"].unique()):
            mask = df["checkpoint_name"] == checkpoint_name
            df.loc[mask, "is_pareto_3d"] = compute_pareto_mask(
                df.loc[mask],
                ["fid", "lpips_mean", "deeplab_fd"],
            )

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
) -> None:
    checkpoint_names = sorted(df["checkpoint_name"].unique())

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
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)


def plot_pairwise_pareto_grid(
    df: pd.DataFrame,
    save_path: str | Path,
    annotate_selected: bool = True,
) -> Path:
    checkpoint_names = sorted(df["checkpoint_name"].unique())
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
    )

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
        ncol=min(len(handles), 6),
        frameon=True,
        bbox_to_anchor=(0.5, 1.03),
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
) -> Path:
    checkpoint_names = sorted(df["checkpoint_name"].unique())
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
    ax.set_zlabel("DeepLab FD ($\\downarrow$)")
    ax.set_title("3D Tradeoff Across Checkpoints")
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper left", frameon=True)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def save_all_pareto_plots(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    annotate_selected_pairwise: bool = True,
    annotate_selected_3d: bool = False,
) -> list[Path]:
    df = load_checkpoint_metrics(csv_path)
    out_root = Path(out_dir) if out_dir is not None else Path(csv_path).parent / "pareto_plots"
    out_root.mkdir(parents=True, exist_ok=True)

    saved_paths = [
        plot_pairwise_pareto_grid(
            df=df,
            save_path=out_root / "pareto_pairwise_grid.pdf",
            annotate_selected=annotate_selected_pairwise,
        ),
        plot_3d_pareto_scatter(
            df=df,
            save_path=out_root / "pareto_3d_scatter.pdf",
            annotate_selected=annotate_selected_3d,
        ),
    ]
    return saved_paths


def main() -> None:
    # Path to the merged checkpoint-metrics CSV produced by
    # `scripts/analyze_checkpoint_metrics.py`.
    csv_path = Path("/path/to/checkpoint_metrics_merged.csv")

    # Directory where the Pareto tradeoff figures will be written.
    out_dir = csv_path.parent / "pareto_plots"

    # Whether to annotate the selected operating points in the pairwise grid.
    annotate_selected_pairwise = True

    # Whether to annotate the selected operating points in the 3D scatter.
    # This is usually noisier than the pairwise figure.
    annotate_selected_3d = False

    saved_paths = save_all_pareto_plots(
        csv_path=csv_path,
        out_dir=out_dir,
        annotate_selected_pairwise=annotate_selected_pairwise,
        annotate_selected_3d=annotate_selected_3d,
    )

    for path in saved_paths:
        print(f"Saved Pareto plot to {path}")


if __name__ == "__main__":
    main()
