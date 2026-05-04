from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
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

METRIC_LABELS = {
    "fid": "FID",
    "lpips_mean": "LPIPS",
    "deeplab_fd": "DeepLabv3-FID",
    "boundary_edge_ratio_mean": "Boundary-Edge Ratio",
    "boundary_edge_inverse_ratio_mean": "Boundary-Edge Inverse Ratio",
    "boundary_edge_contrast_mean": "Boundary-Edge Contrast",
}

PARETO_METRIC_SPECS = {
    "deeplab_lpips": {
        "metric_cols": ["deeplab_fd", "lpips_mean"],
        "pareto_col": "is_pareto_deeplab_lpips",
        "candidate_col": "is_candidate_deeplab_lpips",
    },
    "fid_lpips": {
        "metric_cols": ["fid", "lpips_mean"],
        "pareto_col": "is_pareto_fid_lpips",
        "candidate_col": "is_candidate_fid_lpips",
    },
    "fid_deeplab": {
        "metric_cols": ["fid", "deeplab_fd"],
        "pareto_col": "is_pareto_fid_deeplab",
        "candidate_col": "is_candidate_fid_deeplab",
    },
    "deeplab_boundary": {
        "metric_cols": ["deeplab_fd", "boundary_edge_inverse_ratio_mean"],
        "pareto_col": "is_pareto_deeplab_boundary",
        "candidate_col": "is_candidate_deeplab_boundary",
    },
    "fid_boundary": {
        "metric_cols": ["fid", "boundary_edge_inverse_ratio_mean"],
        "pareto_col": "is_pareto_fid_boundary",
        "candidate_col": "is_candidate_fid_boundary",
    },
    "3d": {
        "metric_cols": ["fid", "lpips_mean", "deeplab_fd"],
        "pareto_col": "is_pareto_3d",
        "candidate_col": "is_candidate_3d",
    },
    "3d_boundary": {
        "metric_cols": ["fid", "deeplab_fd", "boundary_edge_inverse_ratio_mean"],
        "pareto_col": "is_pareto_3d_boundary",
        "candidate_col": "is_candidate_3d_boundary",
    },
}

SELECTION_MODE_SPECS = {
    "fid_lpips_pareto_then_deeplab": {
        "pareto_key": "fid_lpips",
        "pass1_metric_cols": ["fid", "lpips_mean"],
        "tie_break_metric": "deeplab_fd",
        "display_title": "FID / LPIPS Pareto Then DeepLabv3-FID",
        "story_supported": True,
    },
    "fid_deeplab_pareto_then_lpips": {
        "pareto_key": "fid_deeplab",
        "pass1_metric_cols": ["fid", "deeplab_fd"],
        "tie_break_metric": "lpips_mean",
        "display_title": "FID / DeepLabv3-FID Pareto Then LPIPS",
        "story_supported": True,
    },
    "deeplab_lpips_pareto_then_fid": {
        "pareto_key": "deeplab_lpips",
        "pass1_metric_cols": ["deeplab_fd", "lpips_mean"],
        "tie_break_metric": "fid",
        "display_title": "DeepLabv3-FID / LPIPS Pareto Then FID",
        "story_supported": True,
    },
    "three_metric_pareto_then_deeplab": {
        "pareto_key": "3d",
        "pass1_metric_cols": ["fid", "lpips_mean", "deeplab_fd"],
        "tie_break_metric": "deeplab_fd",
        "display_title": "3-Metric Pareto Then DeepLabv3-FID",
        "story_supported": False,
    },
    "best_deeplab_fd": {
        "display_title": "Best DeepLabv3-FID",
        "story_supported": False,
    },
    "deeplab_boundary_pareto_then_fid": {
        "pareto_key": "deeplab_boundary",
        "pass1_metric_cols": ["deeplab_fd", "boundary_edge_inverse_ratio_mean"],
        "tie_break_metric": "fid",
        "display_title": "DeepLabv3-FID / Boundary-Edge Inverse Ratio Pareto Then FID",
        "story_supported": True,
    },
    "deeplab_boundary_pareto_then_knee": {
        "pareto_key": "deeplab_boundary",
        "pass1_metric_cols": ["deeplab_fd", "boundary_edge_inverse_ratio_mean"],
        "display_title": "DeepLabv3-FID / Boundary-Edge Inverse Ratio Pareto Knee",
        "story_supported": False,
        "highlight_exact_pareto": True,
    },
    "fid_boundary_pareto_then_knee": {
        "pareto_key": "fid_boundary",
        "pass1_metric_cols": ["fid", "boundary_edge_inverse_ratio_mean"],
        "display_title": "FID / Boundary-Edge Inverse Ratio Pareto Knee",
        "story_supported": False,
        "highlight_exact_pareto": True,
    },
    "fid_deeplab_boundary_pareto_then_knee": {
        "pareto_key": "3d_boundary",
        "pass1_metric_cols": ["fid", "deeplab_fd", "boundary_edge_inverse_ratio_mean"],
        "display_title": "FID / DeepLabv3-FID / Boundary-Edge Inverse Ratio Pareto Ideal-Point",
        "story_supported": False,
        "highlight_exact_pareto": True,
    },
}

DEFAULT_SELECTION_MODE_ORDER = [
    "fid_lpips_pareto_then_deeplab",
    "fid_deeplab_pareto_then_lpips",
    "deeplab_lpips_pareto_then_fid",
    "deeplab_boundary_pareto_then_fid",
    "deeplab_boundary_pareto_then_knee",
    "fid_boundary_pareto_then_knee",
    "fid_deeplab_boundary_pareto_then_knee",
    "three_metric_pareto_then_deeplab",
    "best_deeplab_fd",
]

# Whether to annotate the selected operating points in the pairwise grid.
annotate_selected_pairwise = True

# Whether to annotate the selected operating points in the 3D scatter.
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


def compute_pareto_candidate_mask(
    df: pd.DataFrame,
    metric_cols: list[str],
    pareto_threshold_pct: float | None,
) -> pd.Series:
    exact_pareto_mask = compute_pareto_mask(df, metric_cols)

    if pareto_threshold_pct is None or pareto_threshold_pct <= 0:
        return exact_pareto_mask

    values = df[metric_cols].to_numpy(dtype=float)
    exact_front = values[exact_pareto_mask.to_numpy()]
    tolerance_factor = 1.0 + (pareto_threshold_pct / 100.0)
    is_candidate: list[bool] = []

    for row in values:
        row_is_candidate = False
        for front_point in exact_front:
            tolerance = np.where(
                front_point == 0.0,
                front_point,
                front_point * tolerance_factor,
            )
            if (row <= tolerance).all():
                row_is_candidate = True
                break
        is_candidate.append(row_is_candidate)

    return pd.Series(is_candidate, index=df.index)


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


def _selection_groups(df: pd.DataFrame) -> list[pd.Index]:
    if "selection_scope" in df.columns and df["selection_scope"].nunique() == 1:
        if str(df["selection_scope"].iloc[0]).lower() == "overall":
            return [df.index]
    return [df.index[df["checkpoint_name"] == name] for name in _ordered_checkpoint_names(df)]


def _metric_label(metric_col: str) -> str:
    return METRIC_LABELS.get(metric_col, metric_col)


def _require_metric_columns(df: pd.DataFrame, metric_cols: list[str], context: str) -> None:
    missing = [metric for metric in metric_cols if metric not in df.columns]
    if missing:
        raise ValueError(
            f"{context} requires metric column(s) not present in the merged CSV: "
            f"{', '.join(missing)}"
        )


def _is_boundary_selection_mode(selection_mode: str) -> bool:
    return selection_mode.lower() in {
        "deeplab_boundary_pareto_then_fid",
        "deeplab_boundary_pareto_then_knee",
        "fid_boundary_pareto_then_knee",
        "fid_deeplab_boundary_pareto_then_knee",
    }


def _selection_pairwise_specs(selection_mode: str) -> list[dict[str, str]]:
    if _is_boundary_selection_mode(selection_mode):
        return [
            {
                "x_col": "boundary_edge_inverse_ratio_mean",
                "y_col": "deeplab_fd",
                "title": "Structure Preservation vs Task-Aware Alignment",
            },
            {
                "x_col": "boundary_edge_inverse_ratio_mean",
                "y_col": "fid",
                "title": "Structure Preservation vs Generic Realism",
            },
            {
                "x_col": "deeplab_fd",
                "y_col": "fid",
                "title": "Task-Aware vs Generic Realism",
            },
        ]

    return [
        {
            "x_col": "lpips_mean",
            "y_col": "deeplab_fd",
            "title": "Preservation vs Task-Aware Alignment",
        },
        {
            "x_col": "lpips_mean",
            "y_col": "fid",
            "title": "Preservation vs Generic Realism",
        },
        {
            "x_col": "deeplab_fd",
            "y_col": "fid",
            "title": "Task-Aware vs Generic Realism",
        },
    ]


def _selection_3d_metric_cols(selection_mode: str) -> list[str]:
    if _is_boundary_selection_mode(selection_mode):
        return ["boundary_edge_inverse_ratio_mean", "fid", "deeplab_fd"]
    return ["lpips_mean", "fid", "deeplab_fd"]


def _selection_tradeoff_axes(selection_mode: str) -> tuple[str, str]:
    if _is_boundary_selection_mode(selection_mode):
        return "boundary_edge_inverse_ratio_mean", "fid"
    return "lpips_mean", "fid"


def selected_col_name(selection_mode: str) -> str:
    return f"is_selected__{selection_mode.lower()}"


def get_selection_mode_spec(selection_mode: str) -> dict[str, object]:
    mode_key = selection_mode.lower()
    if mode_key not in SELECTION_MODE_SPECS:
        supported = ", ".join(sorted(SELECTION_MODE_SPECS))
        raise ValueError(
            f"Unsupported selection_mode '{selection_mode}'. Supported modes: {supported}"
        )

    spec = dict(SELECTION_MODE_SPECS[mode_key])
    pareto_key = spec.get("pareto_key")
    if pareto_key is not None:
        pareto_spec = PARETO_METRIC_SPECS[str(pareto_key)]
        spec["pareto_col"] = pareto_spec["pareto_col"]
        spec["candidate_col"] = pareto_spec["candidate_col"]
    if bool(spec.get("highlight_exact_pareto", False)):
        spec["highlight_col"] = spec.get("pareto_col")
    spec["selected_col"] = selected_col_name(mode_key)
    return spec


def discover_selection_modes(df: pd.DataFrame) -> list[str]:
    discovered = [
        column.removeprefix("is_selected__")
        for column in df.columns
        if column.startswith("is_selected__")
    ]
    ordered = [mode for mode in DEFAULT_SELECTION_MODE_ORDER if mode in discovered]
    extras = sorted(mode for mode in discovered if mode not in ordered)
    return ordered + extras


def _pareto_threshold_pct(df: pd.DataFrame) -> float:
    if "pareto_threshold_pct" not in df.columns:
        return 0.0

    values = df["pareto_threshold_pct"].dropna()
    if values.empty:
        return 0.0
    return float(values.iloc[0])


def _selection_candidate_label(selection_mode: str, pareto_threshold_pct: float) -> str | None:
    spec = get_selection_mode_spec(selection_mode)
    highlight_col = spec.get("highlight_col")
    if highlight_col is None and "candidate_col" not in spec:
        return None

    pass1_metric_cols = list(spec.get("pass1_metric_cols", []))
    metric_label = ", ".join(_metric_label(metric) for metric in pass1_metric_cols)
    if bool(spec.get("highlight_exact_pareto", False)):
        return f"({metric_label}) Pareto front"
    if pareto_threshold_pct > 0:
        return f"({metric_label}) Pareto pool (+{pareto_threshold_pct:g}%)"
    return f"({metric_label}) Pareto front"


def _selected_label() -> str:
    return "Selected configuration"


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


def load_checkpoint_metrics(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Merged checkpoint metrics CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df, REQUIRED_METRIC_COLUMNS, csv_label="Merged checkpoint metrics")
    df = df.copy()

    for column in df.columns:
        if column.startswith("is_"):
            df[column] = _coerce_bool_series(df[column])

    if "pareto_threshold_pct" not in df.columns:
        df["pareto_threshold_pct"] = 0.0

    pareto_threshold_pct = _pareto_threshold_pct(df)
    selection_groups = _selection_groups(df)

    for pareto_spec in PARETO_METRIC_SPECS.values():
        pareto_col = pareto_spec["pareto_col"]
        candidate_col = pareto_spec["candidate_col"]

        if pareto_col not in df.columns:
            df[pareto_col] = False
            for group_index in selection_groups:
                df.loc[group_index, pareto_col] = compute_pareto_mask(
                    df.loc[group_index],
                    metric_cols=pareto_spec["metric_cols"],
                )

        if candidate_col not in df.columns:
            df[candidate_col] = False
            for group_index in selection_groups:
                df.loc[group_index, candidate_col] = compute_pareto_candidate_mask(
                    df.loc[group_index],
                    metric_cols=pareto_spec["metric_cols"],
                    pareto_threshold_pct=pareto_threshold_pct,
                )

    if "is_selected" not in df.columns:
        df["is_selected"] = False

    return df


def _build_common_handles(
    checkpoint_names: list[str],
    palette: dict[str, tuple[float, float, float]],
    candidate_label: str | None,
) -> list[mlines.Line2D]:
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

    if candidate_label is not None:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markersize=6,
                label=candidate_label,
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
            label=_selected_label(),
        )
    )
    return handles


def plot_pairwise_pareto_panel(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    highlight_col: str | None,
    selected_col: str,
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
            zorder=1,
        )

        if highlight_col is not None:
            highlight_df = checkpoint_df.loc[checkpoint_df[highlight_col]].copy()
            if not highlight_df.empty:
                highlight_df = highlight_df.sort_values(x_col)
                ax.plot(
                    highlight_df[x_col],
                    highlight_df[y_col],
                    color=color,
                    linewidth=1.0,
                    alpha=0.85,
                    zorder=2,
                )
                ax.scatter(
                    highlight_df[x_col],
                    highlight_df[y_col],
                    s=58,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.2,
                    zorder=3,
                )

        selected_df = checkpoint_df.loc[checkpoint_df[selected_col]].copy()
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
    selection_mode: str,
    save_path: str | Path,
    annotate_selected: bool = True,
    title_fontsize: float | None = None,
) -> Path:
    spec = get_selection_mode_spec(selection_mode)
    selected_col = str(spec["selected_col"])
    highlight_col = spec.get("highlight_col", spec.get("candidate_col"))
    checkpoint_names = _ordered_checkpoint_names(df)
    palette = _build_checkpoint_palette(checkpoint_names)
    candidate_label = _selection_candidate_label(
        selection_mode=selection_mode,
        pareto_threshold_pct=_pareto_threshold_pct(df),
    )
    panel_specs = _selection_pairwise_specs(selection_mode)
    required_metric_cols = sorted(
        {metric for panel_spec in panel_specs for metric in (panel_spec["x_col"], panel_spec["y_col"])}
    )
    _require_metric_columns(
        df,
        required_metric_cols,
        context=f"Pairwise Pareto grid for '{selection_mode}'",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))

    for ax, panel_spec in zip(axes, panel_specs):
        plot_pairwise_pareto_panel(
            ax=ax,
            df=df,
            x_col=panel_spec["x_col"],
            y_col=panel_spec["y_col"],
            x_label=f"{_metric_label(panel_spec['x_col'])} ($\\downarrow$)",
            y_label=f"{_metric_label(panel_spec['y_col'])} ($\\downarrow$)",
            highlight_col=highlight_col,
            selected_col=selected_col,
            palette=palette,
            annotate_selected=annotate_selected,
            title=panel_spec["title"],
            title_fontsize=title_fontsize,
        )

    handles = _build_common_handles(
        checkpoint_names=checkpoint_names,
        palette=palette,
        candidate_label=candidate_label,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, 1.05),
    )
    fig.suptitle(
        f"Selection Mode: {spec['display_title']}",
        y=1.09,
        fontsize=(title_fontsize + 1.0) if title_fontsize is not None else None,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_fid_lpips_tradeoff(
    df: pd.DataFrame,
    selection_mode: str,
    save_path: str | Path,
    title_fontsize: float | None = None,
) -> Path:
    spec = get_selection_mode_spec(selection_mode)
    selected_col = str(spec["selected_col"])
    highlight_col = spec.get("highlight_col", spec.get("candidate_col"))
    checkpoint_names = _ordered_checkpoint_names(df)
    palette = _build_checkpoint_palette(checkpoint_names)
    candidate_label = _selection_candidate_label(
        selection_mode=selection_mode,
        pareto_threshold_pct=_pareto_threshold_pct(df),
    )
    x_col, y_col = _selection_tradeoff_axes(selection_mode)
    _require_metric_columns(
        df,
        [x_col, y_col],
        context=f"Tradeoff plot for '{selection_mode}'",
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.4))

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
            zorder=1,
        )

        if highlight_col is not None:
            highlight_df = checkpoint_df.loc[checkpoint_df[highlight_col]].copy()
            if not highlight_df.empty:
                highlight_df = highlight_df.sort_values(x_col)
                ax.plot(
                    highlight_df[x_col],
                    highlight_df[y_col],
                    color=color,
                    linewidth=1.0,
                    alpha=0.85,
                    zorder=2,
                )
                ax.scatter(
                    highlight_df[x_col],
                    highlight_df[y_col],
                    s=58,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.2,
                    zorder=3,
                )

        selected_df = checkpoint_df.loc[checkpoint_df[selected_col]].copy()
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

    ax.set_title(
        f"{_metric_label(y_col)} vs. {_metric_label(x_col)}\n{spec['display_title']}",
        fontsize=title_fontsize,
    )
    ax.set_xlabel(f"{_metric_label(x_col)} ($\\downarrow$)")
    ax.set_ylabel(f"{_metric_label(y_col)} ($\\downarrow$)")
    ax.grid(alpha=0.25)

    handles = _build_common_handles(
        checkpoint_names=checkpoint_names,
        palette=palette,
        candidate_label=candidate_label,
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
    selection_mode: str,
    save_path: str | Path,
    elev: float = 24.0,
    azim: float = -52.0,
    annotate_selected: bool = False,
    title_fontsize: float | None = None,
    title_pad: float | None = None,
) -> Path:
    spec = get_selection_mode_spec(selection_mode)
    selected_col = str(spec["selected_col"])
    highlight_col = spec.get("highlight_col", spec.get("candidate_col"))
    checkpoint_names = _ordered_checkpoint_names(df)
    palette = _build_checkpoint_palette(checkpoint_names)
    x_col, y_col, z_col = _selection_3d_metric_cols(selection_mode)
    _require_metric_columns(
        df,
        [x_col, y_col, z_col],
        context=f"3D Pareto scatter for '{selection_mode}'",
    )

    fig = plt.figure(figsize=(8.8, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    for checkpoint_name in checkpoint_names:
        checkpoint_df = df.loc[df["checkpoint_name"] == checkpoint_name].copy()
        color = palette[checkpoint_name]

        ax.scatter(
            checkpoint_df[x_col],
            checkpoint_df[y_col],
            checkpoint_df[z_col],
            s=26,
            alpha=0.28,
            color=color,
            depthshade=False,
        )

        if highlight_col is not None:
            highlight_df = checkpoint_df.loc[checkpoint_df[highlight_col]].copy()
            if not highlight_df.empty:
                ax.scatter(
                    highlight_df[x_col],
                    highlight_df[y_col],
                    highlight_df[z_col],
                    s=58,
                    facecolors="none",
                    edgecolors=[color],
                    linewidths=1.2,
                    depthshade=False,
                )

        selected_df = checkpoint_df.loc[checkpoint_df[selected_col]].copy()
        if not selected_df.empty:
            ax.scatter(
                selected_df[x_col],
                selected_df[y_col],
                selected_df[z_col],
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
                    float(row[x_col]),
                    float(row[y_col]),
                    float(row[z_col]),
                    _build_selected_label(row),
                    fontsize=8,
                )

    ax.set_xlabel(f"{_metric_label(x_col)} ($\\downarrow$)")
    ax.set_ylabel(f"{_metric_label(y_col)} ($\\downarrow$)")
    ax.set_zlabel(f"{_metric_label(z_col)} ($\\downarrow$)", labelpad=12)
    ax.set_title(
        f"3D Tradeoff Across Checkpoints\n{spec['display_title']}",
        fontsize=title_fontsize,
        pad=title_pad,
    )
    ax.view_init(elev=elev, azim=azim)

    handles = _build_common_handles(
        checkpoint_names=checkpoint_names,
        palette=palette,
        candidate_label=_selection_candidate_label(
            selection_mode=selection_mode,
            pareto_threshold_pct=_pareto_threshold_pct(df),
        ),
    )
    ax.legend(handles=handles, loc="upper left", frameon=True)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.02, top=0.94)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_selection_story(
    df: pd.DataFrame,
    selection_mode: str,
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
    spec = get_selection_mode_spec(selection_mode)
    if not bool(spec.get("story_supported", False)):
        raise ValueError(f"Selection story plot is not supported for '{selection_mode}'.")

    selected_col = str(spec["selected_col"])
    highlight_col = str(spec["candidate_col"])
    pass1_metric_cols = list(spec["pass1_metric_cols"])
    tie_break_metric = str(spec["tie_break_metric"])
    x_col = pass1_metric_cols[1]
    y_col = tie_break_metric
    z_col = pass1_metric_cols[0]

    checkpoint_names = _ordered_checkpoint_names(df)
    palette = _build_checkpoint_palette(checkpoint_names)
    candidate_label = _selection_candidate_label(
        selection_mode=selection_mode,
        pareto_threshold_pct=_pareto_threshold_pct(df),
    )

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
    ax_front = fig.add_subplot(grid[0, 1])
    ax_tie = fig.add_subplot(grid[1, 1])

    for checkpoint_name in checkpoint_names:
        checkpoint_df = df.loc[df["checkpoint_name"] == checkpoint_name].copy()
        color = palette[checkpoint_name]
        candidate_df = checkpoint_df.loc[checkpoint_df[highlight_col]].copy()
        selected_df = checkpoint_df.loc[checkpoint_df[selected_col]].copy()

        ax3d.scatter(
            checkpoint_df[x_col],
            checkpoint_df[y_col],
            checkpoint_df[z_col],
            s=20,
            alpha=0.14,
            color=color,
            depthshade=False,
        )
        if not candidate_df.empty:
            candidate_3d_df = candidate_df.sort_values([x_col, z_col, y_col])
            ax3d.plot(
                candidate_3d_df[x_col],
                candidate_3d_df[y_col],
                candidate_3d_df[z_col],
                color=color,
                linewidth=1.0,
                alpha=0.8,
            )
            ax3d.scatter(
                candidate_df[x_col],
                candidate_df[y_col],
                candidate_df[z_col],
                s=64,
                facecolors="none",
                edgecolors=[color],
                linewidths=1.3,
                depthshade=False,
            )
        if not selected_df.empty:
            row = selected_df.iloc[0]
            ax3d.scatter(
                [float(row[x_col])],
                [float(row[y_col])],
                [float(row[z_col])],
                s=135,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.7,
                depthshade=False,
            )

        ax_front.scatter(
            checkpoint_df[x_col],
            checkpoint_df[z_col],
            s=28,
            alpha=0.18,
            color=color,
            edgecolors="none",
            zorder=1,
        )
        if not candidate_df.empty:
            front_2d_df = candidate_df.sort_values(x_col)
            ax_front.plot(
                front_2d_df[x_col],
                front_2d_df[z_col],
                color=color,
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )
            ax_front.scatter(
                candidate_df[x_col],
                candidate_df[z_col],
                s=62,
                facecolors="none",
                edgecolors=color,
                linewidths=1.3,
                zorder=3,
            )
        if not selected_df.empty:
            row = selected_df.iloc[0]
            ax_front.scatter(
                [float(row[x_col])],
                [float(row[z_col])],
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
        candidate_df = checkpoint_df.loc[checkpoint_df[highlight_col]].copy()
        if candidate_df.empty:
            continue

        candidate_df = candidate_df.sort_values(
            [tie_break_metric, x_col, z_col, "noise_strength", "cfg_weight"],
            ascending=[True, True, True, True, True],
        )
        y_value = checkpoint_to_y[checkpoint_name]
        ax_tie.plot(
            candidate_df[tie_break_metric],
            [y_value] * len(candidate_df),
            color=color,
            linewidth=1.0,
            alpha=0.6,
            zorder=1,
        )
        ax_tie.scatter(
            candidate_df[tie_break_metric],
            [y_value] * len(candidate_df),
            s=40,
            color=color,
            alpha=0.7,
            zorder=2,
        )

        selected_df = candidate_df.loc[candidate_df[selected_col]].copy()
        if not selected_df.empty:
            row = selected_df.iloc[0]
            ax_tie.scatter(
                [float(row[tie_break_metric])],
                [y_value],
                s=120,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.7,
                zorder=3,
            )
            if annotate_selected:
                ax_tie.annotate(
                    f"$\\gamma={float(row['noise_strength']):g}$, $w={float(row['cfg_weight']):g}$",
                    (float(row[tie_break_metric]), y_value),
                    xytext=(7, 6),
                    textcoords="offset points",
                    fontsize=8,
                    va="bottom",
                )

    ax3d.set_xlabel(f"{_metric_label(x_col)} ($\\downarrow$)")
    ax3d.set_ylabel(f"{_metric_label(y_col)} ($\\downarrow$)")
    ax3d.set_zlabel(f"{_metric_label(z_col)} ($\\downarrow$)", labelpad=12)
    ax3d.set_title(
        "3D Tradeoff Overview",
        fontsize=panel_title_fontsize,
        pad=story_3d_title_pad,
    )
    if story_3d_title_y is not None:
        ax3d.title.set_y(story_3d_title_y)
    ax3d.view_init(elev=elev, azim=azim)

    stage_one_label = candidate_label if candidate_label is not None else "Selection candidate pool"
    ax_front.set_title(
        f"Pass 1: {stage_one_label}",
        fontsize=panel_title_fontsize,
        pad=10.0,
    )
    ax_front.set_xlabel(f"{_metric_label(x_col)} ($\\downarrow$)")
    ax_front.set_ylabel(f"{_metric_label(z_col)} ($\\downarrow$)")
    ax_front.grid(alpha=0.25)

    ax_tie.set_title(
        f"Pass 2: Lowest {_metric_label(tie_break_metric)}",
        fontsize=panel_title_fontsize,
        pad=10.0,
    )
    ax_tie.set_xlabel(f"{_metric_label(tie_break_metric)} ($\\downarrow$)")
    ax_tie.set_ylabel("Checkpoint")
    ax_tie.set_yticks(list(checkpoint_to_y.values()))
    ax_tie.set_yticklabels(checkpoint_names)
    ax_tie.grid(axis="x", alpha=0.25)
    ax_tie.set_axisbelow(True)

    handles = _build_common_handles(
        checkpoint_names=checkpoint_names,
        palette=palette,
        candidate_label=candidate_label,
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
        f"Selection Mode: {spec['display_title']}",
        y=0.995,
        fontsize=suptitle_fontsize,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.08, top=0.89)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(save_path)
    plt.close(fig)
    return save_path


def save_all_pareto_plots(
    csv_path: str | Path,
    out_dir: str | Path | None = None,
    selection_modes: list[str] | None = None,
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

    available_modes = discover_selection_modes(df)
    if selection_modes is None:
        selection_modes = available_modes
    else:
        selection_modes = [mode.lower() for mode in selection_modes]

    saved_paths: list[Path] = []
    for selection_mode in selection_modes:
        if selection_mode not in available_modes:
            raise ValueError(
                f"Selection mode '{selection_mode}' is not present in {csv_path}. "
                f"Available modes: {', '.join(available_modes)}"
            )

        mode_dir = out_root / selection_mode
        saved_paths.extend(
            [
                plot_pairwise_pareto_grid(
                    df=df,
                    selection_mode=selection_mode,
                    save_path=mode_dir / "pareto_pairwise_grid.pdf",
                    annotate_selected=annotate_selected_pairwise,
                    title_fontsize=pairwise_title_fontsize,
                ),
                plot_fid_lpips_tradeoff(
                    df=df,
                    selection_mode=selection_mode,
                    save_path=mode_dir / "pareto_tradeoff.pdf",
                    title_fontsize=pairwise_title_fontsize,
                ),
                plot_3d_pareto_scatter(
                    df=df,
                    selection_mode=selection_mode,
                    save_path=mode_dir / "pareto_3d_scatter.pdf",
                    annotate_selected=annotate_selected_3d,
                    title_fontsize=scatter_3d_title_fontsize,
                    title_pad=scatter_3d_title_pad,
                ),
            ]
        )

        if bool(get_selection_mode_spec(selection_mode).get("story_supported", False)):
            saved_paths.append(
                plot_selection_story(
                    df=df,
                    selection_mode=selection_mode,
                    save_path=mode_dir / "selection_story.pdf",
                    annotate_selected=annotate_selected_pairwise,
                    panel_title_fontsize=story_panel_title_fontsize,
                    suptitle_fontsize=story_suptitle_fontsize,
                    legend_fontsize=story_legend_fontsize,
                    story_3d_title_pad=story_3d_title_pad,
                    story_3d_title_y=story_3d_title_y,
                )
            )

    return saved_paths


def plot_all_pareto_plots(
    merged_csv_path: Path | str,
    selection_modes: list[str] | None = None,
):
    out_dir = Path(merged_csv_path).parent / "plots"

    saved_paths = save_all_pareto_plots(
        csv_path=merged_csv_path,
        out_dir=out_dir,
        selection_modes=selection_modes,
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


def plot_pareto_sweep(
    out_dir: Path | str,
    selection_modes: list[str] | None = None,
    pct_thresholds: list[float] = [0.0, 5.0, 10.0, 25.0],
):
    for pareto_threshold_pct in pct_thresholds:

        if pareto_threshold_pct == 0.0:
            pct_out_dir = Path(out_dir) / "pareto-front"
        else:
            pct_out_dir = Path(out_dir) / f"{int(pareto_threshold_pct)}-pct-pareto-front"

        # -- Verify pct eval results exist
        if not pct_out_dir.exists():
            continue

        merged_csv_path = pct_out_dir / "checkpoint_metrics_merged.csv"

        plot_all_pareto_plots(merged_csv_path, selection_modes)
