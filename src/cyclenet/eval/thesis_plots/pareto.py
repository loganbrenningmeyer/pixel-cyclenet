from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cyclenet.eval.analyze_checkpoint_metrics import (
    get_selection_mode_spec as get_analysis_selection_mode_spec,
)
from cyclenet.eval.plotting.pareto import (
    METRIC_LABELS,
    get_selection_mode_spec,
    load_checkpoint_metrics,
)
from cyclenet.eval.plotting.set_style import (
    CHECKPOINT_MARKERS,
    MODEL_COLORS,
    MODEL_NAMES,
    apply_style,
)

apply_style()

UNFILLED_MARKERS = {"x", "+", "1", "2", "3", "4"}
ANNOTATION_BBOX = {
    "boxstyle": "round,pad=0.25",
    "facecolor": "white",
    "edgecolor": "#9ca3af",
    "linewidth": 0.8,
    "alpha": 0.95,
}


def _display_model_name(model_name: str) -> str:
    display_name = MODEL_NAMES.get(model_name, "")
    display_name = display_name if display_name else model_name.replace("_", " ")
    if display_name == "RGB + SPADE (BN Only)":
        return "RGB + SPADE\n(BN Only)"
    return display_name


def _model_color(model_name: str) -> str:
    color = MODEL_COLORS.get(model_name, "")
    return color if color else "#6f6f6f"


def _checkpoint_marker(step: int | float | str) -> str:
    try:
        step_key = int(step)
    except (TypeError, ValueError):
        return "o"
    return CHECKPOINT_MARKERS.get(step_key, "o")


def _checkpoint_marker_handles() -> list[mlines.Line2D]:
    handles: list[mlines.Line2D] = []
    for step, marker in sorted(CHECKPOINT_MARKERS.items()):
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="#111827",
                marker=marker,
                linestyle="None",
                markerfacecolor="#111827" if marker not in UNFILLED_MARKERS else "none",
                markeredgecolor="#111827",
                markersize=6.0,
                label=f"{int(step / 1000)}k checkpoint",
            )
        )
    return handles


def _selected_handle() -> mlines.Line2D:
    return mlines.Line2D(
        [],
        [],
        color="#c62828",
        marker="o",
        linestyle="None",
        markerfacecolor="#c62828",
        markeredgecolor="black",
        markersize=8,
        label="Selected configuration",
    )


def _normalized_handles() -> list[mlines.Line2D]:
    return [
        mlines.Line2D(
            [],
            [],
            color="#6f6f6f",
            marker="o",
            markerfacecolor="white",
            linestyle="-",
            linewidth=1.2,
            markersize=5.5,
            label="Normalized Pareto front",
        ),
        mlines.Line2D(
            [],
            [],
            color="#6f6f6f",
            marker="o",
            linestyle="--",
            linewidth=1.2,
            markeredgecolor="black",
            markersize=5.5,
            label="Start/end chord",
        ),
        mlines.Line2D(
            [],
            [],
            color="#c62828",
            linestyle=":",
            linewidth=1.5,
            label="Perpendicular to chord",
        ),
        _selected_handle(),
        mlines.Line2D(
            [],
            [],
            color="#c62828",
            marker="x",
            linestyle="None",
            markersize=6.5,
            label="Projection on chord",
        ),
    ]


def _normalized_knee_score(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    safe_spans = np.where(spans > 0.0, spans, 1.0)
    normalized = (points - mins) / safe_spans

    ideal_distance = np.sqrt((normalized**2).sum(axis=1))

    if len(normalized) < 3:
        return np.zeros(len(normalized), dtype=float), ideal_distance

    start = normalized[0]
    end = normalized[-1]
    line = end - start
    line_norm = float(np.linalg.norm(line))

    if line_norm <= 1e-12:
        return np.zeros(len(normalized), dtype=float), ideal_distance

    rel = normalized - start
    proj_scale = (rel @ line) / float(line @ line)
    proj = np.outer(proj_scale, line)
    knee_distance = np.linalg.norm(rel - proj, axis=1)
    return knee_distance, ideal_distance


def _project_point_to_line(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    line = end - start
    denom = float(line @ line)
    if denom <= 1e-12:
        return start.copy()
    scale = float(((point - start) @ line) / denom)
    return start + (scale * line)


def _build_knee_geometry(
    df: pd.DataFrame,
    pareto_col: str,
    selected_col: str,
    metric_cols: list[str],
) -> dict[str, object]:
    front_df = df.loc[df[pareto_col]].copy()
    if front_df.empty:
        raise ValueError(f"No exact Pareto-front rows found for '{pareto_col}'.")

    front_df = front_df.sort_values(
        metric_cols + ["noise_strength", "cfg_weight"],
        ascending=[True] * (len(metric_cols) + 2),
    ).copy()
    points = front_df[metric_cols].to_numpy(dtype=float)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    safe_spans = np.where(spans > 0.0, spans, 1.0)
    normalized = (points - mins) / safe_spans

    knee_distance, ideal_distance = _normalized_knee_score(points)
    front_df["knee_distance"] = knee_distance
    front_df["ideal_distance"] = ideal_distance

    if selected_col in front_df.columns and bool(front_df[selected_col].any()):
        selected_row = front_df.loc[front_df[selected_col]].iloc[0]
    else:
        selected_row = front_df.sort_values(
            [
                "knee_distance",
                "ideal_distance",
                metric_cols[0],
                metric_cols[1],
                "noise_strength",
                "cfg_weight",
            ],
            ascending=[False, True, True, True, True, True],
        ).iloc[0]

    selected_position = int(front_df.index.get_loc(selected_row.name))
    start = normalized[0]
    end = normalized[-1]
    selected_point = normalized[selected_position]
    projected_point = _project_point_to_line(selected_point, start, end)

    return {
        "front_df": front_df,
        "normalized": normalized,
        "start": start,
        "end": end,
        "selected_row": selected_row,
        "selected_point": selected_point,
        "projected_point": projected_point,
    }


def _format_checkpoint_annotation(step: int | float | str) -> str:
    return f"$c = {int(float(step) / 1000):d}\\mathrm{{k}}$"


def _format_selected_annotation(row: pd.Series, include_checkpoint: bool = False) -> str:
    lines = [
        f"$s={float(row['noise_strength']):g}$",
        f"$w={float(row['cfg_weight']):g}$",
    ]
    if include_checkpoint:
        lines.insert(0, _format_checkpoint_annotation(row["step"]))
    return "\n".join(lines)


def _draw_selected_annotation(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    xytext: tuple[float, float],
    annotation_fontsize: float,
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=annotation_fontsize,
        bbox=ANNOTATION_BBOX,
        arrowprops={
            "arrowstyle": "-",
            "color": "#6b7280",
            "linewidth": 0.8,
            "shrinkA": 4,
            "shrinkB": 4,
        },
    )


def _load_model_metric_dfs(
    model_metrics: dict[str, str | Path],
    required_cols: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    if not model_metrics:
        raise ValueError("model_metrics cannot be empty.")

    model_dfs: list[tuple[str, pd.DataFrame]] = []
    for model_name, csv_path in model_metrics.items():
        df = load_checkpoint_metrics(csv_path).copy()
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Model '{model_name}' is missing required columns {missing_cols} "
                f"in merged metrics CSV: {csv_path}"
            )
        model_dfs.append((model_name, df))
    return model_dfs


def _compute_raw_axis_limits(
    model_dfs: list[tuple[str, pd.DataFrame]],
    x_col: str,
    y_col: str,
    pareto_col: str,
    selected_col: str,
    clip_quantile: float | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_x = np.concatenate([df[x_col].to_numpy(dtype=float) for _, df in model_dfs])
    all_y = np.concatenate([df[y_col].to_numpy(dtype=float) for _, df in model_dfs])

    x_min = float(np.min(all_x))
    y_min = float(np.min(all_y))

    if clip_quantile is None:
        x_max = float(np.max(all_x))
        y_max = float(np.max(all_y))
    else:
        if not 0.0 < clip_quantile <= 1.0:
            raise ValueError(f"clip_quantile must be in (0, 1], got {clip_quantile}.")
        x_max = float(np.quantile(all_x, clip_quantile))
        y_max = float(np.quantile(all_y, clip_quantile))

        for _, df in model_dfs:
            front_df = df.loc[df[pareto_col]].copy()
            if not front_df.empty:
                x_max = max(x_max, float(front_df[x_col].max()))
                y_max = max(y_max, float(front_df[y_col].max()))

            if selected_col in df.columns:
                selected_df = df.loc[df[selected_col]].copy()
                if not selected_df.empty:
                    x_max = max(x_max, float(selected_df[x_col].max()))
                    y_max = max(y_max, float(selected_df[y_col].max()))

    x_pad = 0.04 * (x_max - x_min) if x_max > x_min else 0.05
    y_pad = 0.04 * (y_max - y_min) if y_max > y_min else 0.05
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def _draw_raw_pareto_panel(
    ax: plt.Axes,
    model_name: str,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    pareto_col: str,
    selected_col: str,
    annotate_selected: bool,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title_fontsize: float,
    tick_label_fontsize: float,
    annotation_fontsize: float,
    show_title: bool = True,
) -> None:
    model_color = _model_color(model_name)
    for step, step_df in df.groupby("step", sort=True):
        marker = _checkpoint_marker(step)
        ax.scatter(
            step_df[x_col],
            step_df[y_col],
            s=28,
            alpha=0.38,
            color=model_color,
            marker=marker,
            edgecolors="none" if marker not in UNFILLED_MARKERS else model_color,
            linewidths=0.8,
            zorder=1,
        )

    front_df = df.loc[df[pareto_col]].sort_values(x_col).copy()
    if not front_df.empty:
        ax.plot(
            front_df[x_col],
            front_df[y_col],
            color=model_color,
            linewidth=1.2,
            zorder=2,
        )
        for step, step_front_df in front_df.groupby("step", sort=True):
            marker = _checkpoint_marker(step)
            ax.scatter(
                step_front_df[x_col],
                step_front_df[y_col],
                s=44,
                marker=marker,
                facecolors="white" if marker not in UNFILLED_MARKERS else "none",
                edgecolors=model_color,
                linewidths=1.1,
                zorder=3,
            )

    if selected_col in df.columns:
        selected_df = df.loc[df[selected_col]].copy()
        if not selected_df.empty:
            row = selected_df.iloc[0]
            marker = _checkpoint_marker(row["step"])
            ax.scatter(
                [float(row[x_col])],
                [float(row[y_col])],
                s=44,
                marker=marker,
                color="#c62828",
                facecolors="#c62828" if marker not in UNFILLED_MARKERS else "none",
                edgecolors="black",
                linewidths=1.0,
                zorder=5,
            )
            if annotate_selected:
                _draw_selected_annotation(
                    ax=ax,
                    x=float(row[x_col]),
                    y=float(row[y_col]),
                    text=_format_selected_annotation(row),
                    xytext=(12, 8),
                    annotation_fontsize=annotation_fontsize,
                )

    if show_title:
        ax.set_title(_display_model_name(model_name), fontsize=title_fontsize)
    ax.grid(alpha=0.25)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)


def _draw_normalized_knee_panel(
    ax: plt.Axes,
    model_name: str,
    geometry: dict[str, object],
    annotate_selected: bool,
    annotate_chord: bool,
    include_checkpoint_in_annotation: bool,
    title_fontsize: float,
    tick_label_fontsize: float,
    annotation_fontsize: float,
    chord_label_fontsize: float,
    show_title: bool = True,
) -> None:
    model_color = _model_color(model_name)
    normalized = np.asarray(geometry["normalized"], dtype=float)
    start = np.asarray(geometry["start"], dtype=float)
    end = np.asarray(geometry["end"], dtype=float)
    selected_point = np.asarray(geometry["selected_point"], dtype=float)
    projected_point = np.asarray(geometry["projected_point"], dtype=float)
    selected_row = geometry["selected_row"]
    front_df = geometry["front_df"]

    ax.plot(
        normalized[:, 0],
        normalized[:, 1],
        color=model_color,
        linewidth=1.2,
        zorder=2,
    )
    for point_idx, (_, row) in enumerate(front_df.iterrows()):
        marker = _checkpoint_marker(row["step"])
        ax.scatter(
            [normalized[point_idx, 0]],
            [normalized[point_idx, 1]],
            s=44,
            marker=marker,
            facecolors="white" if marker not in UNFILLED_MARKERS else "none",
            edgecolors=model_color,
            linewidths=1.1,
            zorder=3,
        )
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=model_color,
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.plot(
        [selected_point[0], projected_point[0]],
        [selected_point[1], projected_point[1]],
        color="#c62828",
        linestyle=":",
        linewidth=1.5,
        zorder=4,
    )
    ax.scatter(
        [start[0], end[0]],
        [start[1], end[1]],
        s=58,
        color=model_color,
        edgecolors="black",
        linewidths=0.7,
        zorder=5,
    )
    ax.scatter(
        [projected_point[0]],
        [projected_point[1]],
        s=54,
        marker="x",
        color="#c62828",
        linewidths=1.3,
        zorder=5,
    )
    selected_marker = _checkpoint_marker(selected_row["step"])
    ax.scatter(
        [selected_point[0]],
        [selected_point[1]],
        s=44,
        marker=selected_marker,
        color="#c62828",
        facecolors="#c62828" if selected_marker not in UNFILLED_MARKERS else "none",
        edgecolors="black",
        linewidths=1.0,
        zorder=6,
    )

    if annotate_chord:
        ax.annotate(
            "start",
            (start[0], start[1]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=chord_label_fontsize,
        )
        ax.annotate(
            "end",
            (end[0], end[1]),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=chord_label_fontsize,
        )

    if annotate_selected:
        _draw_selected_annotation(
            ax=ax,
            x=float(selected_point[0]),
            y=float(selected_point[1]),
            text=_format_selected_annotation(
                selected_row,
                include_checkpoint=include_checkpoint_in_annotation,
            ),
            xytext=(32, 6),
            annotation_fontsize=annotation_fontsize,
        )

    if show_title:
        ax.set_title(_display_model_name(model_name), fontsize=title_fontsize)
    ax.grid(alpha=0.25)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=tick_label_fontsize)


def plot_deeplab_boundary_pareto_row(
    model_metrics: dict[str, str | Path],
    save_path: str | Path,
    selection_mode: str = "deeplab_boundary_pareto_then_knee",
    title: str | None = None,
    annotate_selected: bool = False,
    clip_quantile: float | None = None,
    model_title_fontsize: float = 16.0,
    axis_label_fontsize: float = 12.0,
    tick_label_fontsize: float = 11.0,
    legend_fontsize: float = 11.0,
    selected_annotation_fontsize: float = 10.0,
) -> Path:
    spec = get_selection_mode_spec(selection_mode)
    selected_col = str(spec["selected_col"])
    pareto_col = str(spec.get("pareto_col", "is_pareto_deeplab_boundary"))
    x_col = "boundary_edge_inverse_ratio_mean"
    y_col = "deeplab_fd"

    model_dfs = _load_model_metric_dfs(model_metrics, required_cols=[x_col, y_col])
    xlim, ylim = _compute_raw_axis_limits(
        model_dfs=model_dfs,
        x_col=x_col,
        y_col=y_col,
        pareto_col=pareto_col,
        selected_col=selected_col,
        clip_quantile=clip_quantile,
    )

    n_models = len(model_dfs)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(3.5 * n_models, 3.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_row = list(axes[0])

    for idx, ((model_name, df), ax) in enumerate(zip(model_dfs, axes_row, strict=True)):
        _draw_raw_pareto_panel(
            ax=ax,
            model_name=model_name,
            df=df,
            x_col=x_col,
            y_col=y_col,
            pareto_col=pareto_col,
            selected_col=selected_col,
            annotate_selected=annotate_selected,
            xlim=xlim,
            ylim=ylim,
            title_fontsize=model_title_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            annotation_fontsize=selected_annotation_fontsize,
        )
        if idx == 0:
            ax.set_ylabel(f"{METRIC_LABELS[y_col]} ($\\downarrow$)", fontsize=axis_label_fontsize)
        ax.set_xlabel(f"{METRIC_LABELS[x_col]} ($\\downarrow$)", fontsize=axis_label_fontsize)

    handles = [
        mlines.Line2D(
            [],
            [],
            color="#6f6f6f",
            marker="o",
            linestyle="None",
            markersize=5.5,
            label="All candidates",
        ),
        mlines.Line2D(
            [],
            [],
            color="#6f6f6f",
            marker="o",
            markerfacecolor="white",
            linestyle="-",
            linewidth=1.2,
            markersize=5.5,
            label="Exact Pareto front",
        ),
    ]
    if any(selected_col in df.columns and bool(df[selected_col].any()) for _, df in model_dfs):
        handles.append(_selected_handle())
    handles.extend(_checkpoint_marker_handles())

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 4),
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=legend_fontsize,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_deeplab_boundary_pareto_knee_normalized_row(
    model_metrics: dict[str, str | Path],
    save_path: str | Path,
    selection_mode: str = "deeplab_boundary_pareto_then_knee",
    title: str | None = None,
    annotate_selected: bool = False,
    annotate_chord: bool = False,
    annotate_checkpoint: bool = False,
    model_title_fontsize: float = 12.5,
    axis_label_fontsize: float = 12.0,
    tick_label_fontsize: float = 11.0,
    legend_fontsize: float = 11.0,
    selected_annotation_fontsize: float = 10.0,
    chord_label_fontsize: float = 7.0,
) -> Path:
    spec = get_selection_mode_spec(selection_mode)
    analysis_spec = get_analysis_selection_mode_spec(selection_mode)
    if str(analysis_spec.get("selection_kind", "")) != "pareto_knee":
        raise ValueError(
            f"Normalized knee visualization requires a pareto_knee selection mode, got "
            f"'{selection_mode}'."
        )

    selected_col = str(spec["selected_col"])
    pareto_col = str(analysis_spec.get("pareto_col", "is_pareto_deeplab_boundary"))
    metric_cols = list(analysis_spec.get("knee_metric_cols", []))
    if len(metric_cols) != 2:
        raise ValueError(
            f"Normalized knee visualization expects exactly 2 knee metrics, got {metric_cols}."
        )

    if not model_metrics:
        raise ValueError("model_metrics cannot be empty.")

    x_col = metric_cols[0]
    y_col = metric_cols[1]
    x_label = f"Normalized {METRIC_LABELS.get(x_col, x_col)}"
    y_label = f"Normalized {METRIC_LABELS.get(y_col, y_col)}"

    model_dfs = _load_model_metric_dfs(model_metrics, required_cols=metric_cols)
    model_geometries: list[tuple[str, dict[str, object]]] = [
        (
            model_name,
            _build_knee_geometry(
                df=df,
                pareto_col=pareto_col,
                selected_col=selected_col,
                metric_cols=metric_cols,
            ),
        )
        for model_name, df in model_dfs
    ]

    n_models = len(model_geometries)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(3.7 * n_models, 3.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_row = list(axes[0])

    for idx, ((model_name, geometry), ax) in enumerate(zip(model_geometries, axes_row, strict=True)):
        _draw_normalized_knee_panel(
            ax=ax,
            model_name=model_name,
            geometry=geometry,
            annotate_selected=annotate_selected,
            annotate_chord=annotate_chord,
            include_checkpoint_in_annotation=annotate_checkpoint,
            title_fontsize=model_title_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            annotation_fontsize=selected_annotation_fontsize,
            chord_label_fontsize=chord_label_fontsize,
        )
        if idx == 0:
            ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
        ax.set_xlabel(x_label, fontsize=axis_label_fontsize)

    handles = _normalized_handles()
    handles.extend(_checkpoint_marker_handles())

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=legend_fontsize,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_deeplab_boundary_pareto_two_row(
    model_metrics: dict[str, str | Path],
    save_path: str | Path,
    selection_mode: str = "deeplab_boundary_pareto_then_knee",
    title: str | None = None,
    annotate_normalized: bool = False,
    annotate_chord: bool = False,
    annotate_checkpoint: bool = False,
    clip_quantile: float | None = None,
    raw_model_title_fontsize: float = 16.0,
    normalized_model_title_fontsize: float = 12.5,
    row_header_fontsize: float = 16.0,
    axis_label_fontsize: float = 12.0,
    tick_label_fontsize: float = 11.0,
    legend_fontsize: float = 11.0,
    selected_annotation_fontsize: float = 10.0,
    chord_label_fontsize: float = 7.0,
) -> Path:
    spec = get_selection_mode_spec(selection_mode)
    analysis_spec = get_analysis_selection_mode_spec(selection_mode)
    if str(analysis_spec.get("selection_kind", "")) != "pareto_knee":
        raise ValueError(
            f"Combined Pareto figure requires a pareto_knee selection mode, got '{selection_mode}'."
        )

    selected_col = str(spec["selected_col"])
    pareto_col = str(analysis_spec.get("pareto_col", "is_pareto_deeplab_boundary"))
    x_col = "boundary_edge_inverse_ratio_mean"
    y_col = "deeplab_fd"
    metric_cols = list(analysis_spec.get("knee_metric_cols", []))
    if len(metric_cols) != 2:
        raise ValueError(f"Combined Pareto figure expects exactly 2 knee metrics, got {metric_cols}.")

    model_dfs = _load_model_metric_dfs(model_metrics, required_cols=[x_col, y_col] + metric_cols)
    xlim, ylim = _compute_raw_axis_limits(
        model_dfs=model_dfs,
        x_col=x_col,
        y_col=y_col,
        pareto_col=pareto_col,
        selected_col=selected_col,
        clip_quantile=clip_quantile,
    )
    model_geometries: list[tuple[str, pd.DataFrame, dict[str, object]]] = []
    for model_name, df in model_dfs:
        model_geometries.append(
            (
                model_name,
                df,
                _build_knee_geometry(
                    df=df,
                    pareto_col=pareto_col,
                    selected_col=selected_col,
                    metric_cols=metric_cols,
                ),
            )
        )

    n_models = len(model_geometries)
    row_label_x = 0.08
    content_center_x = 0.5 * (row_label_x + 1.0)
    fig, axes = plt.subplots(
        2,
        n_models,
        figsize=(3.65 * n_models, 7.1),
        squeeze=False,
    )

    for idx, (model_name, df, geometry) in enumerate(model_geometries):
        top_ax = axes[0, idx]
        bottom_ax = axes[1, idx]
        _draw_raw_pareto_panel(
            ax=top_ax,
            model_name=model_name,
            df=df,
            x_col=x_col,
            y_col=y_col,
            pareto_col=pareto_col,
            selected_col=selected_col,
            annotate_selected=False,
            xlim=xlim,
            ylim=ylim,
            title_fontsize=raw_model_title_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            annotation_fontsize=selected_annotation_fontsize,
            show_title=True,
        )
        _draw_normalized_knee_panel(
            ax=bottom_ax,
            model_name=model_name,
            geometry=geometry,
            annotate_selected=annotate_normalized,
            annotate_chord=annotate_chord,
            include_checkpoint_in_annotation=annotate_checkpoint,
            title_fontsize=normalized_model_title_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            annotation_fontsize=selected_annotation_fontsize,
            chord_label_fontsize=chord_label_fontsize,
            show_title=False,
        )

        if idx == 0:
            top_ax.set_ylabel(f"{METRIC_LABELS[y_col]} ($\\downarrow$)", fontsize=axis_label_fontsize)
            bottom_ax.set_ylabel(
                f"Normalized {METRIC_LABELS.get(metric_cols[1], metric_cols[1])}",
                fontsize=axis_label_fontsize,
            )
        top_ax.set_xlabel(f"{METRIC_LABELS[x_col]} ($\\downarrow$)", fontsize=axis_label_fontsize)
        bottom_ax.set_xlabel(
            f"Normalized {METRIC_LABELS.get(metric_cols[0], metric_cols[0])}",
            fontsize=axis_label_fontsize,
        )

    handles = [
        mlines.Line2D(
            [],
            [],
            color="#6f6f6f",
            marker="o",
            linestyle="None",
            markersize=5.5,
            label="All candidates",
        ),
        mlines.Line2D(
            [],
            [],
            color="#6f6f6f",
            marker="o",
            markerfacecolor="white",
            linestyle="-",
            linewidth=1.2,
            markersize=5.5,
            label="Pareto front",
        ),
        _selected_handle(),
    ]
    handles.extend(_checkpoint_marker_handles())

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=min(len(handles), 4),
        frameon=True,
        bbox_to_anchor=(content_center_x, 1.08),
        bbox_transform=fig.transFigure,
        fontsize=legend_fontsize,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.12, 0.0, 1.0, 0.96))

    top_bbox = axes[0, 0].get_position()
    bottom_bbox = axes[1, 0].get_position()
    fig.text(
        row_label_x,
        0.5 * (top_bbox.y0 + top_bbox.y1),
        "Pareto\nTradeoff",
        ha="center",
        va="center",
        fontsize=row_header_fontsize,
        fontweight="semibold",
    )
    fig.text(
        row_label_x,
        0.5 * (bottom_bbox.y0 + bottom_bbox.y1),
        "Knee\nSelection",
        ha="center",
        va="center",
        fontsize=row_header_fontsize,
        fontweight="semibold",
    )

    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def main() -> None:
    # Mapping from thesis display / model names to merged checkpoint-metrics CSVs.
    model_metrics = {
        "oem_only": "/develop/code/eval/checkpoints/oem_only/pareto-front/checkpoint_metrics_merged.csv",
        "oem_only_rgb_only_spade": "/develop/code/eval/checkpoints/oem_only_rgb_only_spade/pareto-front/checkpoint_metrics_merged.csv",
        "oem_only_rgb_only_spade_mid_skips": "/develop/code/eval/checkpoints/oem_only_rgb_only_spade_mid_skips/pareto-front/checkpoint_metrics_merged.csv",
        "oem_only_seg_only": "/develop/code/eval/checkpoints/oem_only_seg_only/pareto-front/checkpoint_metrics_merged.csv",
        "oem_only_seg_only_spade": "/develop/code/eval/checkpoints/oem_only_seg_only_spade/pareto-front/checkpoint_metrics_merged.csv",
    }
    # Output path for the compact row figure.
    save_path = Path("/develop/code/eval/thesis/pareto/deeplab_boundary_pareto_row.pdf")
    # Output path for the normalized knee-geometry figure.
    normalized_save_path = Path(
        "/develop/code/eval/thesis/pareto/deeplab_boundary_pareto_knee_normalized_row.pdf"
    )
    # Output path for the combined 2-row raw-plus-normalized selection figure.
    two_row_save_path = Path(
        "/develop/code/eval/thesis/pareto/deeplab_boundary_pareto_two_row.pdf"
    )
    # Selection mode whose selected-point column should be highlighted if present.
    selection_mode = "deeplab_boundary_pareto_then_knee"
    # Optional figure title override. Use `None` to fall back to the automatic
    # selection-mode-based title.
    title = "Pareto Model Selection"
    # Whether to annotate each selected point with gamma / CFG text.
    annotate_pareto = False
    annotate_normalized = True
    # Whether selected-point annotations should also include checkpoint text
    # like `$c = 10\\mathrm{k}$`.
    annotate_checkpoint = True
    annotate_chord = False
    # Optional display-only quantile clip for shared axis limits. Use `None`
    # to show the full range, or e.g. `0.99` to suppress extreme outliers.
    clip_quantile = 0.99
    # Font size for model-name subplot titles in the raw Pareto row.
    raw_model_title_fontsize = 18.0
    # Font size for model-name subplot titles in the normalized knee row.
    normalized_model_title_fontsize = 12.5
    # Font size for x/y-axis labels across Pareto figures.
    axis_label_fontsize = 16.0
    # Font size for numeric x/y tick labels across Pareto figures.
    tick_label_fontsize = 12.0
    # Font size for legend labels across Pareto figures.
    legend_fontsize = 16.0
    # Font size for selected-point annotation text.
    selected_annotation_fontsize = 12.0
    # Font size for optional normalized-chord start/end labels.
    chord_label_fontsize = 7.0
    # Font size for the two-row figure's left-side row headers.
    row_header_fontsize = 20.0

    saved_path = plot_deeplab_boundary_pareto_row(
        model_metrics=model_metrics,
        save_path=save_path,
        selection_mode=selection_mode,
        title=title,
        annotate_selected=annotate_pareto,
        clip_quantile=clip_quantile,
        model_title_fontsize=raw_model_title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_fontsize=legend_fontsize,
        selected_annotation_fontsize=selected_annotation_fontsize,
    )
    normalized_saved_path = plot_deeplab_boundary_pareto_knee_normalized_row(
        model_metrics=model_metrics,
        save_path=normalized_save_path,
        selection_mode=selection_mode,
        title=title,
        annotate_selected=annotate_normalized,
        annotate_chord=annotate_chord,
        annotate_checkpoint=annotate_checkpoint,
        model_title_fontsize=normalized_model_title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_fontsize=legend_fontsize,
        selected_annotation_fontsize=selected_annotation_fontsize,
        chord_label_fontsize=chord_label_fontsize,
    )
    two_row_saved_path = plot_deeplab_boundary_pareto_two_row(
        model_metrics=model_metrics,
        save_path=two_row_save_path,
        selection_mode=selection_mode,
        title=title,
        annotate_normalized=annotate_normalized,
        annotate_chord=annotate_chord,
        annotate_checkpoint=annotate_checkpoint,
        clip_quantile=clip_quantile,
        raw_model_title_fontsize=raw_model_title_fontsize,
        normalized_model_title_fontsize=normalized_model_title_fontsize,
        row_header_fontsize=row_header_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_fontsize=legend_fontsize,
        selected_annotation_fontsize=selected_annotation_fontsize,
        chord_label_fontsize=chord_label_fontsize,
    )
    print(f"Saved Pareto row plot to {saved_path}")
    print(f"Saved normalized knee plot to {normalized_saved_path}")
    print(f"Saved combined two-row Pareto plot to {two_row_saved_path}")


if __name__ == "__main__":
    main()
