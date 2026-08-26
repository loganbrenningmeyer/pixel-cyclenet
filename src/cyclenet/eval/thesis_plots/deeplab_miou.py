from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cyclenet.eval.plotting.set_style import MODEL_COLORS, MODEL_NAMES, apply_style

apply_style()


ANNOTATION_BBOX = {
    "boxstyle": "round,pad=0.22",
    "facecolor": "white",
    "edgecolor": "#9ca3af",
    "linewidth": 0.8,
    "alpha": 0.95,
}

X_METRIC_LABELS = {
    "deeplab_fd": r"DeepLab-FD $\downarrow$",
    "fid": r"FID $\downarrow$",
    "ber": r"BER $\downarrow$",
}


def _display_model_name(model_name: str) -> str:
    if model_name == "sim":
        return "Sim"
    display_name = MODEL_NAMES.get(model_name, "")
    display_name = display_name if display_name else model_name.replace("_", " ")
    if display_name == "RGB + SPADE (BN Only)":
        return "RGB + SPADE\n(BN Only)"
    return display_name


def _model_color(model_name: str) -> str:
    if model_name == "sim":
        return "#6b7280"
    color = MODEL_COLORS.get(model_name, "")
    return color if color else "#6f6f6f"


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "nan":
        return None
    return float(text)


def _spearman_rho(df: pd.DataFrame, x_metric: str) -> float:
    return float(df[x_metric].corr(df["miou_mean"], method="spearman"))


def _legend_model_order(df: pd.DataFrame) -> list[str]:
    present_models = set(df["model_name"].astype(str).tolist())
    ordered_models = [model_name for model_name in MODEL_NAMES.keys() if model_name in present_models]
    if "sim" in present_models:
        ordered_models.append("sim")
    return ordered_models


def _build_legend_handles(
    plot_df: pd.DataFrame,
    show_trend_line: bool,
    legend_use_model_names: bool,
) -> list[mlines.Line2D]:
    legend_handles: list[mlines.Line2D] = []
    if show_trend_line:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="#374151",
                linestyle="--",
                linewidth=1.3,
                label="Linear fit",
            )
        )

    if legend_use_model_names:
        row_by_model_name = {
            str(row.model_name): row for row in plot_df.itertuples(index=False)
        }
        for model_name in _legend_model_order(plot_df):
            row = row_by_model_name[model_name]
            is_sim = model_name == "sim"
            marker = "D" if is_sim else "o"
            markersize = 8.0 if is_sim else 7.0
            color = _model_color(model_name)
            legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    marker=marker,
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=markersize,
                    label=str(row.display_name),
                )
            )
    else:
        legend_handles.extend(
            [
                mlines.Line2D(
                    [],
                    [],
                    color="#111827",
                    marker="o",
                    linestyle="None",
                    markerfacecolor="#111827",
                    markeredgecolor="black",
                    markersize=7.0,
                    label=r"Translated model ($\pm 1$ std mIoU)",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color="#6b7280",
                    marker="D",
                    linestyle="None",
                    markerfacecolor="#6b7280",
                    markeredgecolor="black",
                    markersize=8.0,
                    label=r"Sim baseline ($\pm 1$ std mIoU)",
                ),
            ]
        )
    return legend_handles


def _plot_metric_vs_miou_on_axes(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    x_metric: str,
    show_trend_line: bool,
    show_spearman_rho: bool,
    annotate_model_names: bool,
    annotation_fontsize: float,
    axis_label_fontsize: float,
    tick_label_fontsize: float,
    legend_fontsize: float,
    show_ylabel: bool = True,
) -> None:
    ax.grid(True, axis="both", alpha=0.22)

    for row in plot_df.itertuples(index=False):
        x = float(getattr(row, x_metric))
        y = float(row.miou_mean)
        yerr = float(row.miou_std)
        color = _model_color(str(row.model_name))
        is_sim = str(row.model_name) == "sim"
        marker = "D" if is_sim else "o"
        markersize = 8.0 if is_sim else 7.0

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3.2,
            capthick=1.2,
            markersize=markersize,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.8,
            linestyle="None",
            zorder=3,
        )

    translated_df = plot_df.loc[plot_df["model_name"] != "sim"].copy()

    if show_trend_line and len(translated_df) >= 2:
        x_vals = translated_df[x_metric].to_numpy(dtype=float)
        y_vals = translated_df["miou_mean"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, deg=1)
        x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 200)
        y_line = (slope * x_line) + intercept
        ax.plot(
            x_line,
            y_line,
            color="#374151",
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            zorder=2,
        )

    x_span = float(plot_df[x_metric].max() - plot_df[x_metric].min())
    y_span = float(plot_df["miou_mean"].max() - plot_df["miou_mean"].min())

    if annotate_model_names:
        for idx, row in enumerate(plot_df.itertuples(index=False)):
            x_offset_pts = 6
            y_offset_pts = 8 if idx % 2 == 0 else -10
            if str(row.model_name) == "sim":
                y_offset_pts = -10
            ax.annotate(
                str(row.display_name),
                (float(getattr(row, x_metric)), float(row.miou_mean)),
                xytext=(x_offset_pts, y_offset_pts),
                textcoords="offset points",
                fontsize=annotation_fontsize,
                ha="left",
                va="center",
            )

    ax.set_xlabel(X_METRIC_LABELS[x_metric], fontsize=axis_label_fontsize)
    if show_ylabel:
        ax.set_ylabel(r"mIoU $\uparrow$", fontsize=axis_label_fontsize)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=tick_label_fontsize)
    ax.tick_params(axis="y", labelsize=tick_label_fontsize)

    x_min = float(plot_df[x_metric].min())
    x_max = float(plot_df[x_metric].max())
    y_min = float((plot_df["miou_mean"] - plot_df["miou_std"]).min())
    y_max = float((plot_df["miou_mean"] + plot_df["miou_std"]).max())
    if x_metric == "ber":
        x_pad = max(0.04 * x_span, 0.005)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    else:
        ax.set_xlim(x_min - 0.08 * max(x_span, 1.0), x_max + 0.18 * max(x_span, 1.0))
    ax.set_ylim(y_min - 0.12 * max(y_span, 0.05), y_max + 0.12 * max(y_span, 0.05))

    if show_spearman_rho and len(translated_df) >= 2:
        rho = _spearman_rho(translated_df, x_metric=x_metric)
        ax.text(
            0.03,
            0.97,
            rf"Spearman $\rho$ = {rho:.2f}" "\n" r"(translated only)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=legend_fontsize,
            bbox=ANNOTATION_BBOX,
        )


def load_deeplab_miou_rows(
    csv_path: str | Path,
    x_metric: str = "deeplab_fd",
    sim_x_metric_override: float | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Final-results CSV does not exist: {csv_path}")

    if x_metric not in X_METRIC_LABELS:
        raise ValueError(
            f"Unsupported x_metric '{x_metric}'. Expected one of: {', '.join(X_METRIC_LABELS)}"
        )

    df = pd.read_csv(csv_path).copy()
    required_columns = {"model_name", x_metric, "miou_mean", "miou_std"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            "Final-results CSV is missing required columns: "
            + ", ".join(missing)
        )

    keep_names = set(MODEL_NAMES.keys()) | {"sim"}
    plot_df = df.loc[df["model_name"].astype(str).isin(keep_names)].copy()
    if plot_df.empty:
        raise ValueError("No translated-model or sim rows were found in the final-results CSV.")

    plot_df["model_name"] = plot_df["model_name"].astype(str)
    plot_df["display_name"] = plot_df["model_name"].map(_display_model_name)
    plot_df[x_metric] = plot_df[x_metric].map(_parse_optional_float)
    plot_df["miou_mean"] = plot_df["miou_mean"].map(_parse_optional_float)
    plot_df["miou_std"] = plot_df["miou_std"].map(_parse_optional_float)

    sim_mask = plot_df["model_name"] == "sim"
    if sim_mask.any() and plot_df.loc[sim_mask, x_metric].isna().any():
        if sim_x_metric_override is None:
            raise ValueError(
                f"The sim baseline row is missing {x_metric} in the final-results CSV. "
                "Pass sim_x_metric_override to place the sim point on the x-axis."
            )
        plot_df.loc[sim_mask, x_metric] = float(sim_x_metric_override)

    if plot_df[x_metric].isna().any():
        missing_models = plot_df.loc[plot_df[x_metric].isna(), "model_name"].tolist()
        raise ValueError(
            f"{x_metric} is missing for: " + ", ".join(str(name) for name in missing_models)
        )
    if plot_df["miou_mean"].isna().any():
        missing_models = plot_df.loc[plot_df["miou_mean"].isna(), "model_name"].tolist()
        raise ValueError(
            "miou_mean is missing for: " + ", ".join(str(name) for name in missing_models)
        )
    if plot_df["miou_std"].isna().any():
        missing_models = plot_df.loc[plot_df["miou_std"].isna(), "model_name"].tolist()
        raise ValueError(
            "miou_std is missing for: " + ", ".join(str(name) for name in missing_models)
        )

    model_order = ["sim", *MODEL_NAMES.keys()]
    plot_df["model_order"] = plot_df["model_name"].map(
        lambda name: model_order.index(name) if name in model_order else len(model_order)
    )
    plot_df = plot_df.sort_values([x_metric, "model_order"]).reset_index(drop=True)
    return plot_df


def plot_metric_vs_miou(
    final_results_csv: str | Path,
    save_path: str | Path,
    x_metric: str = "deeplab_fd",
    sim_x_metric_override: float | None = None,
    show_trend_line: bool = True,
    show_spearman_rho: bool = False,
    annotate_model_names: bool = True,
    legend_use_model_names: bool = False,
    figure_size: tuple[float, float] = (6.6, 4.4),
    annotation_fontsize: float = 10.0,
    axis_label_fontsize: float = 12.0,
    tick_label_fontsize: float = 10.0,
    legend_fontsize: float = 10.0,
) -> Path:
    plot_df = load_deeplab_miou_rows(
        csv_path=final_results_csv,
        x_metric=x_metric,
        sim_x_metric_override=sim_x_metric_override,
    )

    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    _plot_metric_vs_miou_on_axes(
        ax=ax,
        plot_df=plot_df,
        x_metric=x_metric,
        show_trend_line=show_trend_line,
        show_spearman_rho=show_spearman_rho,
        annotate_model_names=annotate_model_names,
        annotation_fontsize=annotation_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_fontsize=legend_fontsize,
        show_ylabel=True,
    )

    legend_handles = _build_legend_handles(
        plot_df=plot_df,
        show_trend_line=show_trend_line,
        legend_use_model_names=legend_use_model_names,
    )
    ax.legend(handles=legend_handles, loc="best", frameon=True, fontsize=legend_fontsize)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return save_path


def plot_deeplab_fd_vs_miou(
    final_results_csv: str | Path,
    save_path: str | Path,
    x_metric: str = "deeplab_fd",
    sim_x_metric_override: float | None = None,
    show_trend_line: bool = True,
    show_spearman_rho: bool = False,
    annotate_model_names: bool = True,
    legend_use_model_names: bool = False,
    figure_size: tuple[float, float] = (6.6, 4.4),
    annotation_fontsize: float = 10.0,
    axis_label_fontsize: float = 12.0,
    tick_label_fontsize: float = 10.0,
    legend_fontsize: float = 10.0,
) -> Path:
    return plot_metric_vs_miou(
        final_results_csv=final_results_csv,
        save_path=save_path,
        x_metric=x_metric,
        sim_x_metric_override=sim_x_metric_override,
        show_trend_line=show_trend_line,
        show_spearman_rho=show_spearman_rho,
        annotate_model_names=annotate_model_names,
        legend_use_model_names=legend_use_model_names,
        figure_size=figure_size,
        annotation_fontsize=annotation_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_fontsize=legend_fontsize,
    )


def plot_side_by_side_metrics_vs_miou(
    final_results_csv: str | Path,
    save_path: str | Path,
    left_x_metric: str = "fid",
    right_x_metric: str = "deeplab_fd",
    sim_left_x_metric_override: float | None = None,
    sim_right_x_metric_override: float | None = None,
    show_trend_line: bool = False,
    show_spearman_rho: bool = False,
    annotate_model_names: bool = False,
    legend_use_model_names: bool = True,
    figure_size: tuple[float, float] = (12.8, 4.4),
    annotation_fontsize: float = 10.0,
    axis_label_fontsize: float = 12.0,
    tick_label_fontsize: float = 10.0,
    legend_fontsize: float = 10.0,
) -> Path:
    return plot_multi_metrics_vs_miou(
        final_results_csv=final_results_csv,
        save_path=save_path,
        x_metrics=(left_x_metric, right_x_metric),
        sim_x_metric_overrides=(
            sim_left_x_metric_override,
            sim_right_x_metric_override,
        ),
        show_trend_line=show_trend_line,
        show_spearman_rho=show_spearman_rho,
        annotate_model_names=annotate_model_names,
        legend_use_model_names=legend_use_model_names,
        figure_size=figure_size,
        annotation_fontsize=annotation_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        legend_fontsize=legend_fontsize,
    )


def plot_multi_metrics_vs_miou(
    final_results_csv: str | Path,
    save_path: str | Path,
    x_metrics: Sequence[str] = ("fid", "deeplab_fd"),
    sim_x_metric_overrides: Sequence[float | None] | None = None,
    show_trend_line: bool = False,
    show_spearman_rho: bool = False,
    annotate_model_names: bool = False,
    legend_use_model_names: bool = True,
    figure_size: tuple[float, float] | None = None,
    annotation_fontsize: float = 10.0,
    axis_label_fontsize: float = 12.0,
    tick_label_fontsize: float = 10.0,
    legend_fontsize: float = 10.0,
) -> Path:
    metric_names = tuple(str(metric) for metric in x_metrics)
    if not metric_names:
        raise ValueError("x_metrics must contain at least one metric.")

    if sim_x_metric_overrides is None:
        metric_overrides = (None,) * len(metric_names)
    else:
        metric_overrides = tuple(sim_x_metric_overrides)
        if len(metric_overrides) != len(metric_names):
            raise ValueError(
                "sim_x_metric_overrides must match x_metrics length when provided."
            )

    plot_dfs = [
        load_deeplab_miou_rows(
            csv_path=final_results_csv,
            x_metric=x_metric,
            sim_x_metric_override=sim_override,
        )
        for x_metric, sim_override in zip(metric_names, metric_overrides)
    ]

    if figure_size is None:
        figure_size = (6.2 * len(metric_names), 4.4)

    fig, axes = plt.subplots(
        1,
        len(metric_names),
        figsize=figure_size,
        constrained_layout=True,
        sharey=True,
    )
    axes_list = [axes] if len(metric_names) == 1 else list(axes)

    for idx, (ax, plot_df, x_metric) in enumerate(zip(axes_list, plot_dfs, metric_names)):
        _plot_metric_vs_miou_on_axes(
            ax=ax,
            plot_df=plot_df,
            x_metric=x_metric,
            show_trend_line=show_trend_line,
            show_spearman_rho=show_spearman_rho,
            annotate_model_names=annotate_model_names,
            annotation_fontsize=annotation_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            legend_fontsize=legend_fontsize,
            show_ylabel=idx == 0,
        )

    combined_y_min = min(
        float((plot_df["miou_mean"] - plot_df["miou_std"]).min()) for plot_df in plot_dfs
    )
    combined_y_max = max(
        float((plot_df["miou_mean"] + plot_df["miou_std"]).max()) for plot_df in plot_dfs
    )
    combined_y_span = combined_y_max - combined_y_min
    y_pad = 0.12 * max(combined_y_span, 0.05)
    for ax in axes_list:
        ax.set_ylim(combined_y_min - y_pad, combined_y_max + y_pad)

    legend_handles = _build_legend_handles(
        plot_df=plot_dfs[-1],
        show_trend_line=show_trend_line,
        legend_use_model_names=legend_use_model_names,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=legend_fontsize,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return save_path


def main() -> None:
    # Final-results CSV containing the chosen x-axis metric plus miou_mean and miou_std columns.
    final_results_csv = "/develop/code/eval/thesis/final_results_table.csv"
    # Output path for the x-metric vs mIoU scatter plot.
    save_path = "/develop/code/eval/thesis/deeplab_miou/deeplab_miou.pdf"
    # Whether to save a multi-panel metric comparison figure instead of a single plot.
    output_multi_panel = True
    # Output path for the multi-panel comparison figure.
    multi_panel_save_path = "/develop/code/eval/thesis/deeplab_miou/fid_deeplab_ber_miou.pdf"
    # X-axis metric. Supported values: `deeplab_fd`, `fid`, and `ber`.
    x_metric = "deeplab_fd"
    # X-axis metric value for the sim baseline when the CSV leaves that cell blank.
    sim_x_metric_override = None
    # Ordered x-axis metrics for the multi-panel comparison figure.
    multi_panel_x_metrics = ("fid", "deeplab_fd", "ber")
    # Sim overrides for the multi-panel figure when the CSV leaves a metric blank.
    multi_panel_sim_x_metric_overrides = (None, None, None)
    # Whether to overlay a least-squares trend line across the plotted points.
    show_trend_line = False
    # Whether to show a translated-only Spearman rho annotation.
    show_spearman_rho = False
    # Whether to annotate each point directly with its model name.
    annotate_model_names = False
    # Whether the legend should list the model names instead of generic point categories.
    legend_use_model_names = True
    # Figure width and height in inches.
    figure_size = (6.6, 4.4)
    # Font size for per-point model annotations.
    annotation_fontsize = 10.0
    # Font size for x/y-axis labels.
    axis_label_fontsize = 18.0
    # Font size for axis tick labels.
    tick_label_fontsize = 14.0
    # Font size for the legend.
    legend_fontsize = 14.0

    if output_multi_panel:
        saved_path = plot_multi_metrics_vs_miou(
            final_results_csv=final_results_csv,
            save_path=multi_panel_save_path,
            x_metrics=multi_panel_x_metrics,
            sim_x_metric_overrides=multi_panel_sim_x_metric_overrides,
            show_trend_line=show_trend_line,
            show_spearman_rho=show_spearman_rho,
            annotate_model_names=annotate_model_names,
            legend_use_model_names=legend_use_model_names,
            figure_size=(6.2 * len(multi_panel_x_metrics), figure_size[1]),
            annotation_fontsize=annotation_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            legend_fontsize=legend_fontsize,
        )
        print(f"Saved multi-panel x-metric vs mIoU plot to {saved_path}")
    else:
        saved_path = plot_metric_vs_miou(
            final_results_csv=final_results_csv,
            save_path=save_path,
            x_metric=x_metric,
            sim_x_metric_override=sim_x_metric_override,
            show_trend_line=show_trend_line,
            show_spearman_rho=show_spearman_rho,
            annotate_model_names=annotate_model_names,
            legend_use_model_names=legend_use_model_names,
            figure_size=figure_size,
            annotation_fontsize=annotation_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_label_fontsize=tick_label_fontsize,
            legend_fontsize=legend_fontsize,
        )
        print(f"Saved x-metric vs mIoU plot to {saved_path}")


if __name__ == "__main__":
    main()
