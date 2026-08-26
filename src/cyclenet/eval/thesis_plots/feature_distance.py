from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cyclenet.eval.plotting.set_style import CLASS_NAMES, MODEL_COLORS, MODEL_NAMES, apply_style

apply_style()

FEATURE_EXTRACTOR_LABELS = {
    "deeplab": "DeepLab-FD",
    "fid": "FID Feature Distance",
}


def _normalize_feature_extractor(feature_extractor: str) -> str:
    normalized = feature_extractor.lower().strip()
    if normalized not in FEATURE_EXTRACTOR_LABELS:
        raise ValueError(
            f"Unsupported feature_extractor '{feature_extractor}'. "
            f"Expected one of: {', '.join(sorted(FEATURE_EXTRACTOR_LABELS))}."
        )
    return normalized


def _feature_distance_label(feature_extractor: str, metric_label: str | None = None) -> str:
    if metric_label is not None:
        return metric_label
    return FEATURE_EXTRACTOR_LABELS[_normalize_feature_extractor(feature_extractor)]


def _default_analysis_csv_path(cache_dir: str | Path, feature_extractor: str) -> Path:
    feature_extractor = _normalize_feature_extractor(feature_extractor)
    analysis_dir = "analysis" if feature_extractor == "deeplab" else f"analysis_{feature_extractor}"
    return Path(cache_dir) / analysis_dir / "frechet_distance_delta_vs_sim_by_class.csv"


def _default_output_stem(base_stem: str, feature_extractor: str) -> str:
    feature_extractor = _normalize_feature_extractor(feature_extractor)
    if feature_extractor == "deeplab":
        return base_stem
    return f"{base_stem}_{feature_extractor}"


def _display_model_name(model_name: str) -> str:
    display_name = MODEL_NAMES.get(model_name, "")
    display_name = display_name if display_name else model_name.replace("_", " ")
    if display_name == "RGB + SPADE (BN Only)":
        return "RGB + SPADE\n(BN Only)"
    if display_name == "RGB + SPADE":
        return "RGB +\nSPADE"
    if display_name == "Seg + SPADE":
        return "Seg +\nSPADE"
    return display_name


def _display_class_name(class_name: str) -> str:
    display_name = CLASS_NAMES.get(class_name, "")
    return display_name if display_name else class_name.replace("_", " ").title()


def _class_slug_to_display_name(slug: str) -> str:
    display_name = CLASS_NAMES.get(slug, "")
    return display_name if display_name else slug.replace("_", " ").title()


def _display_name_to_class_slug(display_name: str) -> str:
    normalized_display_name = display_name.strip().casefold()
    for slug, canonical_display_name in CLASS_NAMES.items():
        if canonical_display_name.casefold() == normalized_display_name:
            return slug
    return display_name


def _model_color(model_name: str) -> str:
    color = MODEL_COLORS.get(model_name, "")
    return color if color else "#6f6f6f"


def _ordered_models(df: pd.DataFrame) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for model_name in df["comparison_dataset"].tolist():
        model_name = str(model_name)
        if model_name not in seen:
            seen.add(model_name)
            ordered.append(model_name)
    return ordered


def load_delta_fd_table(analysis_csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(analysis_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Delta-FD analysis CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    required_columns = {
        "comparison_dataset",
        "label_id",
        "label_name",
        "frechet_distance_delta_vs_baseline",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            f"Delta-FD analysis CSV is missing required columns: {', '.join(missing)}"
        )
    if df.empty:
        raise ValueError(f"Delta-FD analysis CSV is empty: {csv_path}")
    return df


def build_delta_fd_grid(delta_df: pd.DataFrame) -> pd.DataFrame:
    model_order = _ordered_models(delta_df)
    if "label_slug" in delta_df.columns:
        class_key_col = "label_slug"
    else:
        class_key_col = "label_name"
        delta_df = delta_df.copy()
        delta_df[class_key_col] = delta_df["label_name"].map(
            lambda display_name: _display_name_to_class_slug(str(display_name))
        )

    present_classes = set(delta_df[class_key_col].astype(str).tolist())
    class_order = [class_name for class_name in CLASS_NAMES.keys() if class_name in present_classes]
    remaining = sorted(present_classes - set(class_order))
    class_order = class_order + remaining

    grid_df = delta_df.pivot(
        index="comparison_dataset",
        columns=class_key_col,
        values="frechet_distance_delta_vs_baseline",
    )
    grid_df = grid_df.reindex(index=model_order, columns=class_order)
    grid_df.index = [_display_model_name(model_name) for model_name in grid_df.index]
    grid_df.columns = [_class_slug_to_display_name(class_name) for class_name in grid_df.columns]
    return grid_df


def plot_delta_fd_heatmap(
    delta_df: pd.DataFrame,
    save_path: str | Path,
    annotate: bool = True,
    cmap: str = "RdBu_r",
    feature_extractor: str = "deeplab",
    metric_label: str | None = None,
    column_header_fontsize: float = 12.0,
    row_header_fontsize: float = 12.0,
    axis_label_fontsize: float = 12.0,
    cell_annotation_fontsize: float = 8.5,
    colorbar_label_fontsize: float = 12.0,
    colorbar_tick_fontsize: float = 11.0,
) -> Path:
    grid_df = build_delta_fd_grid(delta_df)
    values = grid_df.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite delta-FD values available to plot.")

    metric_label = _feature_distance_label(feature_extractor, metric_label)
    vmax = float(np.max(np.abs(finite_values)))
    vmax = max(vmax, 1e-6)

    fig_w = max(6.4, 1.0 + 1.05 * grid_df.shape[1])
    fig_h = max(3.2, 1.0 + 0.58 * grid_df.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        values,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(grid_df.shape[1]))
    ax.set_xticklabels(list(grid_df.columns), rotation=25, ha="right", fontsize=column_header_fontsize)
    ax.set_yticks(np.arange(grid_df.shape[0]))
    ax.set_yticklabels(list(grid_df.index), fontsize=row_header_fontsize)
    ax.set_xlabel("Class", fontsize=axis_label_fontsize)
    ax.set_ylabel("Model Family", labelpad=12, fontsize=axis_label_fontsize)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for y_idx in range(grid_df.shape[0] + 1):
        ax.axhline(y_idx - 0.5, color="white", linewidth=0.8, alpha=0.9)
    for x_idx in range(grid_df.shape[1] + 1):
        ax.axvline(x_idx - 0.5, color="white", linewidth=0.8, alpha=0.9)

    if annotate:
        for row_idx in range(grid_df.shape[0]):
            for col_idx in range(grid_df.shape[1]):
                value = values[row_idx, col_idx]
                if not np.isfinite(value):
                    label = "NA"
                    text_color = "black"
                else:
                    label = f"{value:.2f}"
                    text_color = "white" if abs(value) > 0.45 * vmax else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=cell_annotation_fontsize,
                    color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label(rf"$\Delta$ {metric_label} $\downarrow$", fontsize=colorbar_label_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_delta_fd_macro_bar(
    delta_df: pd.DataFrame,
    save_path: str | Path,
    feature_extractor: str = "deeplab",
    metric_label: str | None = None,
    x_tick_fontsize: float = 12.0,
    y_tick_fontsize: float = 12.0,
    axis_label_fontsize: float = 12.0,
    value_label_fontsize: float = 9.0,
) -> Path:
    summary_df = (
        delta_df.groupby("comparison_dataset", as_index=False)["frechet_distance_delta_vs_baseline"]
        .mean(numeric_only=True)
        .rename(columns={"frechet_distance_delta_vs_baseline": "macro_avg_delta_fd"})
    )
    model_order = _ordered_models(delta_df)
    summary_df["comparison_dataset"] = pd.Categorical(
        summary_df["comparison_dataset"],
        categories=model_order,
        ordered=True,
    )
    summary_df = summary_df.sort_values("comparison_dataset").reset_index(drop=True)
    summary_df["display_name"] = summary_df["comparison_dataset"].map(lambda name: _display_model_name(str(name)))
    summary_df["color"] = summary_df["comparison_dataset"].map(lambda name: _model_color(str(name)))
    metric_label = _feature_distance_label(feature_extractor, metric_label)

    fig_w = max(6.4, 1.4 + 1.25 * len(summary_df))
    fig, ax = plt.subplots(figsize=(fig_w, 3.0))
    bars = ax.bar(
        summary_df["display_name"],
        summary_df["macro_avg_delta_fd"],
        color=summary_df["color"],
        edgecolor="black",
        linewidth=0.8,
    )

    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel(
        "Macro-Averaged\n" + rf"$\Delta$ {metric_label} $\downarrow$",
        fontsize=axis_label_fontsize,
    )
    ax.set_xlabel("Model Family", labelpad=12, fontsize=axis_label_fontsize)
    ax.tick_params(axis="x", rotation=0, labelsize=x_tick_fontsize)
    ax.tick_params(axis="y", labelsize=y_tick_fontsize)

    y_values = summary_df["macro_avg_delta_fd"].to_numpy(dtype=float)
    y_extent = max(np.max(np.abs(y_values)), 1e-6)
    ax.set_ylim(
        float(np.min(y_values)) - 0.18 * y_extent,
        float(np.max(y_values)) + 0.18 * y_extent,
    )

    for bar, value in zip(bars, y_values, strict=True):
        y_text = value + (0.03 * y_extent if value >= 0 else -0.05 * y_extent)
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text,
            f"{value:.2f}",
            ha="center",
            va=va,
            fontsize=value_label_fontsize,
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def make_feature_distance_plots(
    analysis_csv_path: str | Path,
    heatmap_save_path: str | Path,
    bar_save_path: str | Path,
    annotate_heatmap: bool = True,
    feature_extractor: str = "deeplab",
    metric_label: str | None = None,
    heatmap_column_header_fontsize: float = 12.0,
    heatmap_row_header_fontsize: float = 12.0,
    heatmap_axis_label_fontsize: float = 12.0,
    heatmap_cell_annotation_fontsize: float = 8.5,
    heatmap_colorbar_label_fontsize: float = 12.0,
    heatmap_colorbar_tick_fontsize: float = 11.0,
    bar_x_tick_fontsize: float = 12.0,
    bar_y_tick_fontsize: float = 12.0,
    bar_axis_label_fontsize: float = 12.0,
    bar_value_label_fontsize: float = 9.0,
) -> tuple[Path, Path]:
    delta_df = load_delta_fd_table(analysis_csv_path)
    heatmap_path = plot_delta_fd_heatmap(
        delta_df=delta_df,
        save_path=heatmap_save_path,
        annotate=annotate_heatmap,
        feature_extractor=feature_extractor,
        metric_label=metric_label,
        column_header_fontsize=heatmap_column_header_fontsize,
        row_header_fontsize=heatmap_row_header_fontsize,
        axis_label_fontsize=heatmap_axis_label_fontsize,
        cell_annotation_fontsize=heatmap_cell_annotation_fontsize,
        colorbar_label_fontsize=heatmap_colorbar_label_fontsize,
        colorbar_tick_fontsize=heatmap_colorbar_tick_fontsize,
    )
    bar_path = plot_delta_fd_macro_bar(
        delta_df=delta_df,
        save_path=bar_save_path,
        feature_extractor=feature_extractor,
        metric_label=metric_label,
        x_tick_fontsize=bar_x_tick_fontsize,
        y_tick_fontsize=bar_y_tick_fontsize,
        axis_label_fontsize=bar_axis_label_fontsize,
        value_label_fontsize=bar_value_label_fontsize,
    )
    return heatmap_path, bar_path


def main() -> None:
    # Feature-distance analysis to plot. Supported values: `deeplab` and `fid`.
    feature_extractor = "deeplab"
    # Optional plot label override. Use `None` for the feature-extractor default.
    metric_label = None
    # Root cache directory produced by `cache_class_features.py`.
    cache_dir = "/develop/code/eval/thesis/class_feature_cache"
    # CSV produced by the class-feature-distance analysis helper. Use `None`
    # to infer `analysis/` for DeepLab and `analysis_fid/` for FID.
    analysis_csv_path = None
    # Output directory for the class-feature-distance thesis plots.
    out_dir = "/develop/code/eval/thesis/feature_distance"
    # Whether to annotate each heatmap cell numerically.
    annotate_heatmap = True
    # Font size for heatmap class column headers.
    heatmap_column_header_fontsize = 12.0
    # Font size for heatmap model row headers.
    heatmap_row_header_fontsize = 12.0
    # Font size for heatmap x/y-axis labels.
    heatmap_axis_label_fontsize = 14.0
    # Font size for numeric heatmap cell annotations.
    heatmap_cell_annotation_fontsize = 11.0
    # Font size for the heatmap colorbar label.
    heatmap_colorbar_label_fontsize = 14.0
    # Font size for the heatmap colorbar tick labels.
    heatmap_colorbar_tick_fontsize = 11.0
    # Font size for bar-chart model labels on the x-axis.
    bar_x_tick_fontsize = 11.0
    # Font size for bar-chart numeric y-axis ticks.
    bar_y_tick_fontsize = 11.0
    # Font size for bar-chart x/y-axis labels.
    bar_axis_label_fontsize = 12.0
    # Font size for the numeric labels above each bar.
    bar_value_label_fontsize = 11.0

    if analysis_csv_path is None:
        analysis_csv_path = _default_analysis_csv_path(cache_dir, feature_extractor)
    output_stem_suffix = _default_output_stem("delta_fd", feature_extractor).removeprefix("delta_fd")
    heatmap_save_path = Path(out_dir) / f"delta_fd_heatmap{output_stem_suffix}.pdf"
    bar_save_path = Path(out_dir) / f"delta_fd_macro_bar{output_stem_suffix}.pdf"

    heatmap_path, bar_path = make_feature_distance_plots(
        analysis_csv_path=analysis_csv_path,
        heatmap_save_path=heatmap_save_path,
        bar_save_path=bar_save_path,
        annotate_heatmap=annotate_heatmap,
        feature_extractor=feature_extractor,
        metric_label=metric_label,
        heatmap_column_header_fontsize=heatmap_column_header_fontsize,
        heatmap_row_header_fontsize=heatmap_row_header_fontsize,
        heatmap_axis_label_fontsize=heatmap_axis_label_fontsize,
        heatmap_cell_annotation_fontsize=heatmap_cell_annotation_fontsize,
        heatmap_colorbar_label_fontsize=heatmap_colorbar_label_fontsize,
        heatmap_colorbar_tick_fontsize=heatmap_colorbar_tick_fontsize,
        bar_x_tick_fontsize=bar_x_tick_fontsize,
        bar_y_tick_fontsize=bar_y_tick_fontsize,
        bar_axis_label_fontsize=bar_axis_label_fontsize,
        bar_value_label_fontsize=bar_value_label_fontsize,
    )
    print(f"Saved delta-FD heatmap to {heatmap_path}")
    print(f"Saved delta-FD macro bar chart to {bar_path}")


if __name__ == "__main__":
    main()
