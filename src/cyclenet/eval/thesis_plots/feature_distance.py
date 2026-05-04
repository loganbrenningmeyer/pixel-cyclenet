from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cyclenet.eval.plotting.set_style import MODEL_COLORS, MODEL_NAMES, apply_style

apply_style()


def _display_model_name(model_name: str) -> str:
    display_name = MODEL_NAMES.get(model_name, "")
    return display_name if display_name else model_name.replace("_", " ")


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
    class_order_df = (
        delta_df.loc[:, ["label_id", "label_name"]]
        .drop_duplicates()
        .sort_values(["label_id", "label_name"], ascending=[True, True])
    )
    class_order = class_order_df["label_name"].tolist()

    grid_df = delta_df.pivot(
        index="comparison_dataset",
        columns="label_name",
        values="frechet_distance_delta_vs_baseline",
    )
    grid_df = grid_df.reindex(index=model_order, columns=class_order)
    grid_df.index = [_display_model_name(model_name) for model_name in grid_df.index]
    return grid_df


def plot_delta_fd_heatmap(
    delta_df: pd.DataFrame,
    save_path: str | Path,
    annotate: bool = True,
    cmap: str = "RdBu_r",
) -> Path:
    grid_df = build_delta_fd_grid(delta_df)
    values = grid_df.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("No finite delta-FD values available to plot.")

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
    ax.set_xticklabels(list(grid_df.columns), rotation=25, ha="right")
    ax.set_yticks(np.arange(grid_df.shape[0]))
    ax.set_yticklabels(list(grid_df.index))
    ax.set_xlabel("Class")
    ax.set_ylabel("Model Type")
    ax.set_title(r"Class-wise $\Delta$ Fr\'echet Distance vs Sim")

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
                    fontsize=8.5,
                    color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label(r"$\Delta$ FD vs Sim")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_delta_fd_macro_bar(
    delta_df: pd.DataFrame,
    save_path: str | Path,
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

    fig_w = max(6.4, 1.4 + 1.25 * len(summary_df))
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))
    bars = ax.bar(
        summary_df["display_name"],
        summary_df["macro_avg_delta_fd"],
        color=summary_df["color"],
        edgecolor="black",
        linewidth=0.8,
    )

    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_ylabel(r"Macro-Averaged $\Delta$ FD vs Sim")
    ax.set_xlabel("Model Type")
    ax.set_title(r"Macro-Averaged $\Delta$ Fr\'echet Distance vs Sim")
    ax.tick_params(axis="x", rotation=25)

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
            fontsize=9,
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
) -> tuple[Path, Path]:
    delta_df = load_delta_fd_table(analysis_csv_path)
    heatmap_path = plot_delta_fd_heatmap(
        delta_df=delta_df,
        save_path=heatmap_save_path,
        annotate=annotate_heatmap,
    )
    bar_path = plot_delta_fd_macro_bar(
        delta_df=delta_df,
        save_path=bar_save_path,
    )
    return heatmap_path, bar_path


def main() -> None:
    # CSV produced by the class-feature-distance analysis helper.
    analysis_csv_path = (
        "/develop/code/eval/thesis/class_feature_cache/analysis/"
        "frechet_distance_delta_vs_sim_by_class.csv"
    )
    # Output path for the class-wise delta-FD heatmap.
    heatmap_save_path = "/develop/code/eval/thesis/feature_distance/delta_fd_heatmap.pdf"
    # Output path for the macro-averaged delta-FD bar chart.
    bar_save_path = "/develop/code/eval/thesis/feature_distance/delta_fd_macro_bar.pdf"
    # Whether to annotate each heatmap cell numerically.
    annotate_heatmap = True

    heatmap_path, bar_path = make_feature_distance_plots(
        analysis_csv_path=analysis_csv_path,
        heatmap_save_path=heatmap_save_path,
        bar_save_path=bar_save_path,
        annotate_heatmap=annotate_heatmap,
    )
    print(f"Saved delta-FD heatmap to {heatmap_path}")
    print(f"Saved delta-FD macro bar chart to {bar_path}")


if __name__ == "__main__":
    main()
