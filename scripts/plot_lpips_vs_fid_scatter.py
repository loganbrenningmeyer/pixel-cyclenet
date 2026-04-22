#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cyclenet.eval.plotting.set_style import apply_style

apply_style()


MERGE_KEYS = ["step", "noise_strength", "cfg_weight"]
REQUIRED_LPIPS_COLUMNS = set(MERGE_KEYS) | {"lpips_mean"}
REQUIRED_FID_COLUMNS = set(MERGE_KEYS) | {"fid"}


def validate_columns(df: pd.DataFrame, required_columns: set[str], csv_label: str) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{csv_label} CSV is missing required columns: {', '.join(missing)}")


def load_merged_metrics(lpips_csv_path: Path, fid_csv_path: Path) -> pd.DataFrame:
    if not lpips_csv_path.exists():
        raise FileNotFoundError(f"LPIPS CSV does not exist: {lpips_csv_path}")
    if not fid_csv_path.exists():
        raise FileNotFoundError(f"FID CSV does not exist: {fid_csv_path}")

    lpips_df = pd.read_csv(lpips_csv_path)
    fid_df = pd.read_csv(fid_csv_path)
    validate_columns(lpips_df, REQUIRED_LPIPS_COLUMNS, csv_label="LPIPS stats")
    validate_columns(fid_df, REQUIRED_FID_COLUMNS, csv_label="FID stats")

    merged = lpips_df.merge(
        fid_df[list(MERGE_KEYS) + ["fid"]],
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(
            "No shared `(step, noise_strength, cfg_weight)` rows were found between "
            f"{lpips_csv_path} and {fid_csv_path}."
        )

    merged = merged.sort_values(MERGE_KEYS).reset_index(drop=True)
    merged["point_label"] = merged.apply(
        lambda row: build_point_label(
            step=int(row["step"]),
            noise_strength=float(row["noise_strength"]),
            cfg_weight=float(row["cfg_weight"]),
            include_step=merged["step"].nunique() > 1,
        ),
        axis=1,
    )
    return merged


def build_point_label(
    step: int,
    noise_strength: float,
    cfg_weight: float,
    include_step: bool,
) -> str:
    if include_step:
        return f"s{step}, $\\gamma={noise_strength:g}$, $w={cfg_weight:g}$"
    return f"$\\gamma={noise_strength:g}$, $w={cfg_weight:g}$"


def format_step_label(step: int) -> str:
    if step % 1000 == 0:
        return f"{step // 1000}k checkpoint"
    return f"step {step} checkpoint"


def compute_pareto_frontier(
    df: pd.DataFrame,
    x_col: str = "lpips_mean",
    y_col: str = "fid",
) -> pd.DataFrame:
    """
    Returns the lower-left Pareto frontier, where lower LPIPS and lower FID are
    both preferred.
    """
    ordered = df.sort_values([x_col, y_col], ascending=[True, True]).copy()
    frontier_rows: list[int] = []
    best_y = float("inf")

    for row_idx, row in ordered.iterrows():
        y = float(row[y_col])
        if y < best_y:
            frontier_rows.append(row_idx)
            best_y = y

    frontier = ordered.loc[frontier_rows].copy()
    return frontier.sort_values([x_col, y_col]).reset_index(drop=True)


def select_annotation_rows(
    df: pd.DataFrame,
    annotation_mode: str,
    top_k: int | None,
) -> pd.DataFrame:
    mode = annotation_mode.lower()
    if mode == "none":
        return df.iloc[0:0].copy()
    if mode == "all":
        return df

    if top_k is None or top_k <= 0:
        raise ValueError("top_k_annotations must be positive when using a top-k annotation mode.")

    if mode == "best_fid":
        return df.nsmallest(top_k, "fid")
    if mode == "best_lpips":
        return df.nsmallest(top_k, "lpips_mean")
    if mode == "best_joint":
        ranked = df.copy()
        ranked["fid_rank"] = ranked["fid"].rank(method="dense", ascending=True)
        ranked["lpips_rank"] = ranked["lpips_mean"].rank(method="dense", ascending=True)
        ranked["joint_rank"] = ranked["fid_rank"] + ranked["lpips_rank"]
        return ranked.nsmallest(top_k, "joint_rank")

    raise ValueError(
        f"Unsupported annotation_mode '{annotation_mode}'. "
        "Expected one of: none, all, best_fid, best_lpips, best_joint."
    )


def plot_lpips_vs_fid_scatter(
    merged_df: pd.DataFrame,
    save_path: Path,
    title: str,
    split_by_step: bool = True,
    show_step_titles: bool = True,
    connect_by_noise_strength: bool = True,
    show_pareto_frontier: bool = True,
    highlight_best_joint: bool = True,
    annotation_mode: str = "none",
    top_k_annotations: int | None = 6,
    point_size: float = 46.0,
    alpha: float = 0.92,
    label_fontsize: float = 8.0,
    annotation_dx: float = 0.002,
    annotation_dy: float = 0.6,
    noise_palette: str = "viridis",
    line_alpha: float = 0.42,
    frontier_color: str = "#111827",
    best_joint_color: str = "#dc2626",
    start_cfg_marker: str = "s",
    end_cfg_marker: str = ">",
) -> None:
    step_values = sorted(int(step) for step in merged_df["step"].unique())
    if not step_values:
        raise ValueError("Merged LPIPS/FID dataframe is empty.")

    panel_steps = step_values if split_by_step else [step_values[0]]
    n_panels = len(panel_steps) if split_by_step else 1

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(5.3 * n_panels, 5.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    noise_strengths = sorted(float(value) for value in merged_df["noise_strength"].unique())
    palette = sns.color_palette(noise_palette, n_colors=max(len(noise_strengths), 2))
    noise_colors = {
        noise_strength: palette[idx]
        for idx, noise_strength in enumerate(noise_strengths)
    }

    x_min = float(merged_df["lpips_mean"].min())
    x_max = float(merged_df["lpips_mean"].max())
    y_min = float(merged_df["fid"].min())
    y_max = float(merged_df["fid"].max())
    x_pad = max((x_max - x_min) * 0.07, 0.01)
    y_pad = max((y_max - y_min) * 0.08, 1.0)

    for ax_idx, ax in enumerate(axes_flat):
        panel_df = merged_df.copy()
        if split_by_step:
            step = panel_steps[ax_idx]
            panel_df = panel_df.loc[panel_df["step"] == step].copy()
        else:
            step = step_values[0]

        if connect_by_noise_strength:
            for noise_strength in noise_strengths:
                group_df = panel_df.loc[panel_df["noise_strength"] == noise_strength].copy()
                if group_df.empty:
                    continue
                group_df = group_df.sort_values("cfg_weight")
                ax.plot(
                    group_df["lpips_mean"],
                    group_df["fid"],
                    color=noise_colors[noise_strength],
                    linewidth=1.1,
                    alpha=line_alpha,
                    zorder=1,
                )

                start_row = group_df.iloc[0]
                end_row = group_df.iloc[-1]
                ax.scatter(
                    [float(start_row["lpips_mean"])],
                    [float(start_row["fid"])],
                    s=point_size * 1.35,
                    alpha=alpha,
                    color=noise_colors[noise_strength],
                    marker=start_cfg_marker,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
                ax.scatter(
                    [float(end_row["lpips_mean"])],
                    [float(end_row["fid"])],
                    s=point_size * 1.55,
                    alpha=alpha,
                    color=noise_colors[noise_strength],
                    marker=end_cfg_marker,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )

        for noise_strength in noise_strengths:
            group_df = panel_df.loc[panel_df["noise_strength"] == noise_strength].copy()
            if group_df.empty:
                continue
            ax.scatter(
                group_df["lpips_mean"],
                group_df["fid"],
                s=point_size,
                alpha=alpha,
                color=noise_colors[noise_strength],
                edgecolors="white",
                linewidths=0.6,
                zorder=2,
            )

        if show_pareto_frontier:
            frontier_df = compute_pareto_frontier(panel_df)
            ax.plot(
                frontier_df["lpips_mean"],
                frontier_df["fid"],
                color=frontier_color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
                zorder=3,
            )
            ax.scatter(
                frontier_df["lpips_mean"],
                frontier_df["fid"],
                s=point_size * 1.45,
                facecolors="none",
                edgecolors=frontier_color,
                linewidths=1.2,
                zorder=4,
            )

        if highlight_best_joint:
            best_joint_df = select_annotation_rows(panel_df, annotation_mode="best_joint", top_k=1)
            if not best_joint_df.empty:
                best_joint_row = best_joint_df.iloc[0]
                ax.scatter(
                    [float(best_joint_row["lpips_mean"])],
                    [float(best_joint_row["fid"])],
                    s=point_size * 1.8,
                    facecolors=best_joint_color,
                    edgecolors="white",
                    linewidths=0.9,
                    zorder=5,
                )

        annotation_rows = select_annotation_rows(
            panel_df,
            annotation_mode=annotation_mode,
            top_k=top_k_annotations,
        )
        for _, row in annotation_rows.iterrows():
            ax.annotate(
                str(row["point_label"]),
                (float(row["lpips_mean"]), float(row["fid"])),
                xytext=(float(row["lpips_mean"]) + annotation_dx, float(row["fid"]) + annotation_dy),
                textcoords="data",
                fontsize=label_fontsize,
                alpha=0.9,
            )

        if split_by_step and show_step_titles:
            ax.set_title(format_step_label(step))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(alpha=0.25)
        ax.set_xlabel("LPIPS ($\\downarrow$)")

    axes_flat[0].set_ylabel("FID ($\\downarrow$)")

    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=noise_colors[noise_strength],
            marker="o",
            linestyle="-",
            linewidth=1.1,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=fr"$\gamma={noise_strength:g}$",
        )
        for noise_strength in noise_strengths
    ]
    if show_pareto_frontier:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=frontier_color,
                linestyle="--",
                linewidth=1.2,
                label="Pareto frontier",
            )
        )
    if highlight_best_joint:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=best_joint_color,
                marker="o",
                linestyle="None",
                markersize=6,
                label="Best joint tradeoff",
            )
        )
    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color="#4b5563",
            marker=start_cfg_marker,
            linestyle="None",
            markersize=6,
            label="$w = 1$",
        )
    )
    legend_handles.append(
        mlines.Line2D(
            [],
            [],
            color="#4b5563",
            marker=end_cfg_marker,
            linestyle="None",
            markersize=6.5,
            label="$w = 5$",
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(legend_handles), 7),
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(title, y=1.08)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main() -> None:
    # Path to the LPIPS sweep CSV generated by `src/cyclenet/eval/lpips.py`.
    lpips_csv_path = Path("/path/to/lpips_stats.csv")

    # Path to the FID sweep CSV generated by `src/cyclenet/eval/fid.py`.
    fid_csv_path = Path("/path/to/fid_stats.csv")

    # Output directory for the merged CSV and scatter plot.
    out_dir = Path("/tmp/lpips_vs_fid")

    # Plot title.
    title = "LPIPS vs FID Tradeoff Across Translation Sweep"

    # Whether to split the figure into one panel per checkpoint step.
    split_by_step = True

    # Whether to connect points with the same noise strength, ordered by CFG weight.
    connect_by_noise_strength = True

    # Whether to draw the lower-left Pareto frontier.
    show_pareto_frontier = True

    # Whether to highlight the best combined LPIPS/FID tradeoff point in each panel.
    highlight_best_joint = True

    # Annotation mode: `none`, `all`, `best_fid`, `best_lpips`, or `best_joint`.
    # For a cleaner thesis figure, `none` is usually the best default.
    annotation_mode = "none"

    # Number of points to annotate for the top-k annotation modes.
    top_k_annotations = 6

    merged_df = load_merged_metrics(
        lpips_csv_path=lpips_csv_path,
        fid_csv_path=fid_csv_path,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_csv_path = out_dir / "lpips_vs_fid_merged.csv"
    merged_df.to_csv(merged_csv_path, index=False)

    save_path = out_dir / "lpips_vs_fid_scatter.pdf"
    plot_lpips_vs_fid_scatter(
        merged_df=merged_df,
        save_path=save_path,
        title=title,
        split_by_step=split_by_step,
        connect_by_noise_strength=connect_by_noise_strength,
        show_pareto_frontier=show_pareto_frontier,
        highlight_best_joint=highlight_best_joint,
        annotation_mode=annotation_mode,
        top_k_annotations=top_k_annotations,
    )

    print(f"Saved merged LPIPS/FID CSV to {merged_csv_path}")
    print(f"Saved LPIPS vs FID scatter plot to {save_path}")


if __name__ == "__main__":
    main()
