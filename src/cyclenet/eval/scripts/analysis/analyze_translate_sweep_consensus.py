import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


LOWER_BETTER = {
    "real_fid",
    "real_clip_centroid_l2",
    "real_clip_frechet",
    "real_clip_mmd_rbf",
}

HIGHER_BETTER = {
    "real_clip_centroid_cosine",
    "real_clip_nearest_cosine_mean",
}

PRESERVATION_METRICS = {
    "source_lpips_mean",
}

DEFAULT_REALISM_WEIGHTS = {
    "real_fid": 1.0,
    "real_clip_centroid_l2": 1.0,
    "real_clip_centroid_cosine": 1.0,
    "real_clip_nearest_cosine_mean": 0.5,
    "real_clip_frechet": 2.0,
    "real_clip_mmd_rbf": 2.0,
}


def as_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    value = str(value).strip()
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def finite(value: Any) -> bool:
    value = as_float(value)
    return math.isfinite(value)


def read_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for key in [
            "cfg_weight",
            "noise_strength",
            "num_fake",
            "num_real",
            *LOWER_BETTER,
            *HIGHER_BETTER,
            *PRESERVATION_METRICS,
        ]:
            if key in row:
                row[key] = as_float(row[key])

        row["checkpoint_step"] = checkpoint_step(row.get("checkpoint", ""))

    return rows


def checkpoint_step(value: Any) -> float:
    text = str(value)
    if text == "" or text.lower() == "source":
        return math.nan
    stem = Path(text).stem
    if stem.startswith("step-"):
        stem = stem.removeprefix("step-")
    try:
        return float(stem)
    except ValueError:
        return math.nan


def is_candidate(row: dict[str, Any]) -> bool:
    kind = str(row.get("kind", "")).strip()
    comparison = str(row.get("comparison", "")).strip()
    if kind:
        return kind == "translated_candidate"
    if comparison:
        return comparison == "translated_sim_vs_real"
    return finite(row.get("cfg_weight")) and finite(row.get("noise_strength"))


def find_baseline(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        kind = str(row.get("kind", "")).strip()
        comparison = str(row.get("comparison", "")).strip()
        if kind == "source_baseline" or comparison == "source_sim_vs_real":
            return row
    return None


def available_metrics(rows: list[dict[str, Any]], metrics: set[str]) -> list[str]:
    out = []
    for metric in sorted(metrics):
        if any(finite(row.get(metric)) for row in rows):
            out.append(metric)
    return out


def normalized_scores(
    rows: list[dict[str, Any]],
    metric: str,
    direction: str,
) -> dict[int, float]:
    values = [(i, as_float(row.get(metric))) for i, row in enumerate(rows)]
    values = [(i, v) for i, v in values if math.isfinite(v)]
    if not values:
        return {}

    raw = np.asarray([v for _i, v in values], dtype=np.float64)
    lo = float(raw.min())
    hi = float(raw.max())
    denom = hi - lo

    scores: dict[int, float] = {}
    for i, value in values:
        if denom <= 1e-12:
            scores[i] = 0.5
        elif direction == "lower":
            scores[i] = float((hi - value) / denom)
        elif direction == "higher":
            scores[i] = float((value - lo) / denom)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    return scores


def weighted_mean(parts: list[tuple[float, float]]) -> float:
    parts = [(value, weight) for value, weight in parts if math.isfinite(value) and weight > 0]
    if not parts:
        return math.nan
    total_weight = sum(weight for _value, weight in parts)
    return sum(value * weight for value, weight in parts) / total_weight


def add_baseline_deltas(candidates: list[dict[str, Any]], baseline: dict[str, Any] | None):
    if baseline is None:
        return

    for metric in LOWER_BETTER | HIGHER_BETTER | PRESERVATION_METRICS:
        baseline_value = as_float(baseline.get(metric))
        if not math.isfinite(baseline_value):
            continue

        for row in candidates:
            value = as_float(row.get(metric))
            if not math.isfinite(value):
                continue

            if metric in HIGHER_BETTER:
                improvement = value - baseline_value
            else:
                improvement = baseline_value - value

            row[f"{metric}_improvement_vs_source"] = improvement


def add_scores(
    candidates: list[dict[str, Any]],
    realism_weight: float,
    preservation_weight: float,
    lpips_target: float | None,
    max_lpips: float | None,
):
    metric_scores: dict[str, dict[int, float]] = {}

    for metric in available_metrics(candidates, LOWER_BETTER):
        metric_scores[metric] = normalized_scores(candidates, metric, "lower")

    for metric in available_metrics(candidates, HIGHER_BETTER):
        metric_scores[metric] = normalized_scores(candidates, metric, "higher")

    if "source_lpips_mean" in available_metrics(candidates, PRESERVATION_METRICS):
        if lpips_target is None:
            metric_scores["source_lpips_mean"] = normalized_scores(
                candidates,
                "source_lpips_mean",
                "lower",
            )
        else:
            distances = []
            for row in candidates:
                value = as_float(row.get("source_lpips_mean"))
                row["source_lpips_distance_to_target"] = (
                    abs(value - lpips_target) if math.isfinite(value) else math.nan
                )
                distances.append(row)
            metric_scores["source_lpips_mean"] = normalized_scores(
                distances,
                "source_lpips_distance_to_target",
                "lower",
            )

    for i, row in enumerate(candidates):
        lpips_value = as_float(row.get("source_lpips_mean"))
        row["lpips_eligible"] = (
            max_lpips is None
            or not math.isfinite(lpips_value)
            or lpips_value <= max_lpips
        )
        if max_lpips is not None:
            row["max_lpips"] = max_lpips

        for metric, scores in metric_scores.items():
            if i in scores:
                row[f"score_{metric}"] = scores[i]

        realism_parts = [
            (as_float(row.get(f"score_{metric}")), DEFAULT_REALISM_WEIGHTS[metric])
            for metric in DEFAULT_REALISM_WEIGHTS
            if f"score_{metric}" in row
        ]
        row["realism_score"] = weighted_mean(realism_parts)

        lpips_score = as_float(row.get("score_source_lpips_mean"))
        row["preservation_score"] = lpips_score if math.isfinite(lpips_score) else math.nan

        if math.isfinite(row["realism_score"]) and math.isfinite(row["preservation_score"]):
            row["selection_score"] = weighted_mean(
                [
                    (row["realism_score"], realism_weight),
                    (row["preservation_score"], preservation_weight),
                ]
            )
        elif math.isfinite(row["realism_score"]):
            row["selection_score"] = row["realism_score"]
        else:
            row["selection_score"] = math.nan

        if not row["lpips_eligible"]:
            row["selection_score_raw"] = row["selection_score"]
            row["selection_score"] = math.nan


def row_label(row: dict[str, Any]) -> str:
    ckpt = str(row.get("checkpoint", ""))
    cfg = as_float(row.get("cfg_weight"))
    strength = as_float(row.get("noise_strength"))
    return f"{Path(ckpt).stem} | s={strength:g} | cfg={cfg:g}"


def sort_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def desc(value: Any) -> float:
        value = as_float(value)
        return -value if math.isfinite(value) else math.inf

    def asc(value: Any) -> float:
        value = as_float(value)
        return value if math.isfinite(value) else math.inf

    return sorted(
        candidates,
        key=lambda row: (
            desc(row.get("selection_score")),
            desc(row.get("realism_score")),
            asc(row.get("source_lpips_mean")),
        ),
    )


def write_rows(rows: list[dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    ranked: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
    out_path: Path,
    top_k: int,
):
    lines = []
    lines.append("Translate Sweep Metric Analysis")
    lines.append("=" * 32)
    lines.append("")
    lines.append(f"Candidates: {len(ranked)}")
    lines.append("")

    if baseline is not None:
        lines.append("Source baseline:")
        for metric in [
            "real_fid",
            "real_clip_frechet",
            "real_clip_mmd_rbf",
            "real_clip_centroid_cosine",
            "real_clip_nearest_cosine_mean",
        ]:
            value = as_float(baseline.get(metric))
            if math.isfinite(value):
                lines.append(f"  {metric}: {value:.6g}")
        lines.append("")

    lines.append(f"Top {min(top_k, len(ranked))} candidates by selection_score:")
    for rank, row in enumerate(ranked[:top_k], start=1):
        lines.append(
            f"{rank:02d}. {row_label(row)} | "
            f"selection={as_float(row.get('selection_score')):.4f}, "
            f"realism={as_float(row.get('realism_score')):.4f}, "
            f"lpips={as_float(row.get('source_lpips_mean')):.4f}, "
            f"fid={as_float(row.get('real_fid')):.4f}, "
            f"clip_frechet={as_float(row.get('real_clip_frechet')):.4f}, "
            f"clip_mmd={as_float(row.get('real_clip_mmd_rbf')):.4f}"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("- Higher selection_score, realism_score, and preservation_score are better.")
    lines.append("- Lower real_fid, real_clip_frechet, real_clip_mmd_rbf, and source_lpips_mean are better by default.")
    lines.append("- If lpips_target is set, preservation_score favors LPIPS near the target.")
    lines.append("- If max_lpips is set, candidates above it keep selection_score_raw but cannot win selection_score.")
    lines.append("- Treat the ranking as a shortlist generator; verify top candidates visually and downstream.")

    out_path.write_text("\n".join(lines) + "\n")


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it in the environment with "
            "`pip install matplotlib`, or use the CSV/text outputs only."
        ) from exc


def finite_metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    return [as_float(row.get(metric)) for row in rows if finite(row.get(metric))]


def metric_is_higher_better(metric: str) -> bool:
    return metric in HIGHER_BETTER or metric.endswith("_score")


def plot_top_bars(ranked: list[dict[str, Any]], out_dir: Path, top_k: int):
    plt = require_matplotlib()
    rows = [row for row in ranked[:top_k] if finite(row.get("selection_score"))]
    if not rows:
        return

    labels = [row_label(row) for row in rows][::-1]
    values = [as_float(row.get("selection_score")) for row in rows][::-1]

    height = max(5.0, 0.4 * len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(labels, values, color="#2563eb")
    ax.set_xlabel("Selection score (higher is better)")
    ax.set_title(f"Top {len(rows)} Translation Candidates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "top_candidates_selection_score.png", dpi=180)
    plt.close(fig)


def plot_realism_vs_lpips(ranked: list[dict[str, Any]], out_dir: Path, top_k: int):
    plt = require_matplotlib()
    rows = [
        row
        for row in ranked
        if finite(row.get("realism_score")) and finite(row.get("source_lpips_mean"))
    ]
    if not rows:
        return

    x = [as_float(row.get("source_lpips_mean")) for row in rows]
    y = [as_float(row.get("realism_score")) for row in rows]
    score = [as_float(row.get("selection_score")) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(x, y, c=score, cmap="viridis", s=50, alpha=0.8)
    ax.set_xlabel("Source LPIPS mean (lower preserves source more)")
    ax.set_ylabel("Realism score (higher is better)")
    ax.set_title("Realism vs Source Preservation")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Selection score")

    for row in ranked[:top_k]:
        if finite(row.get("realism_score")) and finite(row.get("source_lpips_mean")):
            ax.annotate(
                Path(str(row.get("checkpoint"))).stem,
                (as_float(row.get("source_lpips_mean")), as_float(row.get("realism_score"))),
                fontsize=8,
                alpha=0.8,
            )

    fig.tight_layout()
    fig.savefig(out_dir / "realism_vs_lpips.png", dpi=180)
    plt.close(fig)


def plot_metric_correlations(candidates: list[dict[str, Any]], out_dir: Path):
    plt = require_matplotlib()
    metrics = [
        metric
        for metric in [
            "selection_score",
            "realism_score",
            "preservation_score",
            "real_fid",
            "real_clip_frechet",
            "real_clip_mmd_rbf",
            "real_clip_centroid_cosine",
            "real_clip_nearest_cosine_mean",
            "source_lpips_mean",
        ]
        if len(finite_metric_values(candidates, metric)) >= 2
    ]
    if len(metrics) < 2:
        return

    matrix = np.full((len(candidates), len(metrics)), np.nan, dtype=np.float64)
    for j, metric in enumerate(metrics):
        for i, row in enumerate(candidates):
            matrix[i, j] = as_float(row.get(metric))

    valid_cols = ~np.all(~np.isfinite(matrix), axis=0)
    matrix = matrix[:, valid_cols]
    metrics = [metric for metric, keep in zip(metrics, valid_cols) if keep]
    if len(metrics) < 2:
        return

    col_means = np.nanmean(matrix, axis=0)
    inds = np.where(~np.isfinite(matrix))
    matrix[inds] = np.take(col_means, inds[1])
    corr = np.corrcoef(matrix, rowvar=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(metrics)), metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(metrics)), metrics)
    ax.set_title("Metric Correlation Matrix")
    fig.colorbar(im, ax=ax, label="Pearson r")

    for i in range(len(metrics)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "metric_correlations.png", dpi=180)
    plt.close(fig)


def plot_best_by_checkpoint(candidates: list[dict[str, Any]], out_dir: Path):
    plt = require_matplotlib()
    rows = [row for row in candidates if finite(row.get("checkpoint_step")) and finite(row.get("selection_score"))]
    if not rows:
        return

    best_by_step: dict[float, dict[str, Any]] = {}
    for row in rows:
        step = as_float(row.get("checkpoint_step"))
        if step not in best_by_step or as_float(row.get("selection_score")) > as_float(best_by_step[step].get("selection_score")):
            best_by_step[step] = row

    steps = sorted(best_by_step)
    scores = [as_float(best_by_step[step].get("selection_score")) for step in steps]
    realism = [as_float(best_by_step[step].get("realism_score")) for step in steps]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, scores, marker="o", label="selection_score")
    ax.plot(steps, realism, marker="s", label="realism_score")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Best score at checkpoint")
    ax.set_title("Best Candidate Per Checkpoint")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "best_by_checkpoint.png", dpi=180)
    plt.close(fig)


def metric_grid(
    candidates: list[dict[str, Any]],
    checkpoint: str,
    metric: str,
    strengths: list[float],
    cfgs: list[float],
) -> np.ndarray:
    grid = np.full((len(strengths), len(cfgs)), np.nan, dtype=np.float64)
    for row in candidates:
        if str(row.get("checkpoint")) != checkpoint:
            continue
        strength = as_float(row.get("noise_strength"))
        cfg = as_float(row.get("cfg_weight"))
        value = as_float(row.get(metric))
        if not math.isfinite(value):
            continue
        if strength in strengths and cfg in cfgs:
            grid[strengths.index(strength), cfgs.index(cfg)] = value
    return grid


def plot_heatmaps(candidates: list[dict[str, Any]], metrics: list[str], out_dir: Path):
    plt = require_matplotlib()
    checkpoints = sorted(
        {str(row.get("checkpoint")) for row in candidates if str(row.get("checkpoint", ""))},
        key=lambda x: checkpoint_step(x),
    )
    strengths = sorted({as_float(row.get("noise_strength")) for row in candidates if finite(row.get("noise_strength"))})
    cfgs = sorted({as_float(row.get("cfg_weight")) for row in candidates if finite(row.get("cfg_weight"))})

    if not checkpoints or not strengths or not cfgs:
        return

    for metric in metrics:
        if not any(finite(row.get(metric)) for row in candidates):
            continue

        n = len(checkpoints)
        cols = min(3, n)
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)

        values = finite_metric_values(candidates, metric)
        vmin = min(values)
        vmax = max(values)

        for ax in axes.flat:
            ax.axis("off")

        for i, checkpoint in enumerate(checkpoints):
            ax = axes.flat[i]
            ax.axis("on")
            grid = metric_grid(candidates, checkpoint, metric, strengths, cfgs)
            im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_title(Path(checkpoint).stem)
            ax.set_xticks(range(len(cfgs)), [f"{v:g}" for v in cfgs])
            ax.set_yticks(range(len(strengths)), [f"{v:g}" for v in strengths])
            ax.set_xlabel("CFG")
            ax.set_ylabel("Strength")

            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    value = grid[y, x]
                    if math.isfinite(value):
                        ax.text(x, y, f"{value:.3g}", ha="center", va="center", fontsize=7, color="white")

        direction = "higher is better" if metric_is_higher_better(metric) else "lower is better"
        fig.suptitle(f"{metric} ({direction})", y=0.995)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    # -------------------------
    # Edit these values for each sweep analysis.
    # -------------------------
    model_names = ["all_real_ft_invar_ddim-50"]

    for model_name in model_names:
        
        sweep_dir = Path("/develop/code/eval/cyclenet/remote_sensing/translate_sweep") / model_name
        metrics_path = sweep_dir / "metrics.csv"

        top_k = 15

        weights = [(0.5, 0.5), (0.7, 0.3), (0.75, 0.25), (0.85, 0.15), (0.90, 0.10)]
        lpips_vals = [(0.20, 0.30), (None, None)]

        for realism_weight, preservation_weight in weights:
            for lpips_target, max_lpips in lpips_vals:

                run_name = f"real-{realism_weight:.2f}_pres-{preservation_weight:.2f}"

                if lpips_target and max_lpips:
                    lpips_stats = f"_lpips-[{lpips_target:.2f},{max_lpips:.2f}]"
                elif lpips_target:
                    lpips_stats = f"_lpips-{lpips_target:.2f}"
                elif max_lpips:
                    lpips_stats = f"_lpips-max-{max_lpips:.2f}"
                else:
                    lpips_stats = ""

                out_dir: Path | None = sweep_dir / "analysis" / (run_name + lpips_stats)

                make_plots = True
                heatmap_metrics = [
                    "selection_score",
                    "realism_score",
                    "real_fid",
                    "real_clip_frechet",
                    "real_clip_mmd_rbf",
                    "source_lpips_mean",
                ]

                out_dir = out_dir if out_dir is not None else metrics_path.parent / "analysis"
                plots_dir = out_dir / "plots"
                out_dir.mkdir(parents=True, exist_ok=True)
                plots_dir.mkdir(parents=True, exist_ok=True)

                rows = read_metrics(metrics_path)
                baseline = find_baseline(rows)
                candidates = [row for row in rows if is_candidate(row)]
                if not candidates:
                    raise RuntimeError(f"No translated candidate rows found in {metrics_path}.")

                add_baseline_deltas(candidates, baseline)
                add_scores(
                    candidates,
                    realism_weight=realism_weight,
                    preservation_weight=preservation_weight,
                    lpips_target=lpips_target,
                    max_lpips=max_lpips,
                )
                ranked = sort_candidates(candidates)

                write_rows(ranked, out_dir / "ranked_candidates.csv")
                if baseline is not None:
                    write_rows([baseline], out_dir / "source_baseline.csv")
                write_summary(ranked, baseline, out_dir / "summary.txt", top_k=top_k)

                if make_plots:
                    plot_top_bars(ranked, plots_dir, top_k)
                    plot_realism_vs_lpips(ranked, plots_dir, top_k)
                    plot_best_by_checkpoint(candidates, plots_dir)
                    plot_metric_correlations(candidates, plots_dir)
                    plot_heatmaps(candidates, heatmap_metrics, plots_dir)

                print(f"Wrote analysis to {out_dir}")
                print(f"Best candidate: {row_label(ranked[0])}")
                print(f"Summary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
