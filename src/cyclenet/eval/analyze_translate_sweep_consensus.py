import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from cyclenet.eval.analyze_translate_sweep import (
    LOWER_BETTER,
    HIGHER_BETTER,
    PRESERVATION_METRICS,
    add_baseline_deltas,
    as_float,
    checkpoint_step,
    find_baseline,
    is_candidate,
    read_metrics,
)


RAW_METRIC_COLUMNS = [
    "real_fid",
    "real_clip_frechet",
    "real_clip_mmd_rbf",
    "real_clip_centroid_cosine",
    "real_clip_centroid_l2",
    "real_clip_nearest_cosine_mean",
    "source_lpips_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate translate-sweep analysis runs across multiple models and "
            "report robust candidate settings."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Root translate_sweep directory containing model subdirs such as "
            "all_real/, oem_only/, and oem_only_seg/."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all_real", "oem_only", "oem_only_seg"],
        help="Model subdirectories to scan under --root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <root>/consensus_analysis.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to highlight in summary.txt.",
    )
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Limit the shared-candidate table to settings present in every requested model.",
    )
    return parser.parse_args()


def write_rows(rows: list[dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def candidate_key(row: dict[str, Any]) -> tuple[str, float, float]:
    checkpoint = str(row.get("checkpoint", "")).strip()
    cfg = as_float(row.get("cfg_weight"))
    strength = as_float(row.get("noise_strength"))
    return checkpoint, cfg, strength


def row_label(row: dict[str, Any]) -> str:
    checkpoint, cfg, strength = candidate_key(row)
    return f"{Path(checkpoint).stem} | cfg={cfg:g} | strength={strength:g}"


def key_label(key: tuple[str, float, float]) -> str:
    checkpoint, cfg, strength = key
    return f"{Path(checkpoint).stem} | cfg={cfg:g} | strength={strength:g}"


def parse_analysis_name(name: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {"analysis_name": name}

    match = re.search(r"real-([0-9.]+)_pres-([0-9.]+)", name)
    if match:
        parsed["realism_weight"] = float(match.group(1))
        parsed["preservation_weight"] = float(match.group(2))

    match = re.search(r"lpips-\[([0-9.]+),([0-9.]+)\]", name)
    if match:
        parsed["lpips_target"] = float(match.group(1))
        parsed["max_lpips"] = float(match.group(2))
    else:
        match = re.search(r"lpips-([0-9.]+)", name)
        if match:
            parsed["lpips_target"] = float(match.group(1))

    return parsed


def finite(value: Any) -> bool:
    return math.isfinite(as_float(value))


def rank_score(rank: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return 1.0 - ((rank - 1) / (total - 1))


def find_metrics_path(model_dir: Path) -> Path | None:
    direct = model_dir / "metrics.csv"
    if direct.exists():
        return direct

    candidates = sorted(model_dir.glob("metrics*.csv"))
    if candidates:
        return candidates[0]
    return None


def load_model_metric_map(model_dir: Path) -> tuple[Path | None, dict[tuple[str, float, float], dict[str, Any]], dict[str, Any] | None]:
    metrics_path = find_metrics_path(model_dir)
    if metrics_path is None:
        return None, {}, None

    rows = read_metrics(metrics_path)
    baseline = find_baseline(rows)
    candidates = [row for row in rows if is_candidate(row)]
    add_baseline_deltas(candidates, baseline)

    metric_map: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in candidates:
        metric_map[candidate_key(row)] = row

    return metrics_path, metric_map, baseline


def read_ranked_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    parsed = []
    for idx, row in enumerate(rows, start=1):
        row = dict(row)
        row["rank"] = idx
        row["cfg_weight"] = as_float(row.get("cfg_weight"))
        row["noise_strength"] = as_float(row.get("noise_strength"))
        row["selection_score"] = as_float(row.get("selection_score"))
        row["selection_score_raw"] = as_float(row.get("selection_score_raw"))
        row["realism_score"] = as_float(row.get("realism_score"))
        row["preservation_score"] = as_float(row.get("preservation_score"))
        parsed.append(row)
    return parsed


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = sorted(int(r["rank"]) for r in records)
    rank_scores = [float(r["rank_score"]) for r in records]
    selection_scores = [as_float(r.get("selection_score")) for r in records if finite(r.get("selection_score"))]
    realism_scores = [as_float(r.get("realism_score")) for r in records if finite(r.get("realism_score"))]
    preservation_scores = [as_float(r.get("preservation_score")) for r in records if finite(r.get("preservation_score"))]

    return {
        "analysis_count_seen": len(records),
        "win_count": sum(1 for r in records if int(r["rank"]) == 1),
        "top3_count": sum(1 for r in records if int(r["rank"]) <= 3),
        "top5_count": sum(1 for r in records if int(r["rank"]) <= 5),
        "best_rank": min(ranks),
        "worst_rank": max(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": median(ranks),
        "mean_rank_score": sum(rank_scores) / len(rank_scores),
        "median_rank_score": median(rank_scores),
        "mean_selection_score": mean(selection_scores),
        "mean_realism_score": mean(realism_scores),
        "mean_preservation_score": mean(preservation_scores),
    }


def mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def enrich_with_metrics(row: dict[str, Any], metric_row: dict[str, Any] | None):
    if metric_row is None:
        return

    for metric in RAW_METRIC_COLUMNS:
        if metric in metric_row:
            row[metric] = metric_row[metric]

    for metric in LOWER_BETTER | HIGHER_BETTER | PRESERVATION_METRICS:
        improvement_key = f"{metric}_improvement_vs_source"
        if improvement_key in metric_row:
            row[improvement_key] = metric_row[improvement_key]


def sort_per_model_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("analysis_count_seen", 0)),
            -int(row.get("win_count", 0)),
            -int(row.get("top3_count", 0)),
            -as_float(row.get("mean_rank_score")),
            as_float(row.get("mean_rank")),
        ),
    )


def sort_shared_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -int(row.get("model_count", 0)),
            -as_float(row.get("min_model_mean_rank_score")),
            -as_float(row.get("mean_model_mean_rank_score")),
            -int(row.get("total_win_count", 0)),
            as_float(row.get("max_model_mean_rank")),
        ),
    )


def main():
    args = parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Translate sweep root does not exist: {root}")

    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else (root / "consensus_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_models = list(args.models)
    analysis_index_rows: list[dict[str, Any]] = []
    per_model_rows: list[dict[str, Any]] = []
    shared_rows: list[dict[str, Any]] = []
    summary_lines: list[str] = []

    per_model_aggregate: dict[str, dict[tuple[str, float, float], list[dict[str, Any]]]] = {}
    per_model_metrics: dict[str, dict[tuple[str, float, float], dict[str, Any]]] = {}
    per_model_baselines: dict[str, dict[str, Any] | None] = {}
    per_model_analysis_counts: dict[str, int] = {}
    metrics_paths: dict[str, Path | None] = {}

    for model in requested_models:
        model_dir = root / model
        if not model_dir.exists():
            raise RuntimeError(f"Model directory does not exist: {model_dir}")

        metrics_path, metric_map, baseline = load_model_metric_map(model_dir)
        metrics_paths[model] = metrics_path
        per_model_metrics[model] = metric_map
        per_model_baselines[model] = baseline

        analysis_root = model_dir / "analysis"
        if not analysis_root.exists():
            raise RuntimeError(f"Analysis directory does not exist: {analysis_root}")

        ranked_paths = sorted(analysis_root.glob("*/ranked_candidates.csv"))
        if not ranked_paths:
            raise RuntimeError(f"No ranked_candidates.csv files found under {analysis_root}")

        aggregate: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
        per_model_analysis_counts[model] = len(ranked_paths)

        for ranked_path in ranked_paths:
            analysis_dir = ranked_path.parent
            analysis_meta = parse_analysis_name(analysis_dir.name)
            ranked = read_ranked_candidates(ranked_path)
            total = len(ranked)
            if total == 0:
                continue

            best = ranked[0]
            index_row = {
                "model": model,
                "analysis_dir": str(analysis_dir),
                "ranked_candidates_csv": str(ranked_path),
                "num_ranked": total,
                "best_rank": 1,
                "best_candidate": row_label(best),
                "best_checkpoint": best.get("checkpoint", ""),
                "best_checkpoint_step": checkpoint_step(best.get("checkpoint", "")),
                "best_cfg_weight": as_float(best.get("cfg_weight")),
                "best_noise_strength": as_float(best.get("noise_strength")),
                "best_selection_score": as_float(best.get("selection_score")),
                "best_realism_score": as_float(best.get("realism_score")),
                "best_preservation_score": as_float(best.get("preservation_score")),
            }
            index_row.update(analysis_meta)

            best_metric_row = metric_map.get(candidate_key(best))
            enrich_with_metrics(index_row, best_metric_row)
            analysis_index_rows.append(index_row)

            for row in ranked:
                key = candidate_key(row)
                aggregate[key].append(
                    {
                        "analysis_name": analysis_dir.name,
                        "rank": int(row["rank"]),
                        "rank_score": rank_score(int(row["rank"]), total),
                        "selection_score": as_float(row.get("selection_score")),
                        "realism_score": as_float(row.get("realism_score")),
                        "preservation_score": as_float(row.get("preservation_score")),
                    }
                )

        per_model_aggregate[model] = aggregate

        finalized_rows: list[dict[str, Any]] = []
        for key, records in aggregate.items():
            checkpoint, cfg, strength = key
            row = {
                "model": model,
                "candidate": key_label(key),
                "checkpoint": checkpoint,
                "checkpoint_step": checkpoint_step(checkpoint),
                "cfg_weight": cfg,
                "noise_strength": strength,
                "analysis_count_total": per_model_analysis_counts[model],
            }
            row.update(aggregate_rank_records(records))
            row["analysis_coverage"] = row["analysis_count_seen"] / max(row["analysis_count_total"], 1)
            enrich_with_metrics(row, metric_map.get(key))
            finalized_rows.append(row)

        finalized_rows = sort_per_model_rows(finalized_rows)
        per_model_rows.extend(finalized_rows)
        write_rows(finalized_rows, out_dir / f"{model}_consensus.csv")

    shared_map: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in per_model_rows:
        key = candidate_key(row)
        shared_map[key].append(row)

    for key, rows in shared_map.items():
        if args.shared_only and len(rows) != len(requested_models):
            continue

        checkpoint, cfg, strength = key
        out_row = {
            "candidate": key_label(key),
            "checkpoint": checkpoint,
            "checkpoint_step": checkpoint_step(checkpoint),
            "cfg_weight": cfg,
            "noise_strength": strength,
            "model_count": len(rows),
            "model_count_total": len(requested_models),
            "models_present": ",".join(sorted(str(row["model"]) for row in rows)),
            "total_win_count": sum(int(row.get("win_count", 0)) for row in rows),
            "total_top3_count": sum(int(row.get("top3_count", 0)) for row in rows),
            "total_top5_count": sum(int(row.get("top5_count", 0)) for row in rows),
            "mean_model_mean_rank": mean([as_float(row.get("mean_rank")) for row in rows if finite(row.get("mean_rank"))]),
            "median_model_mean_rank": median([as_float(row.get("mean_rank")) for row in rows if finite(row.get("mean_rank"))]),
            "max_model_mean_rank": max(as_float(row.get("mean_rank")) for row in rows if finite(row.get("mean_rank"))),
            "mean_model_mean_rank_score": mean([as_float(row.get("mean_rank_score")) for row in rows if finite(row.get("mean_rank_score"))]),
            "min_model_mean_rank_score": min(as_float(row.get("mean_rank_score")) for row in rows if finite(row.get("mean_rank_score"))),
        }

        for metric in RAW_METRIC_COLUMNS:
            values = []
            for row in rows:
                model = str(row["model"])
                value = as_float(row.get(metric))
                out_row[f"{model}_{metric}"] = value
                if math.isfinite(value):
                    values.append(value)
            out_row[f"mean_{metric}"] = mean(values)

        for metric in ["real_fid", "real_clip_frechet", "real_clip_mmd_rbf", "source_lpips_mean"]:
            improvement_key = f"{metric}_improvement_vs_source"
            values = []
            for row in rows:
                model = str(row["model"])
                value = as_float(row.get(improvement_key))
                out_row[f"{model}_{improvement_key}"] = value
                if math.isfinite(value):
                    values.append(value)
            out_row[f"mean_{improvement_key}"] = mean(values)

        shared_rows.append(out_row)

    shared_rows = sort_shared_rows(shared_rows)

    write_rows(analysis_index_rows, out_dir / "analysis_index.csv")
    write_rows(per_model_rows, out_dir / "all_models_consensus.csv")
    write_rows(shared_rows, out_dir / "shared_candidates_consensus.csv")

    summary_lines.append("Translate Sweep Consensus Analysis")
    summary_lines.append("=" * 34)
    summary_lines.append("")
    summary_lines.append(f"Root: {root}")
    summary_lines.append(f"Models: {', '.join(requested_models)}")
    summary_lines.append("")

    summary_lines.append("Per-model scan:")
    for model in requested_models:
        metrics_path = metrics_paths.get(model)
        summary_lines.append(
            f"- {model}: analyses={per_model_analysis_counts.get(model, 0)}, "
            f"metrics={metrics_path if metrics_path is not None else 'missing'}"
        )
    summary_lines.append("")

    for model in requested_models:
        model_rows = [row for row in per_model_rows if str(row.get("model")) == model]
        summary_lines.append(f"Top {min(args.top_k, len(model_rows))} robust candidates for {model}:")
        for idx, row in enumerate(model_rows[: args.top_k], start=1):
            summary_lines.append(
                f"{idx:02d}. {row['candidate']} | wins={int(row['win_count'])}/{int(row['analysis_count_total'])}, "
                f"top3={int(row['top3_count'])}, mean_rank={as_float(row.get('mean_rank')):.2f}, "
                f"mean_rank_score={as_float(row.get('mean_rank_score')):.3f}, "
                f"clip_frechet={as_float(row.get('real_clip_frechet')):.4f}, "
                f"clip_mmd={as_float(row.get('real_clip_mmd_rbf')):.4f}, "
                f"lpips={as_float(row.get('source_lpips_mean')):.4f}"
            )
        summary_lines.append("")

    eligible_shared = [
        row for row in shared_rows
        if int(row.get("model_count", 0)) == len(requested_models)
    ]
    if eligible_shared:
        summary_lines.append(
            f"Top {min(args.top_k, len(eligible_shared))} shared candidates present in all models:"
        )
        for idx, row in enumerate(eligible_shared[: args.top_k], start=1):
            summary_lines.append(
                f"{idx:02d}. {row['candidate']} | min_model_rank_score={as_float(row.get('min_model_mean_rank_score')):.3f}, "
                f"mean_model_rank_score={as_float(row.get('mean_model_mean_rank_score')):.3f}, "
                f"max_model_mean_rank={as_float(row.get('max_model_mean_rank')):.2f}, "
                f"mean_clip_frechet={as_float(row.get('mean_real_clip_frechet')):.4f}, "
                f"mean_clip_mmd={as_float(row.get('mean_real_clip_mmd_rbf')):.4f}, "
                f"mean_lpips={as_float(row.get('mean_source_lpips_mean')):.4f}"
            )
        summary_lines.append("")

    summary_lines.append("Notes:")
    summary_lines.append("- Per-model consensus ranks settings by robustness across analysis weightings.")
    summary_lines.append("- Shared candidate ranks are the main table for choosing one common setup across models.")
    summary_lines.append("- Prefer candidates with strong worst-model rank behavior, not only strong average rank.")
    summary_lines.append("- Use raw realism metrics and LPIPS from the joined metrics.csv values to break close ties.")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    print(f"Wrote consensus analysis to {out_dir}")
    print(f"Summary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
