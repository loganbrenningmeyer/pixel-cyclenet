from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cyclenet.eval.frechet_dist import frechet_distance
from cyclenet.eval.plotting.set_style import MODEL_NAMES
from cyclenet.eval.thesis_plots.scripts.cache_class_features import (
    class_feature_cache_filename,
    load_class_embeddings,
    load_selected_models,
    normalize_feature_extractor,
)


CLASS_INFO = {
    1: ("Bareland", "bareland"),
    2: ("Rangeland", "rangeland"),
    3: ("Developed Space", "developed_space"),
    4: ("Road", "road"),
    5: ("Trees", "trees"),
    6: ("Water", "water"),
    7: ("Agriculture land", "agriculture_land"),
    8: ("Buildings", "buildings"),
}


def label_name(label_id: int) -> str:
    return CLASS_INFO.get(label_id, (f"Unknown({label_id})", f"unknown_{label_id}"))[0]


def label_slug(label_id: int) -> str:
    return CLASS_INFO.get(label_id, (f"Unknown({label_id})", f"unknown_{label_id}"))[1]


def display_model_name(model_name: str) -> str:
    display_name = MODEL_NAMES.get(model_name, "")
    return display_name if display_name else model_name.replace("_", " ")


def load_cached_dataset_features(
    cache_root: Path,
    dataset_rel_path: str,
    feature_extractor: str,
) -> dict[int, np.ndarray]:
    cache_path = cache_root / dataset_rel_path / class_feature_cache_filename(feature_extractor)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached class feature bundle: {cache_path}")
    return load_class_embeddings(cache_path)


def available_label_ids(feats_by_dataset: dict[str, dict[int, np.ndarray]]) -> list[int]:
    return sorted({label_id for dataset_feats in feats_by_dataset.values() for label_id in dataset_feats})


def compute_reference_fd_rows(
    feats_by_dataset: dict[str, dict[int, np.ndarray]],
    reference_dataset: str,
    comparison_datasets: list[str],
) -> list[dict[str, object]]:
    if reference_dataset not in feats_by_dataset:
        raise KeyError(f"reference_dataset '{reference_dataset}' not found in cached datasets")

    rows: list[dict[str, object]] = []
    ref_feats_by_class = feats_by_dataset[reference_dataset]
    label_ids = available_label_ids(feats_by_dataset)

    for comparison_dataset in comparison_datasets:
        if comparison_dataset not in feats_by_dataset:
            raise KeyError(f"comparison dataset '{comparison_dataset}' not found in cached datasets")

        cmp_feats_by_class = feats_by_dataset[comparison_dataset]
        for label_id in label_ids:
            real_feats = ref_feats_by_class.get(label_id, np.empty((0, 0), dtype=np.float32))
            cmp_feats = cmp_feats_by_class.get(label_id, np.empty((0, 0), dtype=np.float32))

            row: dict[str, object] = {
                "reference_dataset": reference_dataset,
                "comparison_dataset": comparison_dataset,
                "label_id": int(label_id),
                "label_name": label_name(label_id),
                "label_slug": label_slug(label_id),
                "n_reference": int(real_feats.shape[0]) if real_feats.ndim == 2 else 0,
                "n_comparison": int(cmp_feats.shape[0]) if cmp_feats.ndim == 2 else 0,
                "feature_dim_reference": (
                    int(real_feats.shape[1]) if real_feats.ndim == 2 and real_feats.size else 0
                ),
                "feature_dim_comparison": (
                    int(cmp_feats.shape[1]) if cmp_feats.ndim == 2 and cmp_feats.size else 0
                ),
                "frechet_distance": float("nan"),
            }

            if real_feats.ndim != 2 or cmp_feats.ndim != 2 or len(real_feats) == 0 or len(cmp_feats) == 0:
                rows.append(row)
                continue

            row["frechet_distance"] = frechet_distance(cmp_feats, real_feats)
            rows.append(row)

    return rows


def build_delta_vs_sim_rows(
    fd_rows: list[dict[str, object]],
    reference_dataset: str,
    baseline_dataset: str,
) -> list[dict[str, object]]:
    baseline_by_label = {
        int(row["label_id"]): row
        for row in fd_rows
        if row["reference_dataset"] == reference_dataset and row["comparison_dataset"] == baseline_dataset
    }
    comparison_datasets = sorted(
        {
            str(row["comparison_dataset"])
            for row in fd_rows
            if row["reference_dataset"] == reference_dataset and row["comparison_dataset"] != baseline_dataset
        }
    )

    rows: list[dict[str, object]] = []
    for comparison_dataset in comparison_datasets:
        for row in fd_rows:
            if row["reference_dataset"] != reference_dataset or row["comparison_dataset"] != comparison_dataset:
                continue

            label_id = int(row["label_id"])
            baseline = baseline_by_label.get(label_id)
            baseline_fd = float(baseline["frechet_distance"]) if baseline is not None else float("nan")
            comparison_fd = float(row["frechet_distance"])
            delta_fd = float("nan")
            improvement_fd = float("nan")
            if np.isfinite(baseline_fd) and np.isfinite(comparison_fd):
                delta_fd = comparison_fd - baseline_fd
                improvement_fd = baseline_fd - comparison_fd

            rows.append(
                {
                    "reference_dataset": reference_dataset,
                    "baseline_dataset": baseline_dataset,
                    "comparison_dataset": comparison_dataset,
                    "label_id": label_id,
                    "label_name": row["label_name"],
                    "label_slug": row["label_slug"],
                    "frechet_distance_baseline": baseline_fd,
                    "frechet_distance_comparison": comparison_fd,
                    "frechet_distance_delta_vs_baseline": delta_fd,
                    "frechet_improvement_vs_baseline": improvement_fd,
                }
            )

    return rows


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def cache_class_feature_frechet_distances(
    selected_models_csv: str | Path,
    cache_dir: str | Path,
    out_dir: str | Path | None = None,
    feature_extractor: str = "deeplab",
) -> Path:
    selected_df = load_selected_models(selected_models_csv)
    cache_dir = Path(cache_dir)
    feature_extractor = normalize_feature_extractor(feature_extractor)
    if out_dir is None:
        out_dir = cache_dir / (
            "analysis" if feature_extractor == "deeplab" else f"analysis_{feature_extractor}"
        )
    out_dir = Path(out_dir)

    feats_by_dataset: dict[str, dict[int, np.ndarray]] = {
        "real": load_cached_dataset_features(cache_dir, "real", feature_extractor),
        "sim": load_cached_dataset_features(cache_dir, "sim", feature_extractor),
    }

    model_names = [str(model_name) for model_name in selected_df["model_name"].tolist()]
    for model_name in model_names:
        feats_by_dataset[model_name] = load_cached_dataset_features(
            cache_dir / model_name,
            "translated",
            feature_extractor,
        )

    comparison_datasets = ["sim"] + model_names
    fd_rows = compute_reference_fd_rows(
        feats_by_dataset=feats_by_dataset,
        reference_dataset="real",
        comparison_datasets=comparison_datasets,
    )
    delta_rows = build_delta_vs_sim_rows(
        fd_rows=fd_rows,
        reference_dataset="real",
        baseline_dataset="sim",
    )

    fd_df = pd.DataFrame(fd_rows)
    if not fd_df.empty:
        fd_df["comparison_display_name"] = fd_df["comparison_dataset"].map(
            lambda name: "Sim" if name == "sim" else display_model_name(str(name))
        )
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df["comparison_display_name"] = delta_df["comparison_dataset"].map(
            lambda name: display_model_name(str(name))
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "selected_models_csv": str(Path(selected_models_csv).resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "feature_extractor": feature_extractor,
        "class_feature_cache_filename": class_feature_cache_filename(feature_extractor),
        "reference_dataset": "real",
        "baseline_dataset": "sim",
        "comparison_datasets": comparison_datasets,
    }
    write_json(metadata, out_dir / "metadata.json")
    write_csv(fd_df.to_dict(orient="records"), out_dir / "frechet_distance_to_real_by_class.csv")
    write_csv(delta_df.to_dict(orient="records"), out_dir / "frechet_distance_delta_vs_sim_by_class.csv")

    sim_rows = fd_df.loc[fd_df["comparison_dataset"] == "sim"].to_dict(orient="records")
    if sim_rows:
        write_csv(sim_rows, out_dir / "sim" / "frechet_distance_to_real_by_class.csv")

    for model_name in model_names:
        model_rows = fd_df.loc[fd_df["comparison_dataset"] == model_name].to_dict(orient="records")
        if model_rows:
            write_csv(
                model_rows,
                out_dir / model_name / "frechet_distance_to_real_by_class.csv",
            )
        model_delta_rows = delta_df.loc[delta_df["comparison_dataset"] == model_name].to_dict(orient="records")
        if model_delta_rows:
            write_csv(
                model_delta_rows,
                out_dir / model_name / "frechet_distance_delta_vs_sim_by_class.csv",
            )

    print(f"Saved {feature_extractor} per-class Fréchet distance caches to {out_dir}")
    return out_dir


def main() -> None:
    # CSV listing the selected models whose translated class-feature caches should be analyzed.
    selected_models_csv = "/develop/code/eval/thesis/selected_models.csv"
    # Root cache directory produced by `cache_class_features.py`.
    cache_dir = "/develop/code/eval/thesis/class_feature_cache"
    # Feature extractor cache to analyze. Supported values: `deeplab` and `fid`.
    feature_extractor = "deeplab"
    # Output directory for aggregated and per-model Fréchet-distance CSV caches.
    # Use `None` for the extractor default: `analysis` for DeepLab, `analysis_fid` for FID.
    out_dir = None

    saved_dir = cache_class_feature_frechet_distances(
        selected_models_csv=selected_models_csv,
        cache_dir=cache_dir,
        feature_extractor=feature_extractor,
        out_dir=out_dir,
    )
    print(f"Saved class-feature distance analysis to {saved_dir}")


if __name__ == "__main__":
    main()
