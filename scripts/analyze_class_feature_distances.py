#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cyclenet.eval.embed import DeepLabEmbedder
from cyclenet.eval.plotting.heatmap import plot_heatmap
from cyclenet.eval.plotting.project import (
    plot_proj_density,
    plot_proj_density_marginal,
    plot_proj_scatter,
)
from cyclenet.eval.project import UmapProjector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
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
PLOT_COLORS = {
    "sim": "#6b7280",
    "real": "#dc2626",
    "translated": "#2563eb",
}


def load_config(config_path: str | Path) -> DictConfig:
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str | Path) -> None:
    OmegaConf.save(config, save_path)


def cfg_select(config: DictConfig, key: str, default=None):
    value = OmegaConf.select(config, key)
    return default if value is None else value


def as_parent_dir_set(value) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(v) for v in value}


def label_name(label_id: int) -> str:
    return CLASS_INFO.get(label_id, (f"Unknown({label_id})", f"unknown_{label_id}"))[0]


def label_slug(label_id: int) -> str:
    return CLASS_INFO.get(label_id, (f"Unknown({label_id})", f"unknown_{label_id}"))[1]


def collect_rgb_paths(root: Path, rgb_parent_dirs: set[str] | None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    rgb_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and (rgb_parent_dirs is None or path.parent.name in rgb_parent_dirs)
    ]
    if not rgb_paths:
        if rgb_parent_dirs is None:
            raise ValueError(f"No RGB files found under {root}")
        raise ValueError(f"No RGB files found under {root} with parent dirs {sorted(rgb_parent_dirs)}")
    return rgb_paths


def resolve_matching_file(parent: Path, stem: str) -> Path:
    matches = [
        path
        for path in sorted(parent.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.stem == stem
    ]
    if not matches:
        raise FileNotFoundError(f"No file with stem '{stem}' found under {parent}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files with stem '{stem}' found under {parent}: {matches}")
    return matches[0]


def derive_label_path(
    rgb_path: Path,
    image_root: Path,
    label_root: Path,
    rgb_parent_dirs: set[str] | None,
    label_parent_dir: str | None,
) -> Path:
    try:
        rel_path = rgb_path.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(f"RGB path is outside image_root: {rgb_path}") from exc

    if rgb_parent_dirs is not None and rgb_path.parent.name not in rgb_parent_dirs:
        raise ValueError(
            f"Expected RGB file under one of {sorted(rgb_parent_dirs)}, got '{rgb_path.parent.name}': {rgb_path}"
        )

    rel_parts = list(rel_path.parts)
    if label_parent_dir is not None and len(rel_parts) >= 2:
        rel_parts[-2] = label_parent_dir
    label_path = label_root.joinpath(*rel_parts)
    if label_path.exists():
        return label_path.resolve()

    if label_parent_dir is None:
        flat_label_path = label_root / rgb_path.name
        if flat_label_path.exists():
            return flat_label_path.resolve()

    label_parent = label_path.parent
    if label_parent.exists():
        return resolve_matching_file(label_parent, rgb_path.stem).resolve()

    raise FileNotFoundError(f"Missing label for {rgb_path}: expected {label_path}")


def collect_rgb_label_pairs(
    image_root: Path,
    label_root: Path,
    rgb_parent_dirs: set[str] | None,
    label_parent_dir: str | None,
) -> list[tuple[Path, Path]]:
    pairs = []
    for rgb_path in collect_rgb_paths(image_root, rgb_parent_dirs):
        label_path = derive_label_path(
            rgb_path=rgb_path,
            image_root=image_root,
            label_root=label_root,
            rgb_parent_dirs=rgb_parent_dirs,
            label_parent_dir=label_parent_dir,
        )
        pairs.append((rgb_path.resolve(), label_path))
    return pairs


def sample_pairs(
    pairs: list[tuple[Path, Path]],
    max_count: int | None,
    seed: int,
) -> list[tuple[Path, Path]]:
    if max_count is None:
        return pairs
    if max_count <= 0:
        raise ValueError(f"max_count must be positive, got {max_count}")
    if len(pairs) <= max_count:
        return pairs

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pairs), size=max_count, replace=False))
    return [pairs[i] for i in idx]


def load_or_create_pair_manifest(
    image_root: Path,
    label_root: Path,
    rgb_parent_dirs: set[str] | None,
    label_parent_dir: str | None,
    manifest_path: Path,
    max_count: int | None,
    seed: int,
) -> list[tuple[Path, Path]]:
    if manifest_path.exists():
        rows = list(csv.DictReader(manifest_path.open("r", newline="")))
        pairs = [(Path(row["img_path"]), Path(row["label_path"])) for row in rows]
        missing = [pair for pair in pairs if not pair[0].exists() or not pair[1].exists()]
        if missing:
            first = missing[0]
            raise FileNotFoundError(
                f"Cached manifest contains missing files: image={first[0]}, label={first[1]}"
            )
        return pairs

    pairs = collect_rgb_label_pairs(
        image_root=image_root,
        label_root=label_root,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
    )
    pairs = sample_pairs(pairs, max_count=max_count, seed=seed)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["img_path", "label_path"])
        writer.writeheader()
        for img_path, label_path in pairs:
            writer.writerow({"img_path": str(img_path), "label_path": str(label_path)})

    return pairs


def write_metadata(metadata: dict[str, object], metadata_path: Path) -> None:
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing != metadata:
            raise ValueError(
                "Cache metadata does not match the current script configuration. "
                f"Either use a different cache directory or remove the stale cache at {metadata_path}."
            )
        return

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_class_embeddings(cache_path: Path) -> dict[int, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached class feature bundle: {cache_path}")

    bundle = np.load(cache_path)
    return {
        int(name.removeprefix("class_")): np.asarray(bundle[name], dtype=np.float32)
        for name in sorted(bundle.files)
        if name.startswith("class_")
    }


def load_or_compute_class_embeddings(
    embedder: DeepLabEmbedder,
    pairs: list[tuple[Path, Path]],
    batch_size: int,
    cache_path: Path,
) -> dict[int, np.ndarray]:
    if cache_path.exists():
        return load_class_embeddings(cache_path)

    return embedder.embed_by_class(
        img_paths=[str(img_path) for img_path, _ in pairs],
        label_paths=[str(label_path) for _, label_path in pairs],
        batch_size=batch_size,
        save_path=cache_path,
    )


def summarize_embeddings(dataset_name: str, feats_by_class: dict[int, np.ndarray]) -> None:
    print(f"\n[{dataset_name}] class feature counts")
    print("class_id\tcount\tfeature_dim")
    for class_id in sorted(feats_by_class):
        feats = feats_by_class[class_id]
        print(f"{class_id}\t{len(feats)}\t{feats.shape[1] if feats.ndim == 2 else 0}")


def covariance_matrix(feats: np.ndarray) -> np.ndarray:
    if feats.shape[0] < 2:
        return np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float64)
    return np.cov(feats, rowvar=False).astype(np.float64)


def symmetric_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) * 0.5)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(fake_feats: np.ndarray, real_feats: np.ndarray) -> float:
    fake = fake_feats.astype(np.float64, copy=False)
    real = real_feats.astype(np.float64, copy=False)

    mu_fake = fake.mean(axis=0)
    mu_real = real.mean(axis=0)
    cov_fake = covariance_matrix(fake)
    cov_real = covariance_matrix(real)

    cov_real_sqrt = symmetric_matrix_sqrt(cov_real)
    middle = cov_real_sqrt @ cov_fake @ cov_real_sqrt
    cov_mean = symmetric_matrix_sqrt(middle)

    diff = mu_fake - mu_real
    dist = diff @ diff + np.trace(cov_fake + cov_real - 2.0 * cov_mean)
    return float(max(dist, 0.0))


def centroid_stats(fake_feats: np.ndarray, real_feats: np.ndarray) -> dict[str, float]:
    fake_centroid = fake_feats.mean(axis=0).astype(np.float64, copy=False)
    real_centroid = real_feats.mean(axis=0).astype(np.float64, copy=False)

    fake_norm = float(np.linalg.norm(fake_centroid))
    real_norm = float(np.linalg.norm(real_centroid))
    cosine_similarity = (
        float(fake_centroid @ real_centroid / (fake_norm * real_norm))
        if fake_norm > 0.0 and real_norm > 0.0
        else float("nan")
    )

    return {
        "centroid_cosine_similarity": cosine_similarity,
        "centroid_cosine_distance": float(1.0 - cosine_similarity) if np.isfinite(cosine_similarity) else float("nan"),
        "centroid_l2": float(np.linalg.norm(fake_centroid - real_centroid)),
    }


def spread_stats(fake_feats: np.ndarray, real_feats: np.ndarray) -> dict[str, float]:
    cov_fake = covariance_matrix(fake_feats)
    cov_real = covariance_matrix(real_feats)
    trace_fake = float(np.trace(cov_fake))
    trace_real = float(np.trace(cov_real))
    return {
        "trace_fake": trace_fake,
        "trace_real": trace_real,
        "trace_ratio_fake_over_real": trace_fake / trace_real if trace_real > 0.0 else float("nan"),
    }


def available_label_ids(feats_by_dataset: dict[str, dict[int, np.ndarray]]) -> list[int]:
    return sorted({label_id for dataset_feats in feats_by_dataset.values() for label_id in dataset_feats})


def compute_pairwise_rows(
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
            fake_feats = cmp_feats_by_class.get(label_id, np.empty((0, 0), dtype=np.float32))

            row: dict[str, object] = {
                "reference_dataset": reference_dataset,
                "comparison_dataset": comparison_dataset,
                "label_id": label_id,
                "label_name": label_name(label_id),
                "n_reference": int(real_feats.shape[0]) if real_feats.ndim == 2 else 0,
                "n_comparison": int(fake_feats.shape[0]) if fake_feats.ndim == 2 else 0,
                "feature_dim_reference": int(real_feats.shape[1]) if real_feats.ndim == 2 and real_feats.size else 0,
                "feature_dim_comparison": int(fake_feats.shape[1]) if fake_feats.ndim == 2 and fake_feats.size else 0,
                "frechet_distance": float("nan"),
                "centroid_cosine_similarity": float("nan"),
                "centroid_cosine_distance": float("nan"),
                "centroid_l2": float("nan"),
                "trace_fake": float("nan"),
                "trace_real": float("nan"),
                "trace_ratio_fake_over_real": float("nan"),
            }

            if real_feats.ndim != 2 or fake_feats.ndim != 2 or len(real_feats) == 0 or len(fake_feats) == 0:
                rows.append(row)
                continue

            row["frechet_distance"] = frechet_distance(fake_feats, real_feats)
            row.update(centroid_stats(fake_feats, real_feats))
            row.update(spread_stats(fake_feats, real_feats))
            rows.append(row)

    return rows


def compute_cross_class_centroid_rows(
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
        for comparison_label_id in label_ids:
            fake_feats = cmp_feats_by_class.get(comparison_label_id, np.empty((0, 0), dtype=np.float32))
            for reference_label_id in label_ids:
                real_feats = ref_feats_by_class.get(reference_label_id, np.empty((0, 0), dtype=np.float32))

                row: dict[str, object] = {
                    "reference_dataset": reference_dataset,
                    "comparison_dataset": comparison_dataset,
                    "comparison_label_id": comparison_label_id,
                    "comparison_label_name": label_name(comparison_label_id),
                    "reference_label_id": reference_label_id,
                    "reference_label_name": label_name(reference_label_id),
                    "n_comparison": int(fake_feats.shape[0]) if fake_feats.ndim == 2 else 0,
                    "n_reference": int(real_feats.shape[0]) if real_feats.ndim == 2 else 0,
                    "feature_dim_comparison": int(fake_feats.shape[1]) if fake_feats.ndim == 2 and fake_feats.size else 0,
                    "feature_dim_reference": int(real_feats.shape[1]) if real_feats.ndim == 2 and real_feats.size else 0,
                    "centroid_cosine_similarity": float("nan"),
                    "centroid_cosine_distance": float("nan"),
                    "centroid_l2": float("nan"),
                }

                if real_feats.ndim != 2 or fake_feats.ndim != 2 or len(real_feats) == 0 or len(fake_feats) == 0:
                    rows.append(row)
                    continue

                row.update(centroid_stats(fake_feats, real_feats))
                rows.append(row)

    return rows


def build_cross_class_delta_rows(
    cross_class_rows: list[dict[str, object]],
    reference_dataset: str,
    baseline_dataset: str,
) -> list[dict[str, object]]:
    baseline_by_pair = {
        (int(row["comparison_label_id"]), int(row["reference_label_id"])): row
        for row in cross_class_rows
        if row["reference_dataset"] == reference_dataset and row["comparison_dataset"] == baseline_dataset
    }
    comparison_datasets = sorted(
        {
            str(row["comparison_dataset"])
            for row in cross_class_rows
            if row["reference_dataset"] == reference_dataset and row["comparison_dataset"] != baseline_dataset
        }
    )

    rows: list[dict[str, object]] = []
    for comparison_dataset in comparison_datasets:
        for row in cross_class_rows:
            if row["reference_dataset"] != reference_dataset or row["comparison_dataset"] != comparison_dataset:
                continue

            key = (int(row["comparison_label_id"]), int(row["reference_label_id"]))
            baseline = baseline_by_pair.get(key)

            out_row: dict[str, object] = {
                "reference_dataset": reference_dataset,
                "baseline_dataset": baseline_dataset,
                "comparison_dataset": comparison_dataset,
                "comparison_label_id": int(row["comparison_label_id"]),
                "comparison_label_name": row["comparison_label_name"],
                "reference_label_id": int(row["reference_label_id"]),
                "reference_label_name": row["reference_label_name"],
                "centroid_l2_baseline": float(baseline["centroid_l2"]) if baseline is not None else float("nan"),
                "centroid_l2_comparison": float(row["centroid_l2"]),
                "centroid_l2_delta_vs_baseline": float("nan"),
                "centroid_cosine_distance_baseline": (
                    float(baseline["centroid_cosine_distance"]) if baseline is not None else float("nan")
                ),
                "centroid_cosine_distance_comparison": float(row["centroid_cosine_distance"]),
                "centroid_cosine_distance_delta_vs_baseline": float("nan"),
                "centroid_cosine_similarity_baseline": (
                    float(baseline["centroid_cosine_similarity"]) if baseline is not None else float("nan")
                ),
                "centroid_cosine_similarity_comparison": float(row["centroid_cosine_similarity"]),
                "centroid_cosine_similarity_delta_vs_baseline": float("nan"),
            }

            if baseline is not None:
                baseline_l2 = float(baseline["centroid_l2"])
                comparison_l2 = float(row["centroid_l2"])
                if np.isfinite(baseline_l2) and np.isfinite(comparison_l2):
                    out_row["centroid_l2_delta_vs_baseline"] = comparison_l2 - baseline_l2

                baseline_cos_distance = float(baseline["centroid_cosine_distance"])
                comparison_cos_distance = float(row["centroid_cosine_distance"])
                if np.isfinite(baseline_cos_distance) and np.isfinite(comparison_cos_distance):
                    out_row["centroid_cosine_distance_delta_vs_baseline"] = (
                        comparison_cos_distance - baseline_cos_distance
                    )

                baseline_cos_similarity = float(baseline["centroid_cosine_similarity"])
                comparison_cos_similarity = float(row["centroid_cosine_similarity"])
                if np.isfinite(baseline_cos_similarity) and np.isfinite(comparison_cos_similarity):
                    out_row["centroid_cosine_similarity_delta_vs_baseline"] = (
                        comparison_cos_similarity - baseline_cos_similarity
                    )

            rows.append(out_row)

    return rows


def build_improvement_rows(
    pairwise_rows: list[dict[str, object]],
    reference_dataset: str,
    baseline_dataset: str,
) -> list[dict[str, object]]:
    baseline_by_label = {
        int(row["label_id"]): row
        for row in pairwise_rows
        if row["reference_dataset"] == reference_dataset and row["comparison_dataset"] == baseline_dataset
    }
    comparison_datasets = sorted(
        {
            str(row["comparison_dataset"])
            for row in pairwise_rows
            if row["reference_dataset"] == reference_dataset and row["comparison_dataset"] != baseline_dataset
        }
    )

    rows: list[dict[str, object]] = []
    for comparison_dataset in comparison_datasets:
        for row in pairwise_rows:
            if row["reference_dataset"] != reference_dataset or row["comparison_dataset"] != comparison_dataset:
                continue

            label_id = int(row["label_id"])
            baseline = baseline_by_label.get(label_id)
            out_row = {
                "reference_dataset": reference_dataset,
                "baseline_dataset": baseline_dataset,
                "comparison_dataset": comparison_dataset,
                "label_id": label_id,
                "label_name": row["label_name"],
                "frechet_distance_baseline": float(baseline["frechet_distance"]) if baseline is not None else float("nan"),
                "frechet_distance_comparison": float(row["frechet_distance"]),
                "frechet_improvement_vs_baseline": float("nan"),
                "centroid_l2_baseline": float(baseline["centroid_l2"]) if baseline is not None else float("nan"),
                "centroid_l2_comparison": float(row["centroid_l2"]),
                "centroid_l2_improvement_vs_baseline": float("nan"),
                "centroid_cosine_similarity_baseline": (
                    float(baseline["centroid_cosine_similarity"]) if baseline is not None else float("nan")
                ),
                "centroid_cosine_similarity_comparison": float(row["centroid_cosine_similarity"]),
                "centroid_cosine_similarity_gain_vs_baseline": float("nan"),
            }

            if baseline is not None:
                baseline_frechet = float(baseline["frechet_distance"])
                comparison_frechet = float(row["frechet_distance"])
                if np.isfinite(baseline_frechet) and np.isfinite(comparison_frechet):
                    out_row["frechet_improvement_vs_baseline"] = baseline_frechet - comparison_frechet

                baseline_l2 = float(baseline["centroid_l2"])
                comparison_l2 = float(row["centroid_l2"])
                if np.isfinite(baseline_l2) and np.isfinite(comparison_l2):
                    out_row["centroid_l2_improvement_vs_baseline"] = baseline_l2 - comparison_l2

                baseline_cos = float(baseline["centroid_cosine_similarity"])
                comparison_cos = float(row["centroid_cosine_similarity"])
                if np.isfinite(baseline_cos) and np.isfinite(comparison_cos):
                    out_row["centroid_cosine_similarity_gain_vs_baseline"] = comparison_cos - baseline_cos

            rows.append(out_row)

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


def build_cross_class_grid(
    rows: list[dict[str, object]],
    comparison_dataset: str,
    reference_dataset: str,
    metric: str,
) -> pd.DataFrame:
    filtered_rows = [
        row
        for row in rows
        if row["comparison_dataset"] == comparison_dataset and row["reference_dataset"] == reference_dataset
    ]
    label_ids = sorted(
        {
            int(row["comparison_label_id"])
            for row in filtered_rows
        }
        | {
            int(row["reference_label_id"])
            for row in filtered_rows
        }
    )
    if not label_ids:
        return pd.DataFrame()

    grid = np.full((len(label_ids), len(label_ids)), np.nan, dtype=np.float64)
    row_index = {label_id: idx for idx, label_id in enumerate(label_ids)}
    col_index = {label_id: idx for idx, label_id in enumerate(label_ids)}

    for row in filtered_rows:
        cmp_label_id = int(row["comparison_label_id"])
        ref_label_id = int(row["reference_label_id"])
        grid[row_index[cmp_label_id], col_index[ref_label_id]] = float(row.get(metric, np.nan))

    row_labels = [label_name(label_id) for label_id in label_ids]
    col_labels = [label_name(label_id) for label_id in label_ids]
    return pd.DataFrame(grid, index=row_labels, columns=col_labels)


def write_cross_class_grids(
    rows: list[dict[str, object]],
    reference_dataset: str,
    comparison_datasets: list[str],
    metrics: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for comparison_dataset in comparison_datasets:
        for metric in metrics:
            grid_df = build_cross_class_grid(
                rows=rows,
                comparison_dataset=comparison_dataset,
                reference_dataset=reference_dataset,
                metric=metric,
            )
            if grid_df.empty:
                continue
            grid_df.to_csv(out_dir / f"{comparison_dataset}_to_{reference_dataset}_{metric}.csv")


def plot_cross_class_heatmaps(
    rows: list[dict[str, object]],
    reference_dataset: str,
    comparison_datasets: list[str],
    metrics: list[tuple[str, str, str, str]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, cmap, cbar_label, filename_suffix in metrics:
        metric_values = [
            float(row[metric])
            for row in rows
            if row["reference_dataset"] == reference_dataset and np.isfinite(float(row.get(metric, np.nan)))
        ]
        if not metric_values:
            continue
        vmin = float(min(metric_values))
        vmax = float(max(metric_values))

        for comparison_dataset in comparison_datasets:
            grid_df = build_cross_class_grid(
                rows=rows,
                comparison_dataset=comparison_dataset,
                reference_dataset=reference_dataset,
                metric=metric,
            )
            if grid_df.empty:
                continue

            plot_heatmap(
                grid_df=grid_df,
                title=f"{comparison_dataset.capitalize()} to {reference_dataset.capitalize()} {cbar_label}",
                xlabel=f"Reference class ({reference_dataset})",
                ylabel=f"Comparison class ({comparison_dataset})",
                save_path=out_dir / f"{comparison_dataset}_to_{reference_dataset}_{filename_suffix}.png",
                cmap=cmap,
                annot=True,
                fmt=".2f",
                vmin=vmin,
                vmax=vmax,
                cbar_label=cbar_label,
                square=True,
            )


def build_cross_class_delta_grid(
    rows: list[dict[str, object]],
    comparison_dataset: str,
    reference_dataset: str,
    metric: str,
) -> pd.DataFrame:
    filtered_rows = [
        row
        for row in rows
        if row["comparison_dataset"] == comparison_dataset and row["reference_dataset"] == reference_dataset
    ]
    label_ids = sorted(
        {
            int(row["comparison_label_id"])
            for row in filtered_rows
        }
        | {
            int(row["reference_label_id"])
            for row in filtered_rows
        }
    )
    if not label_ids:
        return pd.DataFrame()

    grid = np.full((len(label_ids), len(label_ids)), np.nan, dtype=np.float64)
    row_index = {label_id: idx for idx, label_id in enumerate(label_ids)}
    col_index = {label_id: idx for idx, label_id in enumerate(label_ids)}

    for row in filtered_rows:
        cmp_label_id = int(row["comparison_label_id"])
        ref_label_id = int(row["reference_label_id"])
        grid[row_index[cmp_label_id], col_index[ref_label_id]] = float(row.get(metric, np.nan))

    row_labels = [label_name(label_id) for label_id in label_ids]
    col_labels = [label_name(label_id) for label_id in label_ids]
    return pd.DataFrame(grid, index=row_labels, columns=col_labels)


def write_cross_class_delta_grids(
    rows: list[dict[str, object]],
    reference_dataset: str,
    comparison_datasets: list[str],
    baseline_dataset: str,
    metrics: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for comparison_dataset in comparison_datasets:
        for metric in metrics:
            grid_df = build_cross_class_delta_grid(
                rows=rows,
                comparison_dataset=comparison_dataset,
                reference_dataset=reference_dataset,
                metric=metric,
            )
            if grid_df.empty:
                continue
            grid_df.to_csv(
                out_dir / f"{comparison_dataset}_vs_{baseline_dataset}_to_{reference_dataset}_{metric}.csv"
            )


def plot_cross_class_delta_heatmaps(
    rows: list[dict[str, object]],
    reference_dataset: str,
    comparison_datasets: list[str],
    baseline_dataset: str,
    metrics: list[tuple[str, str, str, str]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, cmap, cbar_label, filename_suffix in metrics:
        metric_values = [
            float(row[metric])
            for row in rows
            if row["reference_dataset"] == reference_dataset and np.isfinite(float(row.get(metric, np.nan)))
        ]
        if not metric_values:
            continue
        vmax = float(max(abs(min(metric_values)), abs(max(metric_values))))
        vmin = -vmax

        for comparison_dataset in comparison_datasets:
            grid_df = build_cross_class_delta_grid(
                rows=rows,
                comparison_dataset=comparison_dataset,
                reference_dataset=reference_dataset,
                metric=metric,
            )
            if grid_df.empty:
                continue

            plot_heatmap(
                grid_df=grid_df,
                title=(
                    f"{comparison_dataset.capitalize()} vs {baseline_dataset.capitalize()} "
                    f"to {reference_dataset.capitalize()} {cbar_label}"
                ),
                xlabel=f"Reference class ({reference_dataset})",
                ylabel=f"Comparison class ({comparison_dataset})",
                save_path=out_dir / f"{comparison_dataset}_vs_{baseline_dataset}_{filename_suffix}.png",
                cmap=cmap,
                annot=True,
                fmt=".2f",
                vmin=vmin,
                vmax=vmax,
                center=0.0,
                cbar_label=cbar_label,
                square=True,
            )


def rank_reference_rows(
    rows: list[dict[str, object]],
    metric: str,
    descending: bool = False,
) -> list[dict[str, object]]:
    valid_rows = [row for row in rows if np.isfinite(float(row.get(metric, np.nan)))]
    return sorted(valid_rows, key=lambda row: float(row[metric]), reverse=descending)


def metric_for_reference_label(
    row_by_reference_label: dict[int, dict[str, object]],
    reference_label_id: int,
    metric: str,
) -> float:
    row = row_by_reference_label.get(reference_label_id)
    if row is None:
        return float("nan")
    return float(row.get(metric, np.nan))


def compute_class_alignment_summary_rows(
    cross_class_rows: list[dict[str, object]],
    reference_dataset: str,
    comparison_datasets: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped_rows: dict[tuple[str, int], list[dict[str, object]]] = {}

    for row in cross_class_rows:
        if row["reference_dataset"] != reference_dataset:
            continue
        comparison_dataset = str(row["comparison_dataset"])
        if comparison_dataset not in comparison_datasets:
            continue
        key = (comparison_dataset, int(row["comparison_label_id"]))
        grouped_rows.setdefault(key, []).append(row)

    for comparison_dataset in comparison_datasets:
        label_ids = sorted(
            label_id
            for dataset_name, label_id in grouped_rows
            if dataset_name == comparison_dataset
        )
        for comparison_label_id in label_ids:
            key = (comparison_dataset, comparison_label_id)
            class_rows = grouped_rows.get(key, [])
            if not class_rows:
                continue

            row_by_reference_label = {
                int(row["reference_label_id"]): row
                for row in class_rows
            }
            l2_ranked = rank_reference_rows(class_rows, metric="centroid_l2", descending=False)
            cosine_distance_ranked = rank_reference_rows(
                class_rows, metric="centroid_cosine_distance", descending=False
            )
            l2_rank_lookup = {
                int(row["reference_label_id"]): rank
                for rank, row in enumerate(l2_ranked, start=1)
            }
            cosine_distance_rank_lookup = {
                int(row["reference_label_id"]): rank
                for rank, row in enumerate(cosine_distance_ranked, start=1)
            }

            nearest_l2 = l2_ranked[0] if l2_ranked else None
            nearest_cosine_distance = cosine_distance_ranked[0] if cosine_distance_ranked else None

            own_l2 = metric_for_reference_label(
                row_by_reference_label=row_by_reference_label,
                reference_label_id=comparison_label_id,
                metric="centroid_l2",
            )
            own_cosine_distance = metric_for_reference_label(
                row_by_reference_label=row_by_reference_label,
                reference_label_id=comparison_label_id,
                metric="centroid_cosine_distance",
            )
            buildings_l2 = metric_for_reference_label(
                row_by_reference_label=row_by_reference_label,
                reference_label_id=8,
                metric="centroid_l2",
            )
            buildings_cosine_distance = metric_for_reference_label(
                row_by_reference_label=row_by_reference_label,
                reference_label_id=8,
                metric="centroid_cosine_distance",
            )
            water_l2 = metric_for_reference_label(
                row_by_reference_label=row_by_reference_label,
                reference_label_id=6,
                metric="centroid_l2",
            )
            water_cosine_distance = metric_for_reference_label(
                row_by_reference_label=row_by_reference_label,
                reference_label_id=6,
                metric="centroid_cosine_distance",
            )

            row: dict[str, object] = {
                "reference_dataset": reference_dataset,
                "comparison_dataset": comparison_dataset,
                "comparison_label_id": comparison_label_id,
                "comparison_label_name": label_name(comparison_label_id),
                "nearest_real_class_id_by_centroid_l2": (
                    int(nearest_l2["reference_label_id"]) if nearest_l2 is not None else -1
                ),
                "nearest_real_class_name_by_centroid_l2": (
                    str(nearest_l2["reference_label_name"]) if nearest_l2 is not None else ""
                ),
                "nearest_real_class_id_by_centroid_cosine_distance": (
                    int(nearest_cosine_distance["reference_label_id"])
                    if nearest_cosine_distance is not None
                    else -1
                ),
                "nearest_real_class_name_by_centroid_cosine_distance": (
                    str(nearest_cosine_distance["reference_label_name"])
                    if nearest_cosine_distance is not None
                    else ""
                ),
                "own_real_centroid_l2": own_l2,
                "buildings_real_centroid_l2": buildings_l2,
                "water_real_centroid_l2": water_l2,
                "own_real_centroid_cosine_distance": own_cosine_distance,
                "buildings_real_centroid_cosine_distance": buildings_cosine_distance,
                "water_real_centroid_cosine_distance": water_cosine_distance,
                "own_real_rank_by_centroid_l2": float(l2_rank_lookup.get(comparison_label_id, np.nan)),
                "buildings_real_rank_by_centroid_l2": float(l2_rank_lookup.get(8, np.nan)),
                "water_real_rank_by_centroid_l2": float(l2_rank_lookup.get(6, np.nan)),
                "own_real_rank_by_centroid_cosine_distance": float(
                    cosine_distance_rank_lookup.get(comparison_label_id, np.nan)
                ),
                "buildings_real_rank_by_centroid_cosine_distance": float(
                    cosine_distance_rank_lookup.get(8, np.nan)
                ),
                "water_real_rank_by_centroid_cosine_distance": float(
                    cosine_distance_rank_lookup.get(6, np.nan)
                ),
                "is_own_nearest_by_centroid_l2": int(
                    nearest_l2 is not None and int(nearest_l2["reference_label_id"]) == comparison_label_id
                ),
                "is_own_nearest_by_centroid_cosine_distance": int(
                    nearest_cosine_distance is not None
                    and int(nearest_cosine_distance["reference_label_id"]) == comparison_label_id
                ),
                "own_minus_buildings_centroid_l2": float("nan"),
                "own_minus_water_centroid_l2": float("nan"),
                "own_minus_buildings_centroid_cosine_distance": float("nan"),
                "own_minus_water_centroid_cosine_distance": float("nan"),
            }

            if np.isfinite(own_l2) and np.isfinite(buildings_l2):
                row["own_minus_buildings_centroid_l2"] = own_l2 - buildings_l2
            if np.isfinite(own_l2) and np.isfinite(water_l2):
                row["own_minus_water_centroid_l2"] = own_l2 - water_l2
            if np.isfinite(own_cosine_distance) and np.isfinite(buildings_cosine_distance):
                row["own_minus_buildings_centroid_cosine_distance"] = (
                    own_cosine_distance - buildings_cosine_distance
                )
            if np.isfinite(own_cosine_distance) and np.isfinite(water_cosine_distance):
                row["own_minus_water_centroid_cosine_distance"] = own_cosine_distance - water_cosine_distance

            rows.append(row)

    return rows


def build_class_alignment_delta_rows(
    summary_rows: list[dict[str, object]],
    reference_dataset: str,
    baseline_dataset: str,
) -> list[dict[str, object]]:
    baseline_by_label = {
        int(row["comparison_label_id"]): row
        for row in summary_rows
        if row["reference_dataset"] == reference_dataset and row["comparison_dataset"] == baseline_dataset
    }
    comparison_datasets = sorted(
        {
            str(row["comparison_dataset"])
            for row in summary_rows
            if row["reference_dataset"] == reference_dataset and row["comparison_dataset"] != baseline_dataset
        }
    )

    delta_metric_names = [
        "own_real_centroid_l2",
        "buildings_real_centroid_l2",
        "water_real_centroid_l2",
        "own_real_rank_by_centroid_l2",
        "buildings_real_rank_by_centroid_l2",
        "water_real_rank_by_centroid_l2",
        "own_minus_buildings_centroid_l2",
        "own_minus_water_centroid_l2",
        "own_real_centroid_cosine_distance",
        "buildings_real_centroid_cosine_distance",
        "water_real_centroid_cosine_distance",
        "own_real_rank_by_centroid_cosine_distance",
        "buildings_real_rank_by_centroid_cosine_distance",
        "water_real_rank_by_centroid_cosine_distance",
        "own_minus_buildings_centroid_cosine_distance",
        "own_minus_water_centroid_cosine_distance",
    ]

    rows: list[dict[str, object]] = []
    for comparison_dataset in comparison_datasets:
        for row in summary_rows:
            if row["reference_dataset"] != reference_dataset or row["comparison_dataset"] != comparison_dataset:
                continue

            comparison_label_id = int(row["comparison_label_id"])
            baseline = baseline_by_label.get(comparison_label_id)
            out_row: dict[str, object] = {
                "reference_dataset": reference_dataset,
                "baseline_dataset": baseline_dataset,
                "comparison_dataset": comparison_dataset,
                "comparison_label_id": comparison_label_id,
                "comparison_label_name": row["comparison_label_name"],
                "nearest_real_class_name_by_centroid_l2_baseline": (
                    str(baseline["nearest_real_class_name_by_centroid_l2"]) if baseline is not None else ""
                ),
                "nearest_real_class_name_by_centroid_l2_comparison": str(
                    row["nearest_real_class_name_by_centroid_l2"]
                ),
                "nearest_real_class_name_by_centroid_cosine_distance_baseline": (
                    str(baseline["nearest_real_class_name_by_centroid_cosine_distance"])
                    if baseline is not None
                    else ""
                ),
                "nearest_real_class_name_by_centroid_cosine_distance_comparison": str(
                    row["nearest_real_class_name_by_centroid_cosine_distance"]
                ),
                "nearest_real_class_changed_by_centroid_l2": int(False),
                "nearest_real_class_changed_by_centroid_cosine_distance": int(False),
            }

            if baseline is not None:
                out_row["nearest_real_class_changed_by_centroid_l2"] = int(
                    str(baseline["nearest_real_class_name_by_centroid_l2"])
                    != str(row["nearest_real_class_name_by_centroid_l2"])
                )
                out_row["nearest_real_class_changed_by_centroid_cosine_distance"] = int(
                    str(baseline["nearest_real_class_name_by_centroid_cosine_distance"])
                    != str(row["nearest_real_class_name_by_centroid_cosine_distance"])
                )

            for metric_name in delta_metric_names:
                baseline_value = float(baseline[metric_name]) if baseline is not None else float("nan")
                comparison_value = float(row[metric_name])
                out_row[f"{metric_name}_baseline"] = baseline_value
                out_row[f"{metric_name}_comparison"] = comparison_value
                out_row[f"{metric_name}_delta_vs_baseline"] = float("nan")
                if np.isfinite(baseline_value) and np.isfinite(comparison_value):
                    out_row[f"{metric_name}_delta_vs_baseline"] = comparison_value - baseline_value

            rows.append(out_row)

    return rows


def plot_alignment_metric_by_class(
    rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    use_zero_line: bool = False,
) -> None:
    comparison_datasets = sorted({str(row["comparison_dataset"]) for row in rows})
    label_ids = sorted({int(row["comparison_label_id"]) for row in rows})
    x = np.arange(len(label_ids), dtype=np.float64)
    width = 0.36 if len(comparison_datasets) <= 2 else 0.8 / max(len(comparison_datasets), 1)

    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    for dataset_index, comparison_dataset in enumerate(comparison_datasets):
        dataset_rows = {
            int(row["comparison_label_id"]): row
            for row in rows
            if row["comparison_dataset"] == comparison_dataset
        }
        values = np.array(
            [float(dataset_rows.get(label_id, {}).get(metric, np.nan)) for label_id in label_ids],
            dtype=np.float64,
        )
        offsets = x + (dataset_index - (len(comparison_datasets) - 1) / 2.0) * width
        ax.bar(
            offsets,
            values,
            width=width,
            label=comparison_dataset.capitalize(),
            color=PLOT_COLORS.get(comparison_dataset, None),
        )

    if use_zero_line:
        ax.axhline(0.0, color="#111111", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x, [label_name(label_id) for label_id in label_ids], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_class_umap_projections(
    feats_by_dataset: dict[str, dict[int, np.ndarray]],
    dataset_names: list[str],
    config: DictConfig,
    out_dir: Path,
) -> None:
    projection_method = str(cfg_select(config, "projection.method", "umap")).lower()
    if projection_method != "umap":
        raise ValueError(
            f"Expected projection.method to be 'umap' in combined analysis config, got '{projection_method}'"
        )

    n_components = int(cfg_select(config, "projection.n_components", 2))
    if n_components != 2:
        raise ValueError(f"Only 2D UMAP plotting is supported here, got n_components={n_components}")

    random_state = int(cfg_select(config, "projection.random_state", 42))
    title_template = str(cfg_select(config, "plotting.title", "2D Embedding Projection"))
    axis_pad_frac = float(cfg_select(config, "plotting.axis_pad_frac", 0.25))

    point_show = bool(cfg_select(config, "plotting.points.show", True))
    point_alpha = float(cfg_select(config, "plotting.points.alpha", 0.3))
    point_size = float(cfg_select(config, "plotting.points.size", 10.0))
    max_points_per_group = cfg_select(config, "plotting.points.max_points_per_group", 500)
    max_points_per_group = int(max_points_per_group) if max_points_per_group is not None else None

    density_show = bool(cfg_select(config, "plotting.density.show", True))
    density_alpha = float(cfg_select(config, "plotting.density.alpha", 0.7))
    density_fill = bool(cfg_select(config, "plotting.density.fill", False))

    marginal_show = bool(cfg_select(config, "plotting.marginal.show", True))

    labels_cfg = OmegaConf.to_container(cfg_select(config, "plotting.labels", {}), resolve=True) or {}
    colors_cfg = OmegaConf.to_container(cfg_select(config, "plotting.colors", {}), resolve=True) or {}

    umap_dir = out_dir / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)

    label_ids = sorted(
        {
            label_id
            for dataset_name in dataset_names
            for label_id, feats in feats_by_dataset[dataset_name].items()
            if feats.ndim == 2 and len(feats) > 0
        }
    )

    for label_id in label_ids:
        coords: list[np.ndarray] = []
        plot_labels: list[str] = []
        plot_colors: list[str] = []
        features_for_fit: list[np.ndarray] = []

        for dataset_name in dataset_names:
            feats = feats_by_dataset[dataset_name].get(label_id)
            if feats is None or feats.ndim != 2 or len(feats) == 0:
                continue
            features_for_fit.append(feats.astype(np.float32, copy=False))

        if len(features_for_fit) < 2:
            continue

        projector = UmapProjector(n_components=n_components, random_state=random_state)
        stacked_feats = np.concatenate(features_for_fit, axis=0)
        stacked_coords = projector.fit(stacked_feats)

        split_sizes = [len(feats) for feats in features_for_fit]
        coord_slices = []
        start = 0
        for size in split_sizes:
            coord_slices.append(stacked_coords[start : start + size])
            start += size

        coord_index = 0
        for dataset_name in dataset_names:
            feats = feats_by_dataset[dataset_name].get(label_id)
            if feats is None or feats.ndim != 2 or len(feats) == 0:
                continue

            coords.append(coord_slices[coord_index])
            plot_labels.append(str(labels_cfg.get(dataset_name, dataset_name.capitalize())))
            plot_colors.append(str(colors_cfg.get(dataset_name, PLOT_COLORS.get(dataset_name, "#333333"))))
            coord_index += 1

        if not coords:
            continue

        all_coords = np.concatenate(coords, axis=0)
        x_min = float(all_coords[:, 0].min())
        x_max = float(all_coords[:, 0].max())
        y_min = float(all_coords[:, 1].min())
        y_max = float(all_coords[:, 1].max())
        x_pad = max((x_max - x_min) * axis_pad_frac, 1e-6)
        y_pad = max((y_max - y_min) * axis_pad_frac, 1e-6)
        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

        class_title = f"{title_template}: {label_name(label_id)}"
        class_stem = f"class_{label_id}_{label_slug(label_id)}"

        if not density_show:
            density_alpha = 0
        if not point_show:
            point_alpha = 0

        if marginal_show:
            plot_proj_density_marginal(
                coords=coords,
                labels=plot_labels,
                colors=plot_colors,
                title=class_title,
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                save_path=umap_dir / f"{class_stem}_marginal.png",
                max_points_per_group=max_points_per_group,
                seed=random_state,
                fill=density_fill,
                alpha=density_alpha,
                point_alpha=point_alpha,
                point_size=point_size,
                show_points=point_show,
                xlim=xlim,
                ylim=ylim,
            )


def plot_metric_by_class(
    pairwise_rows: list[dict[str, object]],
    reference_dataset: str,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    rows = [row for row in pairwise_rows if row["reference_dataset"] == reference_dataset]
    comparison_datasets = sorted({str(row["comparison_dataset"]) for row in rows})
    label_ids = sorted({int(row["label_id"]) for row in rows})
    x = np.arange(len(label_ids), dtype=np.float64)
    width = 0.36 if len(comparison_datasets) <= 2 else 0.8 / max(len(comparison_datasets), 1)

    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    for dataset_index, comparison_dataset in enumerate(comparison_datasets):
        dataset_rows = {
            int(row["label_id"]): row
            for row in rows
            if row["comparison_dataset"] == comparison_dataset
        }
        values = np.array(
            [float(dataset_rows.get(label_id, {}).get(metric, np.nan)) for label_id in label_ids],
            dtype=np.float64,
        )
        offsets = x + (dataset_index - (len(comparison_datasets) - 1) / 2.0) * width
        ax.bar(
            offsets,
            values,
            width=width,
            label=comparison_dataset.capitalize(),
            color=PLOT_COLORS.get(comparison_dataset, None),
        )

    ax.set_xticks(x, [label_name(label_id) for label_id in label_ids], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_summary(pairwise_rows: list[dict[str, object]], improvement_rows: list[dict[str, object]]) -> None:
    print("\n[class feature distances]")
    print("comparison\tlabel\tfrechet\tcentroid_l2\tcentroid_cosine")
    for row in pairwise_rows:
        print(
            f"{row['comparison_dataset']}->{row['reference_dataset']}\t"
            f"{row['label_name']}\t"
            f"{float(row['frechet_distance']):.6f}\t"
            f"{float(row['centroid_l2']):.6f}\t"
            f"{float(row['centroid_cosine_similarity']):.6f}"
        )

    if improvement_rows:
        print("\n[improvement vs baseline]")
        print("comparison\tlabel\tfrechet_improvement\tcentroid_l2_improvement\tcentroid_cosine_gain")
        for row in improvement_rows:
            print(
                f"{row['comparison_dataset']} vs {row['baseline_dataset']}\t"
                f"{row['label_name']}\t"
                f"{float(row['frechet_improvement_vs_baseline']):.6f}\t"
                f"{float(row['centroid_l2_improvement_vs_baseline']):.6f}\t"
                f"{float(row['centroid_cosine_similarity_gain_vs_baseline']):.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    seed = int(cfg_select(config, "run.seed", 42))

    sim_image_root = Path(config.data.sim_image_root)
    sim_label_root = Path(config.data.sim_label_root)
    real_image_root = Path(config.data.real_image_root)
    real_label_root = Path(config.data.real_label_root)

    translated_image_root_value = cfg_select(config, "data.translated_image_root", None)
    translated_image_root = (
        Path(translated_image_root_value) if translated_image_root_value is not None else None
    )
    translated_label_root = Path(config.data.translated_label_root)

    rgb_parent_dirs = as_parent_dir_set(config.data.rgb_parent_dirs)
    label_parent_dir_value = cfg_select(config, "data.label_parent_dir", None)
    label_parent_dir = str(label_parent_dir_value) if label_parent_dir_value is not None else None

    deeplab_ckpt_path = Path(config.embedding.deeplab_ckpt_path)
    feature_layer = str(cfg_select(config, "embedding.feature_layer", "prelogits"))
    num_classes = int(cfg_select(config, "embedding.num_classes", 8))
    batch_size = int(cfg_select(config, "embedding.batch_size", 32))
    max_samples_per_dataset = cfg_select(config, "embedding.max_samples_per_dataset", None)
    if max_samples_per_dataset is not None:
        max_samples_per_dataset = int(max_samples_per_dataset)

    cache_dir = Path(config.data.cache_dir)
    reference_dataset = str(cfg_select(config, "analysis.reference_dataset", "real"))
    comparison_datasets = [
        str(value) for value in cfg_select(config, "analysis.comparison_datasets", ["sim", "translated"])
    ]
    baseline_dataset_value = cfg_select(config, "analysis.baseline_dataset", "sim")
    baseline_dataset = str(baseline_dataset_value) if baseline_dataset_value is not None else None
    out_dir = Path(config.analysis.out_dir)
    create_umap = bool(cfg_select(config, "projection.create_umap", False))

    if not sim_image_root.exists():
        raise FileNotFoundError(f"sim_image_root does not exist: {sim_image_root}")
    if not sim_label_root.exists():
        raise FileNotFoundError(f"sim_label_root does not exist: {sim_label_root}")
    if not real_image_root.exists():
        raise FileNotFoundError(f"real_image_root does not exist: {real_image_root}")
    if not real_label_root.exists():
        raise FileNotFoundError(f"real_label_root does not exist: {real_label_root}")
    if translated_image_root is not None and not translated_image_root.exists():
        raise FileNotFoundError(f"translated_image_root does not exist: {translated_image_root}")
    if translated_image_root is not None and not translated_label_root.exists():
        raise FileNotFoundError(f"translated_label_root does not exist: {translated_label_root}")
    if not deeplab_ckpt_path.exists():
        raise FileNotFoundError(f"deeplab_ckpt_path does not exist: {deeplab_ckpt_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    metadata = {
        "sim_image_root": str(sim_image_root.resolve()),
        "sim_label_root": str(sim_label_root.resolve()),
        "real_image_root": str(real_image_root.resolve()),
        "real_label_root": str(real_label_root.resolve()),
        "translated_image_root": (
            str(translated_image_root.resolve()) if translated_image_root is not None else None
        ),
        "translated_label_root": str(translated_label_root.resolve()),
        "rgb_parent_dirs": sorted(rgb_parent_dirs) if rgb_parent_dirs is not None else None,
        "label_parent_dir": label_parent_dir,
        "deeplab_ckpt_path": str(deeplab_ckpt_path.resolve()),
        "feature_layer": feature_layer,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "max_samples_per_dataset": max_samples_per_dataset,
        "seed": seed,
    }
    write_metadata(metadata, cache_dir / "metadata.json")

    embedder = DeepLabEmbedder(
        ckpt_path=deeplab_ckpt_path,
        num_classes=num_classes,
        feature_layer=feature_layer,
    )

    dataset_specs = [
        ("sim", sim_image_root, sim_label_root),
        ("real", real_image_root, real_label_root),
    ]
    if translated_image_root is not None:
        dataset_specs.append(("translated", translated_image_root, translated_label_root))

    feats_by_dataset: dict[str, dict[int, np.ndarray]] = {}
    for dataset_name, image_root, label_root in dataset_specs:
        dataset_cache_dir = cache_dir / dataset_name
        pairs = load_or_create_pair_manifest(
            image_root=image_root,
            label_root=label_root,
            rgb_parent_dirs=rgb_parent_dirs,
            label_parent_dir=label_parent_dir,
            manifest_path=dataset_cache_dir / "pairs.csv",
            max_count=max_samples_per_dataset,
            seed=seed,
        )
        feats_by_class = load_or_compute_class_embeddings(
            embedder=embedder,
            pairs=pairs,
            batch_size=batch_size,
            cache_path=dataset_cache_dir / "deeplab_class_features.npz",
        )
        summarize_embeddings(dataset_name, feats_by_class)
        feats_by_dataset[dataset_name] = feats_by_class

    dataset_names = sorted(feats_by_dataset)
    if reference_dataset not in feats_by_dataset:
        raise KeyError(f"reference_dataset '{reference_dataset}' not found in computed datasets")
    for comparison_dataset in comparison_datasets:
        if comparison_dataset not in feats_by_dataset:
            raise KeyError(f"comparison dataset '{comparison_dataset}' not found in computed datasets")
    if baseline_dataset is not None and baseline_dataset not in feats_by_dataset:
        raise KeyError(f"baseline dataset '{baseline_dataset}' not found in computed datasets")

    pairwise_comparison_datasets = list(comparison_datasets)
    if baseline_dataset is not None and baseline_dataset not in pairwise_comparison_datasets:
        pairwise_comparison_datasets.append(baseline_dataset)

    pairwise_rows = compute_pairwise_rows(
        feats_by_dataset=feats_by_dataset,
        reference_dataset=reference_dataset,
        comparison_datasets=pairwise_comparison_datasets,
    )
    write_csv(pairwise_rows, out_dir / "pairwise_class_distances.csv")

    cross_class_rows = compute_cross_class_centroid_rows(
        feats_by_dataset=feats_by_dataset,
        reference_dataset=reference_dataset,
        comparison_datasets=pairwise_comparison_datasets,
    )
    write_csv(cross_class_rows, out_dir / "cross_class_centroid_distances.csv")

    class_alignment_rows = compute_class_alignment_summary_rows(
        cross_class_rows=cross_class_rows,
        reference_dataset=reference_dataset,
        comparison_datasets=pairwise_comparison_datasets,
    )
    write_csv(class_alignment_rows, out_dir / "class_alignment_summary.csv")

    improvement_rows: list[dict[str, object]] = []
    if baseline_dataset is not None:
        improvement_rows = build_improvement_rows(
            pairwise_rows=pairwise_rows,
            reference_dataset=reference_dataset,
            baseline_dataset=baseline_dataset,
        )
        if improvement_rows:
            write_csv(improvement_rows, out_dir / "improvement_vs_baseline.csv")

    cross_class_delta_rows: list[dict[str, object]] = []
    if baseline_dataset is not None:
        cross_class_delta_rows = build_cross_class_delta_rows(
            cross_class_rows=cross_class_rows,
            reference_dataset=reference_dataset,
            baseline_dataset=baseline_dataset,
        )
        if cross_class_delta_rows:
            write_csv(cross_class_delta_rows, out_dir / "cross_class_centroid_delta_vs_baseline.csv")

    class_alignment_delta_rows: list[dict[str, object]] = []
    if baseline_dataset is not None:
        class_alignment_delta_rows = build_class_alignment_delta_rows(
            summary_rows=class_alignment_rows,
            reference_dataset=reference_dataset,
            baseline_dataset=baseline_dataset,
        )
        if class_alignment_delta_rows:
            write_csv(class_alignment_delta_rows, out_dir / "class_alignment_delta_vs_baseline.csv")

    plots_dir = out_dir / "plots"
    plot_metric_by_class(
        pairwise_rows=pairwise_rows,
        reference_dataset=reference_dataset,
        metric="frechet_distance",
        ylabel="Fréchet distance",
        title=f"Per-class Fréchet Distance to {reference_dataset.capitalize()}",
        out_path=plots_dir / "frechet_distance_by_class.png",
    )
    plot_metric_by_class(
        pairwise_rows=pairwise_rows,
        reference_dataset=reference_dataset,
        metric="centroid_l2",
        ylabel="Centroid L2 distance",
        title=f"Per-class Centroid L2 Distance to {reference_dataset.capitalize()}",
        out_path=plots_dir / "centroid_l2_by_class.png",
    )
    plot_metric_by_class(
        pairwise_rows=pairwise_rows,
        reference_dataset=reference_dataset,
        metric="centroid_cosine_similarity",
        ylabel="Centroid cosine similarity",
        title=f"Per-class Centroid Cosine Similarity to {reference_dataset.capitalize()}",
        out_path=plots_dir / "centroid_cosine_similarity_by_class.png",
    )

    cross_class_grid_dir = out_dir / "cross_class_centroid_grids"
    write_cross_class_grids(
        rows=cross_class_rows,
        reference_dataset=reference_dataset,
        comparison_datasets=pairwise_comparison_datasets,
        metrics=["centroid_l2", "centroid_cosine_distance"],
        out_dir=cross_class_grid_dir,
    )
    plot_cross_class_heatmaps(
        rows=cross_class_rows,
        reference_dataset=reference_dataset,
        comparison_datasets=pairwise_comparison_datasets,
        metrics=[
            ("centroid_l2", "viridis_r", "Centroid L2 distance", "centroid_l2_heatmap"),
            (
                "centroid_cosine_distance",
                "viridis_r",
                "Centroid cosine distance",
                "centroid_cosine_distance_heatmap",
            ),
        ],
        out_dir=plots_dir / "cross_class_centroid_heatmaps",
    )

    plot_alignment_metric_by_class(
        rows=class_alignment_rows,
        metric="own_minus_buildings_centroid_l2",
        ylabel="Own minus buildings centroid L2",
        title=f"Own-vs-Buildings L2 Margin to {reference_dataset.capitalize()}",
        out_path=plots_dir / "class_alignment" / "own_minus_buildings_centroid_l2.png",
        use_zero_line=True,
    )
    plot_alignment_metric_by_class(
        rows=class_alignment_rows,
        metric="own_minus_water_centroid_l2",
        ylabel="Own minus water centroid L2",
        title=f"Own-vs-Water L2 Margin to {reference_dataset.capitalize()}",
        out_path=plots_dir / "class_alignment" / "own_minus_water_centroid_l2.png",
        use_zero_line=True,
    )
    plot_alignment_metric_by_class(
        rows=class_alignment_rows,
        metric="own_real_rank_by_centroid_l2",
        ylabel="Own real-class rank by centroid L2",
        title=f"Own Real-Class Rank to {reference_dataset.capitalize()}",
        out_path=plots_dir / "class_alignment" / "own_real_rank_by_centroid_l2.png",
    )

    if baseline_dataset is not None and cross_class_delta_rows:
        comparison_datasets_wo_baseline = [
            dataset_name for dataset_name in pairwise_comparison_datasets if dataset_name != baseline_dataset
        ]
        cross_class_delta_grid_dir = out_dir / "cross_class_centroid_delta_grids"
        write_cross_class_delta_grids(
            rows=cross_class_delta_rows,
            reference_dataset=reference_dataset,
            comparison_datasets=comparison_datasets_wo_baseline,
            baseline_dataset=baseline_dataset,
            metrics=["centroid_l2_delta_vs_baseline", "centroid_cosine_distance_delta_vs_baseline"],
            out_dir=cross_class_delta_grid_dir,
        )
        plot_cross_class_delta_heatmaps(
            rows=cross_class_delta_rows,
            reference_dataset=reference_dataset,
            comparison_datasets=comparison_datasets_wo_baseline,
            baseline_dataset=baseline_dataset,
            metrics=[
                (
                    "centroid_l2_delta_vs_baseline",
                    "RdBu_r",
                    "Centroid L2 delta vs baseline",
                    "centroid_l2_delta_vs_baseline_heatmap",
                ),
                (
                    "centroid_cosine_distance_delta_vs_baseline",
                    "RdBu_r",
                    "Centroid cosine distance delta vs baseline",
                    "centroid_cosine_distance_delta_vs_baseline_heatmap",
                ),
            ],
            out_dir=plots_dir / "cross_class_centroid_delta_heatmaps",
        )

    if baseline_dataset is not None and class_alignment_delta_rows:
        plot_alignment_metric_by_class(
            rows=class_alignment_delta_rows,
            metric="own_minus_buildings_centroid_l2_delta_vs_baseline",
            ylabel="Own minus buildings L2 delta vs baseline",
            title=f"Own-vs-Buildings L2 Margin Delta vs {baseline_dataset.capitalize()}",
            out_path=plots_dir / "class_alignment_delta" / "own_minus_buildings_centroid_l2_delta_vs_baseline.png",
            use_zero_line=True,
        )
        plot_alignment_metric_by_class(
            rows=class_alignment_delta_rows,
            metric="own_minus_water_centroid_l2_delta_vs_baseline",
            ylabel="Own minus water L2 delta vs baseline",
            title=f"Own-vs-Water L2 Margin Delta vs {baseline_dataset.capitalize()}",
            out_path=plots_dir / "class_alignment_delta" / "own_minus_water_centroid_l2_delta_vs_baseline.png",
            use_zero_line=True,
        )
        plot_alignment_metric_by_class(
            rows=class_alignment_delta_rows,
            metric="own_real_rank_by_centroid_l2_delta_vs_baseline",
            ylabel="Own real-class rank delta vs baseline",
            title=f"Own Real-Class Rank Delta vs {baseline_dataset.capitalize()}",
            out_path=plots_dir / "class_alignment_delta" / "own_real_rank_by_centroid_l2_delta_vs_baseline.png",
            use_zero_line=True,
        )

    if create_umap:
        plot_class_umap_projections(
            feats_by_dataset=feats_by_dataset,
            dataset_names=dataset_names,
            config=config,
            out_dir=out_dir,
        )

    print_summary(pairwise_rows, improvement_rows)
    print(f"\nSaved combined class feature cache + analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()
