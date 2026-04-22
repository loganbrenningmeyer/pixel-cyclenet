#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cyclenet.eval.plotting.project import (
    plot_proj_density,
    plot_proj_density_marginal,
    plot_proj_scatter,
)
from cyclenet.eval.project import UmapProjector


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


def label_name(label_id: int) -> str:
    return CLASS_INFO.get(label_id, (f"Unknown({label_id})", f"unknown_{label_id}"))[0]


def label_slug(label_id: int) -> str:
    return CLASS_INFO.get(label_id, (f"Unknown({label_id})", f"unknown_{label_id}"))[1]


def load_class_embeddings(cache_path: Path) -> dict[int, np.ndarray]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached class feature bundle: {cache_path}")

    bundle = np.load(cache_path)
    return {
        int(name.removeprefix("class_")): np.asarray(bundle[name], dtype=np.float32)
        for name in sorted(bundle.files)
        if name.startswith("class_")
    }


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


def compute_pairwise_rows(
    feats_by_dataset: dict[str, dict[int, np.ndarray]],
    reference_dataset: str,
    comparison_datasets: list[str],
) -> list[dict[str, object]]:
    if reference_dataset not in feats_by_dataset:
        raise KeyError(f"reference_dataset '{reference_dataset}' not found in cached datasets")

    rows: list[dict[str, object]] = []
    ref_feats_by_class = feats_by_dataset[reference_dataset]
    label_ids = sorted({label_id for dataset_feats in feats_by_dataset.values() for label_id in dataset_feats})

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


def plot_class_umap_projections(
    feats_by_dataset: dict[str, dict[int, np.ndarray]],
    dataset_names: list[str],
    config: DictConfig,
    out_dir: Path,
) -> None:
    projection_method = str(cfg_select(config, "projection.method", "umap")).lower()
    if projection_method != "umap":
        raise ValueError(
            f"Expected projection.method to be 'umap' in analysis config, got '{projection_method}'"
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

        if marginal_show and (density_show or point_show):
            plot_proj_density_marginal(
                coords=coords,
                labels=plot_labels,
                colors=plot_colors,
                title=class_title,
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                save_path=umap_dir / f"{class_stem}_marginal.pdf",
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
        elif density_show:
            plot_proj_density(
                coords=coords,
                labels=plot_labels,
                colors=plot_colors,
                title=class_title,
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                save_path=umap_dir / f"{class_stem}_density.pdf",
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
        elif point_show:
            plot_proj_scatter(
                coords=coords,
                labels=plot_labels,
                colors=plot_colors,
                title=class_title,
                xlabel="UMAP 1",
                ylabel="UMAP 2",
                save_path=umap_dir / f"{class_stem}_scatter.pdf",
                max_points_per_group=max_points_per_group,
                seed=random_state,
                alpha=point_alpha,
                point_size=point_size,
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

    # Root directory containing cached `<dataset>/deeplab_class_features.npz` bundles.
    cache_dir = Path(config.data.cache_dir)
    # Dataset name treated as the real/reference distribution.
    reference_dataset = str(cfg_select(config, "data.reference_dataset", "real"))
    # Dataset names compared against the reference dataset.
    comparison_datasets = [str(value) for value in cfg_select(config, "data.comparison_datasets", ["sim", "translated"])]
    # Optional baseline dataset used to compute improvement deltas. Set to null to skip.
    baseline_dataset_value = cfg_select(config, "data.baseline_dataset", "sim")
    baseline_dataset = str(baseline_dataset_value) if baseline_dataset_value is not None else None
    # Output directory for CSV tables and metric plots.
    out_dir = Path(config.data.out_dir)
    # Toggle for generating class-wise UMAP projections from cached feature vectors.
    create_umap = bool(cfg_select(config, "projection.create_umap", False))

    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir does not exist: {cache_dir}")

    dataset_names = sorted({reference_dataset, *comparison_datasets})
    if baseline_dataset is not None:
        dataset_names = sorted({*dataset_names, baseline_dataset})

    feats_by_dataset = {
        dataset_name: load_class_embeddings(cache_dir / dataset_name / "deeplab_class_features.npz")
        for dataset_name in dataset_names
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    pairwise_comparison_datasets = list(comparison_datasets)
    if baseline_dataset is not None and baseline_dataset not in pairwise_comparison_datasets:
        pairwise_comparison_datasets.append(baseline_dataset)

    pairwise_rows = compute_pairwise_rows(
        feats_by_dataset=feats_by_dataset,
        reference_dataset=reference_dataset,
        comparison_datasets=pairwise_comparison_datasets,
    )
    write_csv(pairwise_rows, out_dir / "pairwise_class_distances.csv")

    improvement_rows: list[dict[str, object]] = []
    if baseline_dataset is not None:
        improvement_rows = build_improvement_rows(
            pairwise_rows=pairwise_rows,
            reference_dataset=reference_dataset,
            baseline_dataset=baseline_dataset,
        )
        if improvement_rows:
            write_csv(improvement_rows, out_dir / "improvement_vs_baseline.csv")

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

    if create_umap:
        plot_class_umap_projections(
            feats_by_dataset=feats_by_dataset,
            dataset_names=dataset_names,
            config=config,
            out_dir=out_dir,
        )

    print_summary(pairwise_rows, improvement_rows)
    print(f"\nSaved class feature distance analysis to {out_dir}")


if __name__ == "__main__":
    main()
