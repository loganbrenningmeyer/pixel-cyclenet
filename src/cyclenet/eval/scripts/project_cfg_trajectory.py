import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from cyclenet.eval.plotting.set_style import apply_style
from cyclenet.eval.scripts.project_translated import (
    compute_reference_axis_limits,
    load_or_fit_reference_projection,
    load_config,
    save_config,
    title_names,
)

apply_style()


def sample_coords(xy: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or len(xy) <= max_points:
        return xy

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xy), size=max_points, replace=False)
    return xy[idx]


def cfg_select(config: DictConfig, key: str, default=None):
    value = OmegaConf.select(config, key)
    return default if value is None else value


def combo_dir(projection_root: Path, step: int | str, strength: float, cfg_weight: float) -> Path:
    return (
        projection_root
        / f"step-{step}"
        / f"strength-{strength}"
        / f"cfg-{cfg_weight}"
    )


def translated_embed_path(projection_root: Path, model: str, step: int | str, strength: float, cfg_weight: float) -> Path:
    return combo_dir(projection_root, step, strength, cfg_weight) / f"{model}_translated_embed.npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    out_dir = Path(config.data.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    model = str(config.embedding.model).lower()
    method = str(config.projection.method).lower()
    projection_root = Path(config.data.projection_root)
    reference_cache_dir = Path(config.data.reference_cache_dir) / model

    sim_emb = np.load(reference_cache_dir / "sim_embed.npy")
    real_emb = np.load(reference_cache_dir / "real_embed.npy")
    projector, sim_coords, real_coords = load_or_fit_reference_projection(
        method=method,
        projection_config=config.projection,
        cache_dir=reference_cache_dir,
        sim_emb=sim_emb,
        real_emb=real_emb,
    )

    xlim, ylim = compute_reference_axis_limits(
        sim_coords=sim_coords,
        real_coords=real_coords,
        pad_frac=float(cfg_select(config, "plotting.axis_pad_frac", 0.05)),
    )

    step = cfg_select(config, "trajectory.step")
    if step is None:
        raise ValueError("trajectory.step is required.")
    strength = float(cfg_select(config, "trajectory.noise_strength"))
    cfg_weights = [float(v) for v in list(cfg_select(config, "trajectory.cfg_weights", []))]
    if not cfg_weights:
        raise ValueError("trajectory.cfg_weights must contain at least one value.")

    centroids = []
    for cfg_weight in cfg_weights:
        embed_path = translated_embed_path(
            projection_root=projection_root,
            model=model,
            step=step,
            strength=strength,
            cfg_weight=cfg_weight,
        )
        if not embed_path.exists():
            raise FileNotFoundError(f"Missing translated embedding cache: {embed_path}")

        translated_emb = np.load(embed_path)
        translated_coords = projector.transform(translated_emb)
        centroids.append(translated_coords.mean(axis=0))

    centroids = np.stack(centroids, axis=0)

    sim_points = sample_coords(
        sim_coords,
        max_points=cfg_select(config, "plotting.reference.max_points_per_group", 2000),
        seed=int(config.run.seed),
    )
    real_points = sample_coords(
        real_coords,
        max_points=cfg_select(config, "plotting.reference.max_points_per_group", 2000),
        seed=int(config.run.seed) + 1,
    )

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.scatter(
        sim_points[:, 0],
        sim_points[:, 1],
        s=float(cfg_select(config, "plotting.reference.point_size", 8.0)),
        alpha=float(cfg_select(config, "plotting.reference.alpha", 0.10)),
        color=str(config.plotting.colors.sim),
        linewidths=0,
        edgecolors="none",
        rasterized=len(sim_points) > 3000,
        label=str(config.plotting.labels.sim),
    )
    ax.scatter(
        real_points[:, 0],
        real_points[:, 1],
        s=float(cfg_select(config, "plotting.reference.point_size", 8.0)),
        alpha=float(cfg_select(config, "plotting.reference.alpha", 0.10)),
        color=str(config.plotting.colors.real),
        linewidths=0,
        edgecolors="none",
        rasterized=len(real_points) > 3000,
        label=str(config.plotting.labels.real),
    )

    traj_color = str(cfg_select(config, "plotting.trajectory.color", "#2563eb"))
    ax.plot(
        centroids[:, 0],
        centroids[:, 1],
        color=traj_color,
        linewidth=float(cfg_select(config, "plotting.trajectory.line_width", 2.0)),
        alpha=float(cfg_select(config, "plotting.trajectory.alpha", 1.0)),
        zorder=3,
        label=str(config.plotting.labels.trajectory),
    )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=float(cfg_select(config, "plotting.trajectory.marker_size", 55.0)),
        color=traj_color,
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
    )

    if bool(cfg_select(config, "plotting.trajectory.annotate_cfg", True)):
        dx = float(cfg_select(config, "plotting.trajectory.annotation_dx", 0.02))
        dy = float(cfg_select(config, "plotting.trajectory.annotation_dy", 0.02))
        for centroid, cfg_weight in zip(centroids, cfg_weights):
            ax.text(
                centroid[0] + dx,
                centroid[1] + dy,
                f"{cfg_weight:g}",
                color=traj_color,
                ha="left",
                va="bottom",
            )

    method_name = title_names["method"][method]
    model_name = title_names["model"][model]
    title = cfg_select(
        config,
        "plotting.title",
        f"{method_name} CFG Centroid Trajectory ({model_name}, step {step}, strength {strength:g})",
    )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1" if method == "umap" else "PCA 1")
    ax.set_ylabel("UMAP 2" if method == "umap" else "PCA 2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)

    out_path = out_dir / f"{model}_{method}_step-{step}_strength-{strength:g}_cfg_trajectory.pdf"
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Wrote centroid trajectory plot to {out_path}")


if __name__ == "__main__":
    main()
