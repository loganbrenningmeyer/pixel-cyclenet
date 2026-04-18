from pathlib import Path
import argparse
import random
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig


from cyclenet.eval.embed import CLIPEmbedder, InceptionEmbedder, DeepLabEmbedder
from cyclenet.eval.project import UmapProjector, PcaProjector
from cyclenet.eval.plotting.project import (
    plot_proj_scatter, 
    plot_proj_density, 
    plot_proj_density_marginal,
)


def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)


def sample_images(root_dir: str | Path, num_samples: int) -> list[Path]:
    img_paths = []
    for path in Path(root_dir).rglob("*"):
        if path.parent.name != "opt":
            continue
        img_paths.append(path)

    sample_paths = random.sample(img_paths, k=num_samples)

    return sample_paths


def main():
    # -------------------------
    # Parse args / load + save config 
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    out_dir = Path(config.data.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, out_dir / "config.yaml")

    # -------------------------
    # Create Embedder
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.embedding.model == "clip":
        embedder = CLIPEmbedder(device=device, clip_path=config.embedding.model_path)
    elif config.embedding.model == "inception":
        embedder = InceptionEmbedder(device)
    elif config.embedding.model == "deeplab":
        embedder = DeepLabEmbedder(
            device=device,
            ckpt_path=config.embedding.model_path, 
            feature_layer=config.embedding.feature_layer,
        )

    # -------------------------
    # Sample images / get embeddings
    # -------------------------
    num_samples = config.embedding.num_samples
    batch_size = config.embedding.batch_size

    sim_images = sample_images(config.data.sim_dir, num_samples)
    real_images = sample_images(config.data.real_dir, num_samples)
    translated_images = sample_images(config.data.translated_dir, num_samples)

    sim_emb = embedder.embed(sim_images, batch_size, out_dir / "sim_embed.npy")
    real_emb = embedder.embed(real_images, batch_size, out_dir / "real_embed.npy")
    translated_emb = embedder.embed(translated_images, batch_size, out_dir / "translated_embed.npy")

    # -------------------------
    # Project embeddings to 2D coordinates
    # -------------------------
    if config.projection.method == "umap":
        projector = UmapProjector(
            n_components=config.projection.n_components,
            random_state=config.projection.random_state,
        )
        xlabel = "UMAP 1"
        ylabel = "UMAP 2"

    elif config.projection.method == "pca":
        projector = PcaProjector(
            n_components=config.projection.n_components,
            random_state=config.projection.random_state,
        )
        xlabel = "PCA 1"
        ylabel = "PCA 2"
    
    # -- Fit to real + sim embeddings
    coords = projector.fit(np.concatenate([sim_emb, real_emb], axis=0))
    sim_coords = projector.transform(sim_emb)
    real_coords = projector.transform(real_emb)
    translated_coords = projector.transform(translated_emb)

    # -------------------------
    # Plot projected embeddings
    # -------------------------
    show_points = config.plotting.points.show
    show_density = config.plotting.density.show
    show_marginal = config.plotting.marginal.show

    density_alpha = 0 if not show_density else config.plotting.density.alpha

    coords = [sim_coords, real_coords, translated_coords]
    labels = [config.plotting.labels.sim, config.plotting.labels.real, config.plotting.labels.translated]
    colors = [config.plotting.colors.sim, config.plotting.colors.real, config.plotting.colors.translated]

    if show_marginal and (show_density or show_points):
        plot_proj_density_marginal(
            coords=coords,
            labels=labels,
            colors=colors,
            title=config.plotting.title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=out_dir / "projection.pdf",
            max_points_per_group=config.plotting.points.max_points_per_group,
            seed=config.run.seed,
            fill=config.plotting.density.fill,
            alpha=density_alpha,
            point_alpha=config.plotting.points.alpha,
            point_size=config.plotting.points.size,
            show_points=show_points,
        )
    elif show_density:
        plot_proj_density(
            coords=coords,
            labels=labels,
            colors=colors,
            title=config.plotting.title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=out_dir / "projection.pdf",
            max_points_per_group=config.plotting.points.max_points_per_group,
            seed=config.run.seed,
            fill=config.plotting.density.fill,
            alpha=density_alpha,
            point_alpha=config.plotting.points.alpha,
            point_size=config.plotting.points.size,
            show_points=show_points,
        )
    elif show_points:
        plot_proj_scatter(
            coords=coords,
            labels=labels,
            colors=colors,
            title=config.plotting.title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=out_dir / "projection.pdf",
            max_points_per_group=config.plotting.points.max_points_per_group,
            seed=config.run.seed,
            alpha=config.plotting.points.alpha,
            point_size=config.plotting.points.size,
        )
    else:
        raise ValueError("show_points or show_density must be True.")


if __name__ == "__main__":
    main()