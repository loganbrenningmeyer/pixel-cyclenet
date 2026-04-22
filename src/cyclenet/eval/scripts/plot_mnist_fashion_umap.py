from pathlib import Path

import numpy as np

from cyclenet.eval.embed import CLIPEmbedder, InceptionEmbedder
from cyclenet.eval.plotting.project import (
    plot_proj_density,
    plot_proj_density_marginal,
    plot_proj_scatter,
)
from cyclenet.eval.project import UmapProjector


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PLOT_COLORS = {
    "mnist": "#6b7280",
    "fashion": "#dc2626",
    "translated": "#2563eb",
}


def gather_image_paths(root: Path, split: str) -> list[str]:
    if split == "all":
        search_roots = [root / "train", root / "test"]
    else:
        search_roots = [root / split]

    paths: list[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            raise FileNotFoundError(f"Missing dataset split directory: {search_root}")

        for path in sorted(search_root.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                paths.append(path)

    if not paths:
        raise RuntimeError(f"No images found under {root} for split '{split}'.")

    return [str(path) for path in paths]


def maybe_subsample(paths: list[str], max_images: int | None, seed: int) -> list[str]:
    if max_images is None or len(paths) <= max_images:
        return paths

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(paths), size=max_images, replace=False))
    return [paths[i] for i in idx]


def save_array(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)


def run_embedder(
    *,
    name: str,
    embedder,
    mnist_paths: list[str],
    fashion_paths: list[str],
    translated_paths: list[str],
    batch_size: int,
    umap_random_state: int,
    plot_max_points: int,
    fill: bool,
    density_alpha: float,
    point_alpha: float,
    sample_seed: int,
    out_dir: Path,
    save_embeddings: bool,
    save_coords: bool,
):
    embed_dir = out_dir / name
    embed_dir.mkdir(parents=True, exist_ok=True)

    mnist_feats = embedder.embed(
        mnist_paths,
        batch_size=batch_size,
        save_path=(embed_dir / "mnist_embeddings.npy") if save_embeddings else None,
    )
    fashion_feats = embedder.embed(
        fashion_paths,
        batch_size=batch_size,
        save_path=(embed_dir / "fashion_embeddings.npy") if save_embeddings else None,
    )
    translated_feats = embedder.embed(
        translated_paths,
        batch_size=batch_size,
        save_path=(embed_dir / "translated_embeddings.npy") if save_embeddings else None,
    )

    projector = UmapProjector(n_components=2, random_state=umap_random_state)
    stacked_feats = np.concatenate([mnist_feats, fashion_feats], axis=0)
    stacked_coords = projector.fit(stacked_feats)

    mnist_coords = stacked_coords[: len(mnist_feats)]
    fashion_coords = stacked_coords[len(mnist_feats) :]
    translated_coords = projector.transform(translated_feats)

    if save_coords:
        save_array(mnist_coords, embed_dir / "mnist_umap_coords.npy")
        save_array(fashion_coords, embed_dir / "fashion_umap_coords.npy")
        save_array(translated_coords, embed_dir / "translated_umap_coords.npy")

    plot_proj_scatter(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP: MNIST, Fashion-MNIST, USPS",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        save_path=embed_dir / f"{name}_umap_scatter.pdf",
        max_points_per_group=plot_max_points,
        seed=sample_seed,
    )
    plot_proj_density(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP Density: MNIST, Fashion-MNIST, USPS",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        save_path=embed_dir / f"{name}_umap_density.pdf",
        max_points_per_group=plot_max_points,
        fill=fill,
        alpha=density_alpha,
        point_alpha=point_alpha,
        seed=sample_seed,
        show_points=False,
    )
    plot_proj_density(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP Density + Points: MNIST, Fashion-MNIST, USPS",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        save_path=embed_dir / f"{name}_umap_density_points.pdf",
        max_points_per_group=plot_max_points,
        fill=fill,
        alpha=density_alpha,
        point_alpha=point_alpha,
        seed=sample_seed,
        show_points=True,
    )
    plot_proj_density_marginal(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP Marginal Density: MNIST, Fashion-MNIST, USPS",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        save_path=embed_dir / f"{name}_umap_density_marginal.pdf",
        max_points_per_group=plot_max_points,
        fill=fill,
        alpha=density_alpha,
        point_alpha=point_alpha,
        seed=sample_seed,
        show_points=False,
    )
    plot_proj_density_marginal(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP Marginal Density + Points: MNIST, Fashion-MNIST, USPS",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        save_path=embed_dir / f"{name}_umap_density_marginal_points.pdf",
        max_points_per_group=plot_max_points,
        fill=fill,
        alpha=density_alpha,
        point_alpha=point_alpha,
        seed=sample_seed,
        show_points=True,
    )


def main():
    mnist_dir = Path("data/mnist")
    fashion_dir = Path("data/fashion")
    translated_dir = Path("data/usps")
    split = "test"
    batch_size = 64
    max_images = None
    sample_seed = 42
    umap_random_state = 42
    plot_max_points = 2000
    device = None
    out_dir = Path("outputs/umap/mnist_fashion_usps")
    save_embeddings = False
    save_coords = False
    fill = False
    density_alpha = 0.5
    point_alpha = 0.1

    mnist_paths = gather_image_paths(mnist_dir, split)
    fashion_paths = gather_image_paths(fashion_dir, split)
    translated_paths = gather_image_paths(translated_dir, split)

    mnist_paths = maybe_subsample(mnist_paths, max_images, sample_seed)
    fashion_paths = maybe_subsample(fashion_paths, max_images, sample_seed + 1)
    translated_paths = maybe_subsample(translated_paths, max_images, sample_seed + 2)

    out_dir.mkdir(parents=True, exist_ok=True)

    run_embedder(
        name="clip",
        embedder=CLIPEmbedder(device=device),
        mnist_paths=mnist_paths,
        fashion_paths=fashion_paths,
        translated_paths=translated_paths,
        batch_size=batch_size,
        umap_random_state=umap_random_state,
        plot_max_points=plot_max_points,
        fill=fill,
        density_alpha=density_alpha,
        point_alpha=point_alpha,
        sample_seed=sample_seed,
        out_dir=out_dir,
        save_embeddings=save_embeddings,
        save_coords=save_coords,
    )

    run_embedder(
        name="inception",
        embedder=InceptionEmbedder(device=device),
        mnist_paths=mnist_paths,
        fashion_paths=fashion_paths,
        translated_paths=translated_paths,
        batch_size=batch_size,
        umap_random_state=umap_random_state,
        plot_max_points=plot_max_points,
        fill=fill,
        density_alpha=density_alpha,
        point_alpha=point_alpha,
        sample_seed=sample_seed,
        out_dir=out_dir,
        save_embeddings=save_embeddings,
        save_coords=save_coords,
    )


if __name__ == "__main__":
    main()
