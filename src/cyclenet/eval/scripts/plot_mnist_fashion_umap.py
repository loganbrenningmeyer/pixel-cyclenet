import argparse
from pathlib import Path

import numpy as np

from cyclenet.eval.embed import CLIPEmbedder, InceptionEmbedder
from cyclenet.eval.plotting.project import (
    plot_umap_density,
    plot_umap_density_marginal,
    plot_umap_scatter,
)
from cyclenet.eval.project import UmapProjector


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PLOT_COLORS = {
    "mnist": "#6b7280",
    "fashion": "#dc2626",
    "translated": "#2563eb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a shared UMAP on MNIST and Fashion-MNIST embeddings, then "
            "project USPS into that learned space as a translated/OOD set."
        )
    )
    parser.add_argument(
        "--mnist-dir",
        type=Path,
        default=Path("data/mnist"),
        help="Root directory containing MNIST train/test image folders.",
    )
    parser.add_argument(
        "--fashion-dir",
        type=Path,
        default=Path("data/fashion"),
        help="Root directory containing Fashion train/test image folders.",
    )
    parser.add_argument(
        "--translated-dir",
        type=Path,
        default=Path("data/usps"),
        help="Root directory containing USPS train/test image folders.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help="Dataset split to use from each dataset root.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for CLIP/Inception embedding extraction.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional max number of images to keep per dataset before embedding.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used for reproducible image subsampling and plot sampling.",
    )
    parser.add_argument(
        "--umap-random-state",
        type=int,
        default=42,
        help="Random state for the shared UMAP projector.",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=2000,
        help="Maximum plotted points per dataset group.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for embedding extraction, defaults to cuda if available.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/umap/mnist_fashion_usps"),
        help="Directory where plots and optional arrays will be written.",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save raw embedding arrays for both datasets and both embedders.",
    )
    parser.add_argument(
        "--save-coords",
        action="store_true",
        help="Save projected 2D UMAP coordinates for both datasets and both embedders.",
    )
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Fill KDE density regions. If omitted, contours are unfilled.",
    )
    parser.add_argument(
        "--density-alpha",
        type=float,
        default=0.5,
        help="Alpha used for KDE density regions and contours.",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.1,
        help="Alpha used for optional point overlays on density plots.",
    )
    return parser.parse_args()


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

    plot_umap_scatter(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP: MNIST, Fashion-MNIST, USPS",
        save_path=embed_dir / f"{name}_umap_scatter.pdf",
        max_points_per_group=plot_max_points,
        seed=sample_seed,
    )
    plot_umap_density(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP Density: MNIST, Fashion-MNIST, USPS",
        save_path=embed_dir / f"{name}_umap_density.pdf",
        max_points_per_group=plot_max_points,
        fill=fill,
        alpha=density_alpha,
        point_alpha=point_alpha,
        seed=sample_seed,
        show_points=False,
    )
    plot_umap_density(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP Density + Points: MNIST, Fashion-MNIST, USPS",
        save_path=embed_dir / f"{name}_umap_density_points.pdf",
        max_points_per_group=plot_max_points,
        fill=fill,
        alpha=density_alpha,
        point_alpha=point_alpha,
        seed=sample_seed,
        show_points=True,
    )
    plot_umap_density_marginal(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP Marginal Density: MNIST, Fashion-MNIST, USPS",
        save_path=embed_dir / f"{name}_umap_density_marginal.pdf",
        max_points_per_group=plot_max_points,
        fill=fill,
        alpha=density_alpha,
        point_alpha=point_alpha,
        seed=sample_seed,
        show_points=False,
    )
    plot_umap_density_marginal(
        coords=[mnist_coords, fashion_coords, translated_coords],
        labels=["MNIST", "Fashion-MNIST", "USPS (transformed)"],
        colors=[PLOT_COLORS["mnist"], PLOT_COLORS["fashion"], PLOT_COLORS["translated"]],
        title=f"{name.upper()} UMAP Marginal Density + Points: MNIST, Fashion-MNIST, USPS",
        save_path=embed_dir / f"{name}_umap_density_marginal_points.pdf",
        max_points_per_group=plot_max_points,
        fill=fill,
        alpha=density_alpha,
        point_alpha=point_alpha,
        seed=sample_seed,
        show_points=True,
    )


def main():
    args = parse_args()

    mnist_paths = gather_image_paths(args.mnist_dir, args.split)
    fashion_paths = gather_image_paths(args.fashion_dir, args.split)
    translated_paths = gather_image_paths(args.translated_dir, args.split)

    mnist_paths = maybe_subsample(mnist_paths, args.max_images, args.sample_seed)
    fashion_paths = maybe_subsample(fashion_paths, args.max_images, args.sample_seed + 1)
    translated_paths = maybe_subsample(translated_paths, args.max_images, args.sample_seed + 2)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_embedder(
        name="clip",
        embedder=CLIPEmbedder(device=args.device),
        mnist_paths=mnist_paths,
        fashion_paths=fashion_paths,
        translated_paths=translated_paths,
        batch_size=args.batch_size,
        umap_random_state=args.umap_random_state,
        plot_max_points=args.plot_max_points,
        fill=args.fill,
        density_alpha=args.density_alpha,
        point_alpha=args.point_alpha,
        sample_seed=args.sample_seed,
        out_dir=args.out_dir,
        save_embeddings=args.save_embeddings,
        save_coords=args.save_coords,
    )

    run_embedder(
        name="inception",
        embedder=InceptionEmbedder(device=args.device),
        mnist_paths=mnist_paths,
        fashion_paths=fashion_paths,
        translated_paths=translated_paths,
        batch_size=args.batch_size,
        umap_random_state=args.umap_random_state,
        plot_max_points=args.plot_max_points,
        fill=args.fill,
        density_alpha=args.density_alpha,
        point_alpha=args.point_alpha,
        sample_seed=args.sample_seed,
        out_dir=args.out_dir,
        save_embeddings=args.save_embeddings,
        save_coords=args.save_coords,
    )


if __name__ == "__main__":
    main()
