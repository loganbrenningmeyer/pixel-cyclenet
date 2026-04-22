#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from cyclenet.eval.plotting.image_grid import plot_image_grid


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def collect_image_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    image_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not image_paths:
        raise ValueError(f"No image files found under {root}")
    return image_paths


def sample_image_paths(
    image_paths: list[Path],
    num_samples: int | None,
    seed: int,
) -> list[Path]:
    if num_samples is None:
        return image_paths
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if len(image_paths) <= num_samples:
        return image_paths

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(image_paths), size=num_samples, replace=False))
    return [image_paths[i] for i in idx]


def infer_grid_shape(
    num_images: int,
    n_rows: int | None,
    n_cols: int | None,
) -> tuple[int, int]:
    if num_images <= 0:
        raise ValueError("num_images must be positive")

    if n_rows is None and n_cols is None:
        raise ValueError("Set at least one of n_rows or n_cols.")
    if n_rows is not None and n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if n_cols is not None and n_cols <= 0:
        raise ValueError(f"n_cols must be positive, got {n_cols}")

    if n_rows is None:
        n_rows = int(np.ceil(num_images / n_cols))
    elif n_cols is None:
        n_cols = int(np.ceil(num_images / n_rows))

    if n_rows * n_cols < num_images:
        raise ValueError(
            f"Grid shape ({n_rows} x {n_cols}) cannot fit {num_images} images. "
            "Increase n_rows or n_cols."
        )

    return n_rows, n_cols


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def main() -> None:
    # Root directory containing images to sample into the grid.
    data_dir = Path("/develop/data/remote_sensing/tiled/projection/cyclenet_sim_proj/all_real_ft_invar/step-30000/strength-0.5/cfg-5.0")

    # Output path for the saved image grid.
    save_path = Path("/develop/data/remote_sensing/tiled/projection/cyclenet_sim_proj/all_real_ft_invar/30000-0.5-5.0-image_grid.png")

    # Number of images to randomly sample from `data_dir`. Set to `None` to use all images.
    num_samples: int | None = 16

    # Number of grid rows. Set to `None` to infer it from `n_cols`.
    n_rows: int | None = None

    # Number of grid columns. Set to `None` to infer it from `n_rows`.
    n_cols: int | None = 4

    # RNG seed used for deterministic random sampling.
    seed = 42

    # Optional title shown above the image grid. Set to `None` for no title.
    title: str | None = None

    # Horizontal spacing between panels. Set to `0.0` for no horizontal gap.
    grid_wspace = 0.02

    # Vertical spacing between panels. Set to `0.0` for no vertical gap.
    grid_hspace = 0.02

    # Scale factor applied to the final figure size.
    scale = 1.0

    # DPI used when saving the figure.
    dpi = 300

    image_paths = collect_image_paths(data_dir)
    image_paths = sample_image_paths(image_paths, num_samples=num_samples, seed=seed)
    inferred_rows, inferred_cols = infer_grid_shape(
        num_images=len(image_paths),
        n_rows=n_rows,
        n_cols=n_cols,
    )
    images = [load_image(path) for path in image_paths]

    plot_image_grid(
        images=images,
        n_cols=inferred_cols,
        row_labels=None,
        col_labels=None,
        title=title,
        save_path=str(save_path),
        dpi=dpi,
        scale=scale,
        grid_wspace=grid_wspace,
        grid_hspace=grid_hspace,
        subplot_top_with_title=0.96,
        subplot_top_without_title=0.995,
        subplot_bottom_with_xlabels=0.04,
        subplot_bottom_without_xlabels=0.005,
        subplot_left_with_ylabels=0.02,
        subplot_left_without_ylabels=0.005,
        subplot_right=0.995,
        save_pad_inches=0.01,
        image_interpolation="nearest",
    )

    print(f"Saved {len(images)} sampled images as a {inferred_rows}x{inferred_cols} grid to {save_path}")


if __name__ == "__main__":
    main()
