#!/usr/bin/env python3
"""
Save random sample grids for each tiled real-image dataset under a root directory.

Example:
    python scripts/plot_real_dataset_samples.py \
        --root /cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/real
"""

from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "This script requires OpenCV (`cv2`). Activate the project environment first."
        ) from exc
    return cv2


def load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "This script requires `matplotlib`. Activate the project environment first."
        ) from exc
    return plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot random sample grids for each dataset under a tiled real-image root."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/real"),
        help="Root directory whose immediate subdirectories are treated as datasets.",
    )
    parser.add_argument(
        "--parent-dir",
        type=str,
        default="opt",
        help="Parent dir of image files to collect.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/real_dataset_sample_grids"),
        help="Directory where per-dataset grid images will be written.",
    )
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=25,
        help="Maximum number of random images to show for each dataset.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in each grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed used for sampling.",
    )
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Optional cap on how many dataset subdirectories to process.",
    )
    parser.add_argument(
        "--max-label-chars",
        type=int,
        default=40,
        help="Maximum number of relative-path characters to show per tile title.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def discover_datasets(root: Path) -> list[Path]:
    datasets = [path for path in sorted(root.iterdir()) if path.is_dir()]
    return datasets


def collect_images(dataset_dir: Path, parent_dir: str) -> list[Path]:
    return [
        path
        for path in sorted(dataset_dir.rglob("*"))
        if is_image_file(path) and path.parent.name == parent_dir
    ]


def load_rgb(path: Path):
    cv2 = load_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def shorten_label(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return "..." + text[-(max_chars - 3) :]


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return slug.strip("._") or "dataset"


def save_grid(
    dataset_dir: Path,
    image_paths: list[Path],
    total_images: int,
    out_dir: Path,
    cols: int,
    max_label_chars: int,
) -> Path:
    plt = load_pyplot()
    num_images = len(image_paths)
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))

    if hasattr(axes, "ravel"):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]

    for ax, image_path in zip(axes_list, image_paths):
        image = load_rgb(image_path)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        rel_path = image_path.relative_to(dataset_dir)
        ax.set_title(shorten_label(str(rel_path), max_label_chars), fontsize=8)

    for ax in axes_list[num_images:]:
        ax.axis("off")

    fig.suptitle(
        f"{dataset_dir.name}: {num_images} random samples "
        f"(dataset contains {total_images} images)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = out_dir / f"{slugify(dataset_dir.name)}_samples.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()

    if args.samples_per_dataset < 1:
        raise ValueError("--samples-per-dataset must be >= 1")
    if args.cols < 1:
        raise ValueError("--cols must be >= 1")

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = discover_datasets(root)
    if args.max_datasets is not None:
        dataset_dirs = dataset_dirs[: args.max_datasets]

    if not dataset_dirs:
        raise RuntimeError(f"No dataset subdirectories found under: {root}")

    print(f"Found {len(dataset_dirs)} dataset directories under {root}")
    print(f"Saving sample grids to {out_dir}")

    processed = 0
    skipped = 0

    for dataset_index, dataset_dir in enumerate(dataset_dirs):
        all_images = collect_images(dataset_dir, args.parent_dir)
        if not all_images:
            skipped += 1
            print(f"[skip] {dataset_dir.name}: no image files found")
            continue

        rng = random.Random(args.seed + dataset_index)
        sample_count = min(args.samples_per_dataset, len(all_images))
        sampled_images = rng.sample(all_images, sample_count)

        out_path = save_grid(
            dataset_dir=dataset_dir,
            image_paths=sampled_images,
            total_images=len(all_images),
            out_dir=out_dir,
            cols=args.cols,
            max_label_chars=args.max_label_chars,
        )
        processed += 1
        print(
            f"[ok] {dataset_dir.name}: saved {sample_count} samples "
            f"out of {len(all_images)} images -> {out_path}"
        )

    print(
        f"Finished: processed {processed} datasets, skipped {skipped}, "
        f"output dir = {out_dir}"
    )


if __name__ == "__main__":
    main()
