#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_image_dirs(root_dir: Path, image_parent_dir: str) -> list[Path]:
    image_dirs: list[Path] = []
    seen: set[Path] = set()

    for path in sorted(root_dir.rglob(image_parent_dir)):
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        image_dirs.append(resolved)

    return image_dirs


def count_matching_image_labels(
    image_dirs: list[Path],
    label_parent_dir: str,
) -> tuple[int, int, Counter[tuple[int, int]]]:
    total_images = 0
    matched_images = 0
    size_counts: Counter[tuple[int, int]] = Counter()

    for image_dir in image_dirs:
        dataset_dir = image_dir.parent
        label_dir = dataset_dir / label_parent_dir
        if not label_dir.is_dir():
            continue

        for image_path in image_dir.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            total_images += 1
            rel_image_path = image_path.relative_to(image_dir)
            label_path = label_dir / rel_image_path

            if label_path.is_file():
                matched_images += 1
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(f"Could not read image: {image_path}")

                height, width = image.shape[:2]
                size_counts[(width, height)] += 1

    return total_images, matched_images, size_counts


def main() -> None:
    root_dir = Path("/path/to/data/root")  # Root directory to scan for image and label folder pairs.
    image_parent_dir = "opt"  # Immediate folder name expected to contain the images.
    label_parent_dir = "gt_ss_mask"  # Immediate sibling folder name expected to contain the labels.

    if root_dir == Path("/path/to/data/root"):
        raise ValueError("Set root_dir in main() before running this script.")

    root_dir = root_dir.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise ValueError(f"Root directory is not a directory: {root_dir}")

    image_dirs = find_image_dirs(root_dir=root_dir, image_parent_dir=image_parent_dir)
    if not image_dirs:
        raise FileNotFoundError(
            f"No '{image_parent_dir}' directories found under {root_dir}"
        )

    total_images, matched_images, size_counts = count_matching_image_labels(
        image_dirs=image_dirs,
        label_parent_dir=label_parent_dir,
    )

    print(f"Root directory: {root_dir}")
    print(f"Found {len(image_dirs)} '{image_parent_dir}' directories")
    print(f"Total images under '{image_parent_dir}/': {total_images}")
    print(
        f"Images with matching labels in sibling '{label_parent_dir}/': {matched_images}"
    )
    print("Paired image size counts:")
    for (width, height), count in sorted(size_counts.items()):
        print(f"{width}x{height}: {count}")


if __name__ == "__main__":
    main()
