#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from cyclenet.data.dataset import load_label_mask


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CLASS_NAMES = {
    1: "Bareland",
    2: "Rangeland",
    3: "Developed Space",
    4: "Road",
    5: "Trees",
    6: "Water",
    7: "Agriculture land",
    8: "Buildings",
}


def find_dataset_dirs(root_dir: Path, rgb_parent_dir: str, label_parent_dir: str) -> list[Path]:
    dataset_dirs: list[Path] = []
    seen: set[Path] = set()

    for label_dir in sorted(path for path in root_dir.rglob(label_parent_dir) if path.is_dir()):
        dataset_dir = label_dir.parent.resolve()
        rgb_dir = dataset_dir / rgb_parent_dir
        if not rgb_dir.is_dir():
            continue
        if dataset_dir in seen:
            continue
        seen.add(dataset_dir)
        dataset_dirs.append(dataset_dir)

    return dataset_dirs


def collect_mask_paths(dataset_dirs: list[Path], label_parent_dir: str) -> list[Path]:
    mask_paths: list[Path] = []

    for dataset_dir in dataset_dirs:
        label_dir = dataset_dir / label_parent_dir
        for mask_path in sorted(label_dir.rglob("*")):
            if not mask_path.is_file():
                continue
            if mask_path.suffix.lower() not in IMAGE_EXTS:
                continue
            mask_paths.append(mask_path)

    return mask_paths


def validate_image_pairs(
    dataset_dirs: list[Path],
    mask_paths: list[Path],
    rgb_parent_dir: str,
    label_parent_dir: str,
) -> None:
    dataset_dir_set = set(dataset_dirs)

    for mask_path in mask_paths:
        dataset_dir = next(parent for parent in mask_path.parents if parent in dataset_dir_set)
        label_dir = dataset_dir / label_parent_dir
        rgb_dir = dataset_dir / rgb_parent_dir
        rel_mask_path = mask_path.relative_to(label_dir)
        image_path = rgb_dir / rel_mask_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image for mask {mask_path}: expected {image_path}"
            )
        if not image_path.is_file():
            raise ValueError(f"Expected paired image to be a file: {image_path}")


def count_labels(mask_paths: list[Path]) -> Counter[int]:
    counts: Counter[int] = Counter()
    valid_labels = set(CLASS_NAMES) | {0}

    for mask_index, mask_path in enumerate(mask_paths, start=1):
        if mask_index % 100 == 0 or mask_index == len(mask_paths):
            print(f"Processed {mask_index}/{len(mask_paths)} masks")

        mask = np.asarray(load_label_mask(mask_path), dtype=np.int64)
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {mask.shape} for {mask_path}")

        values, value_counts = np.unique(mask, return_counts=True)
        unexpected = sorted(int(value) for value in values.tolist() if int(value) not in valid_labels)
        if unexpected:
            raise ValueError(f"Found out-of-range labels {unexpected} in {mask_path}")

        for value, value_count in zip(values.tolist(), value_counts.tolist(), strict=True):
            counts[int(value)] += int(value_count)

    return counts


def build_rows(
    root_dir: Path,
    dataset_dirs: list[Path],
    mask_paths: list[Path],
    counts: Counter[int],
) -> list[dict[str, object]]:
    total_pixels = int(sum(counts.values()))
    ignore_pixels = int(counts.get(0, 0))
    non_ignore_pixels = total_pixels - ignore_pixels

    rows: list[dict[str, object]] = []
    for class_id, class_name in CLASS_NAMES.items():
        pixel_count = int(counts.get(class_id, 0))
        pixel_percent_all = 100.0 * pixel_count / total_pixels if total_pixels else 0.0
        pixel_percent_non_ignore = (
            100.0 * pixel_count / non_ignore_pixels if non_ignore_pixels else 0.0
        )

        rows.append(
            {
                "root_dir": str(root_dir),
                "dataset_dir_count": len(dataset_dirs),
                "mask_file_count": len(mask_paths),
                "class_id": class_id,
                "class_name": class_name,
                "pixel_count": pixel_count,
                "pixel_percent_all": pixel_percent_all,
                "pixel_percent_non_ignore": pixel_percent_non_ignore,
                "total_pixels": total_pixels,
                "ignore_pixels": ignore_pixels,
                "ignore_percent_all": 100.0 * ignore_pixels / total_pixels if total_pixels else 0.0,
                "non_ignore_pixels": non_ignore_pixels,
            }
        )

    return rows


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    if not rows:
        raise ValueError("No rows were generated for CSV output.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    first_row = rows[0]
    print(f"Root directory: {first_row['root_dir']}")
    print(f"Found {first_row['dataset_dir_count']} dataset directories")
    print(f"Counted {first_row['mask_file_count']} mask files")
    print(
        "class_id\tclass_name\tpixel_count\tpixel_percent_all\tpixel_percent_non_ignore"
    )

    for row in rows:
        print(
            f"{row['class_id']}\t{row['class_name']}\t{row['pixel_count']}"
            f"\t{float(row['pixel_percent_all']):.6f}"
            f"\t{float(row['pixel_percent_non_ignore']):.6f}"
        )


def main() -> None:
    root_dir = Path("/path/to/data/root")  # Root directory to scan for subdirectories containing sibling opt/ and gt_ss_mask/ folders.
    output_csv = Path("/path/to/output/class_pixel_percentages.csv")  # CSV file path where the aggregated per-class pixel percentages will be written.
    rgb_parent_dir = "opt"  # Immediate folder name expected to contain the RGB images.
    label_parent_dir = "gt_ss_mask"  # Immediate folder name expected to contain the segmentation masks.

    if root_dir == Path("/path/to/data/root"):
        raise ValueError("Set root_dir in main() before running this script.")
    if output_csv == Path("/path/to/output/class_pixel_percentages.csv"):
        raise ValueError("Set output_csv in main() before running this script.")

    root_dir = root_dir.resolve()
    output_csv = output_csv.resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise ValueError(f"Root directory is not a directory: {root_dir}")

    dataset_dirs = find_dataset_dirs(
        root_dir=root_dir,
        rgb_parent_dir=rgb_parent_dir,
        label_parent_dir=label_parent_dir,
    )
    if not dataset_dirs:
        raise FileNotFoundError(
            f"No dataset directories with sibling '{rgb_parent_dir}' and '{label_parent_dir}' folders were found under {root_dir}"
        )

    mask_paths = collect_mask_paths(dataset_dirs=dataset_dirs, label_parent_dir=label_parent_dir)
    if not mask_paths:
        raise FileNotFoundError(
            f"No mask files were found under '{label_parent_dir}' folders beneath {root_dir}"
        )

    validate_image_pairs(
        dataset_dirs=dataset_dirs,
        mask_paths=mask_paths,
        rgb_parent_dir=rgb_parent_dir,
        label_parent_dir=label_parent_dir,
    )

    counts = count_labels(mask_paths)
    rows = build_rows(
        root_dir=root_dir,
        dataset_dirs=dataset_dirs,
        mask_paths=mask_paths,
        counts=counts,
    )
    write_csv(rows=rows, output_csv=output_csv)
    print_summary(rows)
    print(f"Wrote CSV to {output_csv}")


if __name__ == "__main__":
    main()
