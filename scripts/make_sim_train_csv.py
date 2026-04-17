#!/usr/bin/env python3
"""
Build segmentation-training CSV splits from a sim subset manifest.

The manifest should contain one RGB image path per line, either relative to
``src_root`` or absolute. The script deterministically shuffles the manifest
entries, then writes:

    sim_train.csv
    sim_val.csv
    sim_test.csv

Each CSV contains:

    img_path,mask_path

where ``mask_path`` is derived by swapping the RGB parent directory (default:
``opt``) for the label parent directory (default: ``gt_ss_mask``) while keeping
the same filename.

By default the written paths are relative to ``src_root`` so the CSVs can be
reused across copied or translated subset roots that preserve the same internal
directory layout.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Text manifest with one RGB image path per line.",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Dataset root used to resolve relative manifest entries.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where sim_train.csv, sim_val.csv, and sim_test.csv will be written.",
    )
    parser.add_argument(
        "--rgb-parent-dir",
        type=str,
        default="opt",
        help="Immediate parent directory name for RGB images listed in the manifest.",
    )
    parser.add_argument(
        "--label-parent-dir",
        type=str,
        default="gt_ss_mask",
        help="Sibling directory name containing masks with matching filenames.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths instead of the default relative-to-src_root paths.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of rows written to sim_train.csv.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows written to sim_val.csv.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of rows written to sim_test.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic shuffling before splitting.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=None,
        help="Exact number of rows to write to sim_train.csv. Overrides ratios when used with val/test counts.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="Exact number of rows to write to sim_val.csv. Overrides ratios when used with train/test counts.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Exact number of rows to write to sim_test.csv. Overrides ratios when used with train/val counts.",
    )
    return parser.parse_args()


def iter_manifest_rgb_paths(manifest: Path, src_root: Path) -> list[Path]:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    src_root = src_root.resolve()
    rgb_paths: list[Path] = []
    seen: set[Path] = set()

    for line_num, raw_line in enumerate(manifest.read_text().splitlines(), start=1):
        entry = raw_line.strip()
        if not entry or entry.startswith("#"):
            continue

        path = Path(entry)
        if not path.is_absolute():
            path = src_root / path
        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"Missing RGB file at line {line_num}: {path}")
        if not path.is_file():
            raise ValueError(f"Manifest entry is not a file at line {line_num}: {path}")
        try:
            path.relative_to(src_root)
        except ValueError as exc:
            raise ValueError(
                f"Manifest entry is outside src_root at line {line_num}: {path}"
            ) from exc

        if path in seen:
            continue

        seen.add(path)
        rgb_paths.append(path)

    if not rgb_paths:
        raise ValueError(f"No usable RGB paths found in manifest: {manifest}")

    return rgb_paths


def derive_mask_path(
    img_path: Path,
    src_root: Path,
    rgb_parent_dir: str,
    label_parent_dir: str,
) -> Path:
    try:
        img_path.relative_to(src_root)
    except ValueError as exc:
        raise ValueError(f"Image path is outside src_root: {img_path}") from exc

    if img_path.parent.name != rgb_parent_dir:
        raise ValueError(
            f"Expected RGB file under '{rgb_parent_dir}', got '{img_path.parent.name}': {img_path}"
        )

    mask_path = img_path.parent.parent / label_parent_dir / img_path.name
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask for {img_path}: expected {mask_path}")
    if not mask_path.is_file():
        raise ValueError(f"Derived mask path is not a file: {mask_path}")

    return mask_path.resolve()


def format_path(path: Path, src_root: Path, absolute_paths: bool) -> str:
    if absolute_paths:
        return str(path)
    return str(path.relative_to(src_root))


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    return train_count, val_count, test_count


def resolve_split_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    train_count: int | None,
    val_count: int | None,
    test_count: int | None,
) -> tuple[int, int, int, str]:
    explicit_counts = [train_count, val_count, test_count]
    if any(count is not None for count in explicit_counts):
        if not all(count is not None for count in explicit_counts):
            raise ValueError(
                "When specifying exact split counts, provide --train-count, --val-count, and --test-count together."
            )
        if any(count < 0 for count in explicit_counts):
            raise ValueError("Exact split counts must be non-negative.")
        if train_count + val_count + test_count != total:
            raise ValueError(
                "Exact split counts must sum to the number of manifest rows. "
                f"Got train={train_count}, val={val_count}, test={test_count}, total={total}."
            )
        return train_count, val_count, test_count, "exact_counts"

    train_count, val_count, test_count = split_counts(total, train_ratio, val_ratio, test_ratio)
    return train_count, val_count, test_count, "ratios"


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["img_path", "mask_path"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    src_root = args.src_root.resolve()
    out_dir = args.out_dir.resolve()

    rgb_paths = iter_manifest_rgb_paths(manifest, src_root)

    rows: list[dict[str, str]] = []
    for img_path in rgb_paths:
        mask_path = derive_mask_path(
            img_path=img_path,
            src_root=src_root,
            rgb_parent_dir=args.rgb_parent_dir,
            label_parent_dir=args.label_parent_dir,
        )
        rows.append(
            {
                "img_path": format_path(img_path, src_root, args.absolute_paths),
                "mask_path": format_path(mask_path, src_root, args.absolute_paths),
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    train_count, val_count, test_count, split_mode = resolve_split_counts(
        total=len(rows),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
    )

    train_rows = rows[:train_count]
    val_rows = rows[train_count : train_count + val_count]
    test_rows = rows[train_count + val_count :]

    if len(test_rows) != test_count:
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "sim_train.csv"
    val_csv = out_dir / "sim_val.csv"
    test_csv = out_dir / "sim_test.csv"

    write_csv(train_rows, train_csv)
    write_csv(val_rows, val_csv)
    write_csv(test_rows, test_csv)

    print(f"Manifest: {manifest}")
    print(f"Source root: {src_root}")
    print(f"Output directory: {out_dir}")
    print(f"Seed: {args.seed}")
    if split_mode == "exact_counts":
        print(
            "Exact counts: "
            f"train={train_count}, val={val_count}, test={test_count}"
        )
    else:
        print(
            "Ratios: "
            f"train={args.train_ratio:.4f}, val={args.val_ratio:.4f}, test={args.test_ratio:.4f}"
        )
    print(f"Wrote {train_csv.name}: {len(train_rows)} rows")
    print(f"Wrote {val_csv.name}: {len(val_rows)} rows")
    print(f"Wrote {test_csv.name}: {len(test_rows)} rows")


if __name__ == "__main__":
    main()
