#!/usr/bin/env python3
"""
Copy a manifest-defined sim subset into a new dataset root.

The manifest should contain one RGB image path per line, either relative to
``src_root`` or absolute. Relative directory structure under ``src_root`` is
preserved under ``dst_root``.

By default the script also copies the corresponding segmentation mask with the
same filename from ``label_parent_dir``. This can be disabled with
``--no-copy-labels``. The default mapping is:

``.../<scene>/opt/img1.tif`` -> ``.../<scene>/gt_ss_mask/img1.tif``
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Text manifest with one image path per line.",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Original dataset root used to interpret relative manifest paths.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        required=True,
        help="Destination root where the subset will be copied.",
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
        help="Sibling directory name containing label masks with matching filenames.",
    )
    parser.add_argument(
        "--no-copy-labels",
        action="store_true",
        help="Copy only the RGB images from the manifest and skip label files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned copies without writing files.",
    )
    return parser.parse_args()


def iter_manifest_paths(manifest: Path, src_root: Path) -> list[Path]:
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    src_root = src_root.resolve()
    paths: list[Path] = []
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
            raise FileNotFoundError(f"Missing source file at line {line_num}: {path}")
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
        paths.append(path)

    if not paths:
        raise ValueError(f"No usable file paths found in manifest: {manifest}")

    return paths


def label_path_for_rgb(
    rgb_path: Path,
    src_root: Path,
    rgb_parent_dir: str,
    label_parent_dir: str,
) -> Path:
    try:
        rgb_path.relative_to(src_root)
    except ValueError as exc:
        raise ValueError(f"RGB path is outside src_root: {rgb_path}") from exc

    if rgb_path.parent.name != rgb_parent_dir:
        raise ValueError(
            f"Expected RGB file under '{rgb_parent_dir}', got '{rgb_path.parent.name}': {rgb_path}"
        )

    label_path = rgb_path.parent.parent / label_parent_dir / rgb_path.name
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file for {rgb_path}: expected {label_path}")
    if not label_path.is_file():
        raise ValueError(f"Derived label path is not a file: {label_path}")

    return label_path


def copy_file(src_path: Path, dst_path: Path, overwrite: bool, dry_run: bool) -> str:
    if dst_path.exists() and not overwrite:
        return "skipped"

    if dry_run:
        print(f"{src_path} -> {dst_path}")
        return "copied"

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return "copied"


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()

    rgb_paths = iter_manifest_paths(manifest, src_root)

    copied_rgb = 0
    skipped_rgb = 0
    copied_labels = 0
    skipped_labels = 0
    copy_labels = not args.no_copy_labels

    for rgb_path in rgb_paths:
        rgb_dst_path = dst_root / rgb_path.relative_to(src_root)

        rgb_result = copy_file(
            src_path=rgb_path,
            dst_path=rgb_dst_path,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        if rgb_result == "copied":
            copied_rgb += 1
        else:
            skipped_rgb += 1

        if copy_labels:
            label_path = label_path_for_rgb(
                rgb_path=rgb_path,
                src_root=src_root,
                rgb_parent_dir=args.rgb_parent_dir,
                label_parent_dir=args.label_parent_dir,
            )
            label_dst_path = dst_root / label_path.relative_to(src_root)

            label_result = copy_file(
                src_path=label_path,
                dst_path=label_dst_path,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            if label_result == "copied":
                copied_labels += 1
            else:
                skipped_labels += 1

    print(f"Manifest: {manifest}")
    print(f"Source root: {src_root}")
    print(f"Destination root: {dst_root}")
    print(f"Listed RGB files: {len(rgb_paths)}")
    print(f"Copy labels: {copy_labels}")
    print(f"Copied RGB files: {copied_rgb}")
    print(f"Skipped existing RGB files: {skipped_rgb}")
    print(f"Copied label files: {copied_labels}")
    print(f"Skipped existing label files: {skipped_labels}")


if __name__ == "__main__":
    main()
