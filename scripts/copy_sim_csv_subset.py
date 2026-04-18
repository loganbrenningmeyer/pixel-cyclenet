#!/usr/bin/env python3
"""
Copy a CSV-defined image/mask subset into a new dataset root.

The CSV is expected to contain at least:

- ``img_path``: relative path under ``--sim-image-dir`` or an absolute path
- ``mask_path``: relative path under ``--sim-label-dir`` or an absolute path

For example, a row like:

- ``img_path=opt/scene_a/img_001.tif``
- ``mask_path=gt_ss_mask/scene_a/img_001.tif``

will be copied to:

- ``<out_dir>/opt/scene_a/img_001.tif``
- ``<out_dir>/gt_ss_mask/scene_a/img_001.tif``

The script can also copy the CSV manifest itself into ``out_dir``.

If the CSV stores absolute paths, the script preserves the suffix starting from
the requested anchor directory. For example:

- ``/data/tiles/opt/scene_a/img_001.tif`` -> ``<out_dir>/opt/scene_a/img_001.tif``
- ``/data/tiles/gt_ss_mask/scene_a/img_001.tif`` -> ``<out_dir>/gt_ss_mask/scene_a/img_001.tif``
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim-image-dir",
        type=Path,
        default=None,
        help="Root directory that relative img_path values are interpreted under.",
    )
    parser.add_argument(
        "--sim-label-dir",
        type=Path,
        default=None,
        help="Root directory that relative mask_path values are interpreted under.",
    )
    parser.add_argument(
        "--sim-csv",
        type=Path,
        required=True,
        help="CSV manifest containing img_path and mask_path columns.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination root where the subset will be copied.",
    )
    parser.add_argument(
        "--copy-csv",
        action="store_true",
        help="Also copy the CSV manifest into out_dir.",
    )
    parser.add_argument(
        "--img-anchor-dir",
        type=str,
        default="opt",
        help=(
            "Directory name to preserve from for image destinations when img_path "
            "is absolute."
        ),
    )
    parser.add_argument(
        "--mask-anchor-dir",
        type=str,
        default="gt_ss_mask",
        help=(
            "Directory name to preserve from for mask destinations when mask_path "
            "is absolute."
        ),
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default=None,
        help="Optional output filename for the copied CSV. Defaults to the source CSV name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    return parser.parse_args()


def _normalize_path(path_str: str, field_name: str, row_num: int) -> Path:
    rel_path = Path(path_str.strip())
    if not path_str.strip():
        raise ValueError(f"Empty {field_name} at row {row_num}")
    return rel_path


def load_manifest_rows(sim_csv: Path) -> list[tuple[Path, Path]]:
    if not sim_csv.exists():
        raise FileNotFoundError(f"CSV manifest does not exist: {sim_csv}")

    rows: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()

    with sim_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"img_path", "mask_path"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"CSV manifest is missing required columns {sorted(missing)}: {sim_csv}"
            )

        for row_num, row in enumerate(reader, start=2):
            img_path = _normalize_path(row["img_path"], "img_path", row_num)
            mask_path = _normalize_path(row["mask_path"], "mask_path", row_num)
            pair = (img_path, mask_path)
            if pair in seen:
                continue
            seen.add(pair)
            rows.append(pair)

    if not rows:
        raise ValueError(f"No usable rows found in CSV manifest: {sim_csv}")

    return rows


def resolve_source_path(
    raw_path: Path,
    source_root: Path | None,
    field_name: str,
) -> Path:
    if raw_path.is_absolute():
        src_path = raw_path.resolve()
    else:
        if source_root is None:
            raise ValueError(
                f"{field_name} is relative but no corresponding source root was provided: {raw_path}"
            )
        src_path = (source_root / raw_path).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Missing source file: {src_path}")

    return src_path


def destination_relpath(
    raw_path: Path,
    src_path: Path,
    source_root: Path | None,
    anchor_dir: str,
    field_name: str,
) -> Path:
    if raw_path.is_absolute():
        parts = list(src_path.parts)
        if anchor_dir not in parts:
            raise ValueError(
                f"Absolute {field_name} does not contain anchor dir '{anchor_dir}': {src_path}"
            )
        anchor_idx = parts.index(anchor_dir)
        return Path(*parts[anchor_idx:])

    rel_path = raw_path
    if source_root is not None:
        try:
            src_path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                f"Resolved {field_name} is outside its source root: {src_path}"
            ) from exc

    return rel_path


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

    sim_image_dir = args.sim_image_dir.resolve() if args.sim_image_dir is not None else None
    sim_label_dir = args.sim_label_dir.resolve() if args.sim_label_dir is not None else None
    sim_csv = args.sim_csv.resolve()
    out_dir = args.out_dir.resolve()

    rows = load_manifest_rows(sim_csv)

    copied_images = 0
    skipped_images = 0
    copied_masks = 0
    skipped_masks = 0

    for img_raw, mask_raw in rows:
        img_src = resolve_source_path(img_raw, sim_image_dir, "img_path")
        mask_src = resolve_source_path(mask_raw, sim_label_dir, "mask_path")

        img_rel = destination_relpath(
            raw_path=img_raw,
            src_path=img_src,
            source_root=sim_image_dir,
            anchor_dir=args.img_anchor_dir,
            field_name="img_path",
        )
        mask_rel = destination_relpath(
            raw_path=mask_raw,
            src_path=mask_src,
            source_root=sim_label_dir,
            anchor_dir=args.mask_anchor_dir,
            field_name="mask_path",
        )

        img_dst = out_dir / img_rel
        mask_dst = out_dir / mask_rel

        img_result = copy_file(img_src, img_dst, overwrite=args.overwrite, dry_run=args.dry_run)
        if img_result == "copied":
            copied_images += 1
        else:
            skipped_images += 1

        mask_result = copy_file(mask_src, mask_dst, overwrite=args.overwrite, dry_run=args.dry_run)
        if mask_result == "copied":
            copied_masks += 1
        else:
            skipped_masks += 1

    copied_csv = False
    if args.copy_csv:
        csv_name = args.csv_name or sim_csv.name
        csv_dst = out_dir / csv_name
        copy_file(sim_csv, csv_dst, overwrite=args.overwrite, dry_run=args.dry_run)
        copied_csv = True

    print(f"CSV manifest: {sim_csv}")
    print(f"Sim image root: {sim_image_dir}")
    print(f"Sim label root: {sim_label_dir}")
    print(f"Output root: {out_dir}")
    print(f"Rows copied: {len(rows)}")
    print(f"Copied image files: {copied_images}")
    print(f"Skipped existing image files: {skipped_images}")
    print(f"Copied mask files: {copied_masks}")
    print(f"Skipped existing mask files: {skipped_masks}")
    print(f"Copied CSV manifest: {copied_csv}")


if __name__ == "__main__":
    main()
