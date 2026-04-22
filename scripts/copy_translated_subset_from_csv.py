#!/usr/bin/env python3
"""
Copy translated images that correspond to a manifest-selected sim subset.

The input CSV is expected to contain a `source_path` column produced by
`scripts/generate_projection_subset.py`. Each `source_path` should point to an
RGB image under the original sim dataset root, for example:

    /path/to/sim_root/data_group/opt/img.tif

The translation scripts preserve that relative layout under the translated
output root, so the matching translated image is expected at:

    <translated_root>/data_group/opt/img.tif

This script copies those translated files into a destination directory while
preserving the same relative structure under `dst_root`.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


def read_source_paths(manifest_csv: Path) -> list[Path]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV does not exist: {manifest_csv}")

    with manifest_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        if "source_path" not in (reader.fieldnames or []):
            raise ValueError(
                f"Manifest CSV must contain a 'source_path' column: {manifest_csv}"
            )

        paths: list[Path] = []
        seen: set[Path] = set()
        for row_num, row in enumerate(reader, start=2):
            raw_path = (row.get("source_path") or "").strip()
            if not raw_path:
                continue

            path = Path(raw_path).resolve()
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)

    if not paths:
        raise ValueError(f"No source paths found in manifest CSV: {manifest_csv}")

    return paths


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
    manifest_csv = Path(
        "/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/sim_projection_subset/subset_manifest.csv"
    ).resolve()
    sim_root = Path(
        "/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/sim_subset"
    ).resolve()
    translated_root = Path(
        "/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/sim_translated_full"
    ).resolve()
    dst_root = Path(
        "/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/sim_translated_projection_subset"
    ).resolve()

    overwrite = False
    dry_run = False

    source_paths = read_source_paths(manifest_csv)
    copied = 0
    skipped = 0
    rows: list[dict[str, str]] = []

    for source_path in source_paths:
        if not source_path.exists():
            raise FileNotFoundError(f"Manifest source path does not exist: {source_path}")

        try:
            rel_path = source_path.relative_to(sim_root)
        except ValueError as exc:
            raise ValueError(
                f"Manifest source path is outside sim_root:\n"
                f"  source_path: {source_path}\n"
                f"  sim_root: {sim_root}"
            ) from exc

        translated_path = translated_root / rel_path
        if not translated_path.exists():
            raise FileNotFoundError(
                f"Missing translated file for manifest entry:\n"
                f"  source_path: {source_path}\n"
                f"  expected translated path: {translated_path}"
            )

        dst_path = dst_root / rel_path
        result = copy_file(
            src_path=translated_path,
            dst_path=dst_path,
            overwrite=overwrite,
            dry_run=dry_run,
        )
        if result == "copied":
            copied += 1
        else:
            skipped += 1

        rows.append(
            {
                "source_path": str(source_path),
                "translated_source_path": str(translated_path),
                "translated_out_path": str(dst_path),
            }
        )

    manifest_out = dst_root / "translated_subset_manifest.csv"
    if not dry_run:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with manifest_out.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_path",
                    "translated_source_path",
                    "translated_out_path",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"Manifest CSV: {manifest_csv}")
    print(f"Sim root: {sim_root}")
    print(f"Translated root: {translated_root}")
    print(f"Destination root: {dst_root}")
    print(f"Listed source images: {len(source_paths)}")
    print(f"Copied translated images: {copied}")
    print(f"Skipped existing translated images: {skipped}")
    if not dry_run:
        print(f"Wrote translated manifest: {manifest_out}")


if __name__ == "__main__":
    main()
