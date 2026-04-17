#!/usr/bin/env python3
"""
Generate deterministic sim-image subset manifests for translation sweeps.

The balancing logic is hierarchical:

1. Balance equally across terrain type: ``grid`` vs ``terrain``.
2. Within each terrain type, balance equally across condition:
   ``terrain_type x gsd x height``.
3. Within each condition, balance equally across folder/version so duplicated
   variants like ``mid_v1`` / ``mid_v2`` do not get extra weight.

Every stage uses capped redistribution:
if a group cannot satisfy its nominal quota, the remaining quota is
redistributed across the non-exhausted siblings.

By default the script creates two deterministic manifests of size 39,598 to
match the real-image count discussed for the translation sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
DEFAULT_RGB_PARENT_DIRS = ("opt",)
FOLDER_PATTERN = re.compile(
    r"^(?P<terrain_type>grid|terrain)_(?P<gsd>g005|g05|g1)_(?P<height>low|mid|high)_(?P<version>v\d+)$"
)


@dataclass(frozen=True)
class ImageRecord:
    abs_path: str
    rel_path: str
    folder: str
    terrain_type: str
    gsd: str
    height: str
    version: str
    condition: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sim-root",
        type=Path,
        required=True,
        help="Root directory containing the sim dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/subsets/sim_translation"),
        help="Directory where manifests and summaries will be written.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=39_598,
        help="Number of images per subset. Default matches the real-image count.",
    )
    parser.add_argument(
        "--num-subsets",
        type=int,
        default=2,
        help="How many deterministic subsets to generate.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base seed; subset i uses base_seed + i.",
    )
    parser.add_argument(
        "--rgb-parent-dir",
        dest="rgb_parent_dirs",
        action="append",
        default=list(DEFAULT_RGB_PARENT_DIRS),
        help=(
            "Only include images whose immediate parent directory matches this "
            "name. Can be passed multiple times. Default: opt"
        ),
    )
    parser.add_argument(
        "--extension",
        dest="extensions",
        action="append",
        default=list(DEFAULT_EXTENSIONS),
        help="Allowed image extension. Can be passed multiple times.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths to the .txt manifests instead of paths relative to sim-root.",
    )
    return parser.parse_args()


def normalize_extensions(extensions: Iterable[str]) -> set[str]:
    normalized = set()
    for extension in extensions:
        ext = extension.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.add(ext)
    return normalized


def find_folder_name(path: Path, sim_root: Path, rgb_parent_dirs: set[str]) -> str | None:
    if path.parent.name in rgb_parent_dirs:
        candidate = path.parent.parent
        if candidate != path.parent and FOLDER_PATTERN.match(candidate.name):
            return candidate.name

    for ancestor in path.parents:
        if ancestor == sim_root.parent:
            break
        if ancestor == sim_root:
            if FOLDER_PATTERN.match(ancestor.name):
                return ancestor.name
            break
        if FOLDER_PATTERN.match(ancestor.name):
            return ancestor.name

    return None


def discover_images(
    sim_root: Path,
    rgb_parent_dirs: set[str],
    extensions: set[str],
) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    for path in sorted(sim_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if rgb_parent_dirs and path.parent.name not in rgb_parent_dirs:
            continue

        folder_name = find_folder_name(path, sim_root, rgb_parent_dirs)
        if folder_name is None:
            raise ValueError(f"Could not infer stratification folder for image: {path}")

        match = FOLDER_PATTERN.match(folder_name)
        if match is None:
            raise ValueError(f"Unexpected folder name format for image {path}: {folder_name}")

        terrain_type = match.group("terrain_type")
        gsd = match.group("gsd")
        height = match.group("height")
        version = match.group("version")
        condition = f"{terrain_type}_{gsd}_{height}"

        records.append(
            ImageRecord(
                abs_path=str(path.resolve()),
                rel_path=str(path.relative_to(sim_root)),
                folder=folder_name,
                terrain_type=terrain_type,
                gsd=gsd,
                height=height,
                version=version,
                condition=condition,
            )
        )

    if not records:
        raise ValueError(f"No matching images found under {sim_root}")

    return records


def allocate_even_with_caps(capacities: dict[str, int], target: int) -> dict[str, int]:
    total_capacity = sum(capacities.values())
    if target < 0:
        raise ValueError(f"Target must be non-negative, got {target}")
    if target > total_capacity:
        raise ValueError(f"Target {target} exceeds capacity {total_capacity}")

    allocations = {key: 0 for key in capacities}
    active = sorted(capacities)
    remaining = target

    while remaining > 0 and active:
        num_active = len(active)
        base = remaining // num_active
        extra = remaining % num_active

        progressed = False
        next_active: list[str] = []

        for index, key in enumerate(active):
            room = capacities[key] - allocations[key]
            requested = base + (1 if index < extra else 0)
            if requested <= 0:
                requested = 1

            taken = min(requested, room, remaining)
            if taken > 0:
                allocations[key] += taken
                remaining -= taken
                progressed = True

            if allocations[key] < capacities[key]:
                next_active.append(key)

            if remaining == 0:
                next_active.extend(active[index + 1 :])
                break

        if not progressed:
            raise RuntimeError("Allocation stalled before satisfying target.")

        active = next_active

    if remaining != 0:
        raise RuntimeError(f"Allocation ended with {remaining} items unassigned.")

    return allocations


def allocate_hierarchical(records: list[ImageRecord], target_size: int) -> dict[str, int]:
    folder_counts: dict[str, int] = defaultdict(int)
    condition_to_folders: dict[str, set[str]] = defaultdict(set)
    terrain_to_conditions: dict[str, set[str]] = defaultdict(set)

    for record in records:
        folder_counts[record.folder] += 1
        condition_to_folders[record.condition].add(record.folder)
        terrain_to_conditions[record.terrain_type].add(record.condition)

    terrain_capacities = {
        terrain_type: sum(folder_counts[folder] for condition in sorted(conditions) for folder in sorted(condition_to_folders[condition]))
        for terrain_type, conditions in terrain_to_conditions.items()
    }
    terrain_allocations = allocate_even_with_caps(terrain_capacities, target_size)

    folder_allocations: dict[str, int] = {}
    for terrain_type, terrain_quota in terrain_allocations.items():
        conditions = sorted(terrain_to_conditions[terrain_type])
        condition_capacities = {
            condition: sum(folder_counts[folder] for folder in sorted(condition_to_folders[condition]))
            for condition in conditions
        }
        condition_allocations = allocate_even_with_caps(condition_capacities, terrain_quota)

        for condition, condition_quota in condition_allocations.items():
            folders = sorted(condition_to_folders[condition])
            version_capacities = {folder: folder_counts[folder] for folder in folders}
            version_allocations = allocate_even_with_caps(version_capacities, condition_quota)
            folder_allocations.update(version_allocations)

    total_allocated = sum(folder_allocations.values())
    if total_allocated != target_size:
        raise RuntimeError(f"Allocated {total_allocated} images, expected {target_size}")

    return folder_allocations


def sample_subset(
    records_by_folder: dict[str, list[ImageRecord]],
    folder_allocations: dict[str, int],
    seed: int,
) -> list[ImageRecord]:
    rng = random.Random(seed)
    subset: list[ImageRecord] = []

    for folder in sorted(folder_allocations):
        quota = folder_allocations[folder]
        candidates = list(records_by_folder[folder])
        rng.shuffle(candidates)
        subset.extend(candidates[:quota])

    subset.sort(key=lambda record: record.rel_path)
    return subset


def summarize_subset(records: list[ImageRecord]) -> dict[str, dict[str, int] | int]:
    by_folder: dict[str, int] = defaultdict(int)
    by_condition: dict[str, int] = defaultdict(int)
    by_terrain: dict[str, int] = defaultdict(int)

    for record in records:
        by_folder[record.folder] += 1
        by_condition[record.condition] += 1
        by_terrain[record.terrain_type] += 1

    return {
        "total": len(records),
        "by_terrain_type": dict(sorted(by_terrain.items())),
        "by_condition": dict(sorted(by_condition.items())),
        "by_folder": dict(sorted(by_folder.items())),
    }


def write_manifest_txt(records: list[ImageRecord], out_path: Path, absolute_paths: bool) -> None:
    lines = [record.abs_path if absolute_paths else record.rel_path for record in records]
    out_path.write_text("\n".join(lines) + "\n")


def write_manifest_csv(records: list[ImageRecord], out_path: Path) -> None:
    fieldnames = list(asdict(records[0]).keys())
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> None:
    args = parse_args()
    sim_root = args.sim_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_parent_dirs = set(args.rgb_parent_dirs)
    extensions = normalize_extensions(args.extensions)

    records = discover_images(sim_root, rgb_parent_dirs=rgb_parent_dirs, extensions=extensions)
    if args.target_size > len(records):
        raise ValueError(
            f"Requested target size {args.target_size} exceeds discovered image count {len(records)}"
        )

    folder_allocations = allocate_hierarchical(records, args.target_size)

    records_by_folder: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_folder[record.folder].append(record)

    overall_counts = {
        "total_images": len(records),
        "target_size": args.target_size,
        "num_subsets": args.num_subsets,
        "base_seed": args.base_seed,
        "rgb_parent_dirs": sorted(rgb_parent_dirs),
        "extensions": sorted(extensions),
        "source_root": str(sim_root),
        "capacity_by_folder": {
            folder: len(folder_records) for folder, folder_records in sorted(records_by_folder.items())
        },
        "allocation_by_folder": dict(sorted(folder_allocations.items())),
    }
    (out_dir / "subset_generation_summary.json").write_text(json.dumps(overall_counts, indent=2) + "\n")

    for subset_index in range(args.num_subsets):
        seed = args.base_seed + subset_index
        subset_records = sample_subset(records_by_folder, folder_allocations, seed=seed)
        summary = summarize_subset(subset_records)

        stem = f"sim_subset_{args.target_size}_seed{seed}"
        write_manifest_txt(
            subset_records,
            out_dir / f"{stem}.txt",
            absolute_paths=args.absolute_paths,
        )
        write_manifest_csv(subset_records, out_dir / f"{stem}.csv")
        (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        print(f"Wrote subset {subset_index + 1}/{args.num_subsets}: {out_dir / f'{stem}.txt'}")
        print(json.dumps(summary["by_terrain_type"], indent=2))


if __name__ == "__main__":
    main()
