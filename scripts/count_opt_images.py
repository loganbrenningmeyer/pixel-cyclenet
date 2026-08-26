#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def find_opt_dirs(root_dir: Path, rgb_parent_dir: str) -> list[Path]:
    opt_dirs: list[Path] = []
    seen: set[Path] = set()

    for path in sorted(root_dir.rglob(rgb_parent_dir)):
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        opt_dirs.append(resolved)

    return opt_dirs


def count_images(opt_dirs: list[Path]) -> int:
    total = 0

    for opt_dir in opt_dirs:
        for path in opt_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            total += 1

    return total


def main() -> None:
    root_dir = Path("/path/to/data/root")  # Root directory to scan for opt/ folders.
    rgb_parent_dir = "opt"  # Immediate folder name expected to contain the RGB images.

    if root_dir == Path("/path/to/data/root"):
        raise ValueError("Set root_dir in main() before running this script.")

    root_dir = root_dir.resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise ValueError(f"Root directory is not a directory: {root_dir}")

    opt_dirs = find_opt_dirs(root_dir=root_dir, rgb_parent_dir=rgb_parent_dir)
    if not opt_dirs:
        raise FileNotFoundError(f"No '{rgb_parent_dir}' directories found under {root_dir}")

    image_count = count_images(opt_dirs)

    print(f"Root directory: {root_dir}")
    print(f"Found {len(opt_dirs)} '{rgb_parent_dir}' directories")
    print(f"Total image count under '{rgb_parent_dir}/': {image_count}")


if __name__ == "__main__":
    main()
