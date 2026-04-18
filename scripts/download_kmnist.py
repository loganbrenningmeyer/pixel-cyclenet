#!/usr/bin/env python3
"""
Download USPS with torchvision and export it as flat PNG files under:

    data/usps/train
    data/usps/test

This matches the simple image-only directory layout already used in this repo
for MNIST and Fashion-MNIST.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import USPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download USPS and write train/test images to data/usps."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/usps"),
        help="Output dataset root. Train/test folders will be created under this path.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/_torchvision"),
        help="Root directory for torchvision's downloaded USPS archives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing PNG files in the output train/test directories.",
    )
    return parser.parse_args()


def export_split(dataset: USPS, split_dir: Path, overwrite: bool) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    num_digits = max(6, len(str(len(dataset) - 1)))

    for idx, (img, _label) in enumerate(dataset):
        out_path = split_dir / f"{idx:0{num_digits}d}.png"
        if out_path.exists() and not overwrite:
            continue
        img.save(out_path)


def main() -> None:
    args = parse_args()

    train_ds = USPS(root=args.raw_root, train=True, download=True)
    test_ds = USPS(root=args.raw_root, train=False, download=True)

    export_split(train_ds, args.out_dir / "train", overwrite=args.overwrite)
    export_split(test_ds, args.out_dir / "test", overwrite=args.overwrite)

    print(f"Wrote USPS train split to {args.out_dir / 'train'}")
    print(f"Wrote USPS test split to {args.out_dir / 'test'}")


if __name__ == "__main__":
    main()
