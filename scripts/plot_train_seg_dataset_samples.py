#!/usr/bin/env python3
"""
Sample sim/real image-mask pairs from a CycleNet segmentation config and write
contact sheets showing RGB images beside their segmentation overlays.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


CLASS_INFO = {
    1: ("Bareland", (166, 97, 26)),
    2: ("Rangeland", (223, 194, 125)),
    3: ("Developed Space", (128, 128, 128)),
    4: ("Road", (255, 255, 255)),
    5: ("Trees", (27, 158, 119)),
    6: ("Water", (49, 130, 189)),
    7: ("Agriculture land", (255, 211, 0)),
    8: ("Buildings", (215, 48, 39)),
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
IGNORE_LABEL = 0
SAMPLES_PER_GRID = 32
GRID_ROWS = 8
PAIR_COLS_PER_ROW = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot random sim/real RGB samples and segmentation overlays from a training config."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cyclenet/train_cyclenet_seg.yaml"),
        help="Path to the training config.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset_seg_sample_grids"),
        help="Directory for output contact sheets.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of random pairs to sample per domain.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Segmentation overlay alpha.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampling seed. Defaults to config.run.seed or 42.",
    )
    return parser.parse_args()


def cfg_select(config: DictConfig, key: str, default=None):
    value = OmegaConf.select(config, key)
    return default if value is None else value


def load_config(path: Path) -> DictConfig:
    return OmegaConf.load(path)


def as_parent_dir_set(value) -> set[str]:
    if isinstance(value, str):
        return {value}
    return {str(v) for v in value}


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def resize_mask_if_needed(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, (_label, rgb) in CLASS_INFO.items():
        color[mask == class_id] = np.array(rgb, dtype=np.uint8)
    return color


def overlay_mask(img: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    color_mask = colorize_mask(mask)
    valid = mask != IGNORE_LABEL

    overlay = img.copy().astype(np.float32)
    overlay[valid] = (
        (1.0 - alpha) * img[valid].astype(np.float32)
        + alpha * color_mask[valid].astype(np.float32)
    )
    return overlay.clip(0, 255).astype(np.uint8)


def draw_title_bar(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def resolve_matching_file(parent: Path, stem: str) -> Path:
    matches = [
        path
        for path in sorted(parent.iterdir())
        if path.is_file() and path.stem == stem and path.suffix.lower() in VALID_EXTS
    ]
    if not matches:
        raise FileNotFoundError(f"No file with stem '{stem}' found under {parent}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files with stem '{stem}' found under {parent}: {matches}")
    return matches[0]


def resolve_mask_path(rgb_path: Path, label_parent_dir: str) -> Path:
    label_parent = rgb_path.parent.parent / label_parent_dir
    direct_path = label_parent / rgb_path.name
    if direct_path.exists():
        return direct_path
    return resolve_matching_file(label_parent, rgb_path.stem)


def collect_rgb_mask_pairs(
    root: Path,
    rgb_parent_dirs: set[str],
    label_parent_dir: str,
) -> list[tuple[Path, Path]]:
    pairs = []
    for rgb_path in sorted(root.rglob("*")):
        if not rgb_path.is_file():
            continue
        if rgb_path.suffix.lower() not in VALID_EXTS:
            continue
        if rgb_path.parent.name not in rgb_parent_dirs:
            continue
        mask_path = resolve_mask_path(rgb_path, label_parent_dir)
        pairs.append((rgb_path, mask_path))
    return pairs


def build_pair_panel(rgb_path: Path, mask_path: Path, alpha: float) -> np.ndarray:
    rgb = load_rgb(rgb_path)
    mask = resize_mask_if_needed(load_mask(mask_path), rgb.shape[:2])
    overlay = overlay_mask(rgb, mask, alpha=alpha)

    label = rgb_path.parent.parent.name
    left = draw_title_bar(rgb, f"{label} | RGB")
    right = draw_title_bar(overlay, f"{label} | Mask")
    return np.concatenate([left, right], axis=1)


def make_contact_sheet(panels: list[np.ndarray]) -> np.ndarray:
    if not panels:
        raise ValueError("No panels provided for contact sheet.")

    panel_h, panel_w, _ = panels[0].shape
    blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    rows = []
    total_slots = GRID_ROWS * PAIR_COLS_PER_ROW
    padded = panels + [blank] * max(0, total_slots - len(panels))
    for row_idx in range(GRID_ROWS):
        row_panels = padded[row_idx * PAIR_COLS_PER_ROW : (row_idx + 1) * PAIR_COLS_PER_ROW]
        rows.append(np.concatenate(row_panels, axis=1))
    return np.concatenate(rows, axis=0)


def save_domain_grids(
    domain_name: str,
    pairs: list[tuple[Path, Path]],
    out_dir: Path,
    alpha: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    num_grids = math.ceil(len(pairs) / SAMPLES_PER_GRID)

    for grid_idx in tqdm(range(num_grids), desc=f"{domain_name} grids", unit="grid"):
        chunk = pairs[grid_idx * SAMPLES_PER_GRID : (grid_idx + 1) * SAMPLES_PER_GRID]
        panels = [build_pair_panel(rgb_path, mask_path, alpha=alpha) for rgb_path, mask_path in chunk]
        sheet = make_contact_sheet(panels)
        out_path = out_dir / f"{domain_name}_grid_{grid_idx + 1:02d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def sample_pairs(
    pairs: list[tuple[Path, Path]],
    num_samples: int,
    seed: int,
) -> list[tuple[Path, Path]]:
    if not pairs:
        raise RuntimeError("No RGB/mask pairs found.")
    if num_samples >= len(pairs):
        return pairs

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pairs)), num_samples))
    return [pairs[idx] for idx in indices]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(args.seed if args.seed is not None else cfg_select(config, "run.seed", 42))
    rgb_parent_dirs = as_parent_dir_set(cfg_select(config, "data.rgb_parent_dirs", ["opt"]))
    label_parent_dir = str(cfg_select(config, "data.label_parent_dir", "gt_ss_mask"))

    domains = {
        "sim": Path(cfg_select(config, "data.src_dir")),
        "real": Path(cfg_select(config, "data.tgt_dir")),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, args.out_dir / "config_snapshot.yaml")

    print(f"Config: {args.config}")
    print(f"Output: {args.out_dir}")
    print(f"Seed: {seed}")
    print(f"RGB parent dirs: {sorted(rgb_parent_dirs)}")
    print(f"Label parent dir: {label_parent_dir}")

    for domain_name, root in domains.items():
        print(f"[{domain_name}] collecting pairs from {root}")
        pairs = collect_rgb_mask_pairs(root, rgb_parent_dirs, label_parent_dir)
        print(f"[{domain_name}] found {len(pairs)} RGB/mask pairs")

        sampled_pairs = sample_pairs(
            pairs=pairs,
            num_samples=args.num_samples,
            seed=seed if domain_name == "sim" else seed + 1,
        )
        print(f"[{domain_name}] writing {len(sampled_pairs)} samples")

        save_domain_grids(
            domain_name=domain_name,
            pairs=sampled_pairs,
            out_dir=args.out_dir,
            alpha=args.alpha,
        )

    print("Done.")


if __name__ == "__main__":
    main()
