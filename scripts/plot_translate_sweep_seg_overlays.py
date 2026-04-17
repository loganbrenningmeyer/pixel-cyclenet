#!/usr/bin/env python3
"""
Create 2x2 translated/source segmentation overlay panels for translate_sweep runs.

For each translated image under:
    <sweep_dir>/candidates/<candidate>/fake/.../<rgb_parent_dir>/<image>.png

the script writes a panel to:
    <sweep_dir>/candidates/<candidate>/fake/.../masks/<image>_overlay.png

Each panel contains:
    - top-left: translated image
    - top-right: source sim image
    - bottom-left: translated image with source segmentation overlay
    - bottom-right: source sim image with source segmentation overlay
"""

from __future__ import annotations

import argparse
import re
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
LEGEND_WIDTH = 205
CANDIDATE_DIR_RE = re.compile(
    r"^step-(?P<step>\d+)_strength-(?P<strength>\d+(?:\.\d+)?)_cfg-(?P<cfg>\d+(?:\.\d+)?)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create source/translated segmentation overlay grids for translate_sweep outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="A translate_sweep run directory or a parent directory containing multiple sweep runs.",
    )
    parser.add_argument(
        "--label-parent-dir",
        type=str,
        default=None,
        help="Override the segmentation label parent dir. Defaults to config.data.label_parent_dir or gt_ss_mask.",
    )
    parser.add_argument(
        "--rgb-parent-dirs",
        nargs="+",
        default=None,
        help="Override allowed RGB parent dirs. Defaults to config.data.rgb_parent_dirs or ['opt'].",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Segmentation overlay alpha.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing overlay panels.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Only process candidates whose directory name has one of these step values.",
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=None,
        help="Only process candidates whose directory name has one of these noise strengths.",
    )
    parser.add_argument(
        "--cfg-values",
        nargs="+",
        type=float,
        default=None,
        help="Only process candidates whose directory name has one of these cfg values.",
    )
    return parser.parse_args()


def cfg_select(config: DictConfig | None, key: str, default=None):
    if config is None:
        return default
    value = OmegaConf.select(config, key)
    return default if value is None else value


def load_config(config_path: Path) -> DictConfig:
    return OmegaConf.load(config_path)


def discover_sweep_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    sweep_dirs: set[Path] = set()

    if (root / "config.yaml").exists() and (root / "candidates").exists():
        sweep_dirs.add(root)
    else:
        for config_path in root.rglob("config.yaml"):
            sweep_dir = config_path.parent
            if (sweep_dir / "candidates").exists():
                sweep_dirs.add(sweep_dir)

    return sorted(sweep_dirs)


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
    valid = mask != 0

    overlay = img.copy().astype(np.float32)
    overlay[valid] = (
        (1.0 - alpha) * img[valid].astype(np.float32)
        + alpha * color_mask[valid].astype(np.float32)
    )
    return overlay.clip(0, 255).astype(np.uint8)


def draw_title(panel: np.ndarray, title: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def present_class_ids(mask: np.ndarray | None) -> list[int]:
    if mask is None:
        return []
    values = sorted(int(v) for v in np.unique(mask) if int(v) in CLASS_INFO and int(v) != 0)
    return values


def blank_side_strip(panel: np.ndarray) -> np.ndarray:
    strip = np.zeros((panel.shape[0], LEGEND_WIDTH, 3), dtype=np.uint8)
    strip[:] = (18, 18, 18)
    return np.concatenate([panel, strip], axis=1)


def add_legend_strip(panel: np.ndarray, class_ids: list[int]) -> np.ndarray:
    legend = np.zeros((panel.shape[0], LEGEND_WIDTH, 3), dtype=np.uint8)
    legend[:] = (18, 18, 18)

    y = 22
    cv2.putText(
        legend,
        "Segmentation Labels",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    y += 20

    if not class_ids:
        cv2.putText(
            legend,
            "No labeled classes",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return np.concatenate([panel, legend], axis=1)

    for class_id in class_ids:
        label, rgb = CLASS_INFO[class_id]
        cv2.rectangle(legend, (10, y - 10), (24, y + 4), rgb, thickness=-1)
        cv2.putText(
            legend,
            f"{class_id}: {label}",
            (30, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18

    return np.concatenate([panel, legend], axis=1)


def build_panel(
    title: str,
    img: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
    show_legend: bool,
) -> np.ndarray:
    panel = draw_title(img, title)
    if mask is not None:
        overlay = overlay_mask(img, mask, alpha=alpha)
        panel = draw_title(overlay, title)
    if not show_legend:
        return blank_side_strip(panel)
    return add_legend_strip(panel, present_class_ids(mask))


def stack_grid(
    translated_rgb: np.ndarray,
    source_rgb: np.ndarray,
    translated_overlay: np.ndarray,
    source_overlay: np.ndarray,
) -> np.ndarray:
    top = np.concatenate([translated_rgb, source_rgb], axis=1)
    bottom = np.concatenate([translated_overlay, source_overlay], axis=1)
    return np.concatenate([top, bottom], axis=0)


def resolve_matching_file(parent: Path, stem: str) -> Path:
    direct_matches = [path for path in sorted(parent.iterdir()) if path.is_file() and path.stem == stem and path.suffix.lower() in VALID_EXTS]
    if not direct_matches:
        raise FileNotFoundError(f"No file with stem '{stem}' found under {parent}")
    if len(direct_matches) > 1:
        raise RuntimeError(f"Multiple files with stem '{stem}' found under {parent}: {direct_matches}")
    return direct_matches[0]


def resolve_source_rgb_path(fake_path: Path, fake_root: Path, src_root: Path) -> Path:
    rel_path = fake_path.relative_to(fake_root)
    source_parent = src_root / rel_path.parent

    direct_path = source_parent / fake_path.name
    if direct_path.exists():
        return direct_path

    return resolve_matching_file(source_parent, fake_path.stem)


def resolve_mask_path(source_rgb_path: Path, label_parent_dir: str) -> Path:
    label_parent = source_rgb_path.parent.parent / label_parent_dir
    direct_path = label_parent / source_rgb_path.name
    if direct_path.exists():
        return direct_path

    return resolve_matching_file(label_parent, source_rgb_path.stem)


def fake_image_paths(fake_root: Path, rgb_parent_dirs: set[str]) -> list[Path]:
    paths = []
    for path in sorted(fake_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTS:
            continue
        if path.parent.name not in rgb_parent_dirs:
            continue
        paths.append(path)
    return paths


def output_path_for(fake_path: Path) -> Path:
    masks_dir = fake_path.parent.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    return masks_dir / f"{fake_path.stem}_overlay.png"


def parse_candidate_dir_name(candidate_name: str) -> tuple[int, float, float] | None:
    match = CANDIDATE_DIR_RE.match(candidate_name)
    if match is None:
        return None
    return (
        int(match.group("step")),
        float(match.group("strength")),
        float(match.group("cfg")),
    )


def matches_filter(value: float, allowed: set[float] | None, tol: float = 1e-6) -> bool:
    if allowed is None:
        return True
    return any(abs(value - candidate) <= tol for candidate in allowed)


def filter_candidate_dirs(
    candidate_dirs: list[Path],
    steps: list[int] | None,
    strengths: list[float] | None,
    cfg_values: list[float] | None,
) -> list[Path]:
    allowed_steps = set(steps) if steps is not None else None
    allowed_strengths = set(strengths) if strengths is not None else None
    allowed_cfg_values = set(cfg_values) if cfg_values is not None else None

    filtered = []
    for candidate_dir in candidate_dirs:
        parsed = parse_candidate_dir_name(candidate_dir.name)
        if parsed is None:
            if steps is None and strengths is None and cfg_values is None:
                filtered.append(candidate_dir)
            continue

        step, strength, cfg_value = parsed
        if allowed_steps is not None and step not in allowed_steps:
            continue
        if not matches_filter(strength, allowed_strengths):
            continue
        if not matches_filter(cfg_value, allowed_cfg_values):
            continue
        filtered.append(candidate_dir)

    return filtered


def process_candidate(
    candidate_dir: Path,
    src_root: Path,
    rgb_parent_dirs: set[str],
    label_parent_dir: str,
    alpha: float,
    overwrite: bool,
) -> tuple[int, int]:
    fake_root = candidate_dir / "fake"
    if not fake_root.exists():
        return 0, 0

    fake_paths = fake_image_paths(fake_root, rgb_parent_dirs)
    written = 0
    skipped = 0

    for fake_path in tqdm(
        fake_paths,
        desc=f"{candidate_dir.parent.name}/{candidate_dir.name}",
        unit="img",
        leave=False,
    ):
        out_path = output_path_for(fake_path)
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        source_rgb_path = resolve_source_rgb_path(fake_path, fake_root, src_root)
        mask_path = resolve_mask_path(source_rgb_path, label_parent_dir)

        translated_rgb = load_rgb(fake_path)
        source_rgb = load_rgb(source_rgb_path)
        mask = load_mask(mask_path)

        mask_for_translated = resize_mask_if_needed(mask, translated_rgb.shape[:2])
        mask_for_source = resize_mask_if_needed(mask, source_rgb.shape[:2])

        translated_plain = build_panel("Translated", translated_rgb, None, alpha, show_legend=False)
        source_plain = build_panel("Source Sim", source_rgb, None, alpha, show_legend=False)
        translated_overlay = build_panel(
            "Translated + Seg",
            translated_rgb,
            mask_for_translated,
            alpha,
            show_legend=True,
        )
        source_overlay = build_panel(
            "Source Sim + Seg",
            source_rgb,
            mask_for_source,
            alpha,
            show_legend=True,
        )

        panel = stack_grid(
            translated_rgb=translated_plain,
            source_rgb=source_plain,
            translated_overlay=translated_overlay,
            source_overlay=source_overlay,
        )
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        written += 1

    return written, skipped


def process_sweep_dir(
    sweep_dir: Path,
    label_parent_dir_override: str | None,
    rgb_parent_dirs_override: list[str] | None,
    alpha: float,
    overwrite: bool,
    steps: list[int] | None,
    strengths: list[float] | None,
    cfg_values: list[float] | None,
):
    config = load_config(sweep_dir / "config.yaml")

    src_root = Path(cfg_select(config, "data.src_dir"))
    label_parent_dir = str(
        label_parent_dir_override
        if label_parent_dir_override is not None
        else cfg_select(config, "data.label_parent_dir", "gt_ss_mask")
    )

    rgb_parent_dirs_value = (
        rgb_parent_dirs_override
        if rgb_parent_dirs_override is not None
        else cfg_select(config, "data.rgb_parent_dirs", ["opt"])
    )
    rgb_parent_dirs = {str(value) for value in rgb_parent_dirs_value}

    candidate_dirs = sorted(path for path in (sweep_dir / "candidates").iterdir() if path.is_dir())
    candidate_dirs = filter_candidate_dirs(candidate_dirs, steps, strengths, cfg_values)
    total_written = 0
    total_skipped = 0

    print(f"[{sweep_dir.name}] source root: {src_root}")
    print(f"[{sweep_dir.name}] label parent dir: {label_parent_dir}")
    print(f"[{sweep_dir.name}] rgb parent dirs: {sorted(rgb_parent_dirs)}")
    if steps is not None:
        print(f"[{sweep_dir.name}] step filter: {sorted(steps)}")
    if strengths is not None:
        print(f"[{sweep_dir.name}] strength filter: {sorted(strengths)}")
    if cfg_values is not None:
        print(f"[{sweep_dir.name}] cfg filter: {sorted(cfg_values)}")

    for candidate_dir in tqdm(candidate_dirs, desc=f"{sweep_dir.name} candidates", unit="candidate"):
        written, skipped = process_candidate(
            candidate_dir=candidate_dir,
            src_root=src_root,
            rgb_parent_dirs=rgb_parent_dirs,
            label_parent_dir=label_parent_dir,
            alpha=alpha,
            overwrite=overwrite,
        )
        total_written += written
        total_skipped += skipped
        print(
            f"[{sweep_dir.name}] {candidate_dir.name}: "
            f"wrote {written}, skipped {skipped}"
        )

    print(
        f"[{sweep_dir.name}] done: wrote {total_written}, skipped {total_skipped}"
    )


def main() -> None:
    args = parse_args()
    sweep_dirs = discover_sweep_dirs(args.root)
    if not sweep_dirs:
        raise RuntimeError(f"No translate_sweep run directories found under {args.root}")

    print(f"Found {len(sweep_dirs)} translate_sweep runs under {args.root}")
    for sweep_dir in sweep_dirs:
        process_sweep_dir(
            sweep_dir=sweep_dir,
            label_parent_dir_override=args.label_parent_dir,
            rgb_parent_dirs_override=args.rgb_parent_dirs,
            alpha=args.alpha,
            overwrite=args.overwrite,
            steps=args.steps,
            strengths=args.strengths,
            cfg_values=args.cfg_values,
        )


if __name__ == "__main__":
    main()
