import argparse
import random
from pathlib import Path

import cv2
import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="label_overlays",
        help="Directory to save overlay panels.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay opacity in [0, 1].",
    )
    return parser.parse_args()


def collect_files(data_dir: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VALID_EXTS:
            continue
        files[path.name] = path
    return files


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


def colorize_mask(mask: np.ndarray, ignore_value: int | None) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, (_, rgb) in CLASS_INFO.items():
        if ignore_value is not None and class_id == ignore_value:
            continue
        color[mask == class_id] = np.array(rgb, dtype=np.uint8)
    return color


def overlay_mask(img: np.ndarray, mask: np.ndarray, ignore_value: int, alpha: float) -> np.ndarray:
    color_mask = colorize_mask(mask, ignore_value=ignore_value)
    valid = np.zeros(mask.shape, dtype=bool)
    for class_id in CLASS_INFO:
        if class_id == ignore_value:
            continue
        valid |= mask == class_id

    overlay = img.copy().astype(np.float32)
    overlay[valid] = (
        (1.0 - alpha) * img[valid].astype(np.float32)
        + alpha * color_mask[valid].astype(np.float32)
    )
    return overlay.clip(0, 255).astype(np.uint8)


def draw_label_legend(panel: np.ndarray, ignore_value: int) -> np.ndarray:
    out = panel.copy()
    y = 22
    cv2.putText(
        out,
        f"ignore={ignore_value}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += 18
    for class_id, (label, rgb) in CLASS_INFO.items():
        status = "ignored" if class_id == ignore_value else label
        cv2.rectangle(out, (10, y - 10), (24, y + 4), rgb, thickness=-1)
        cv2.putText(
            out,
            f"{class_id}: {status}",
            (30, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return out


def build_panel(img: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay_ignore_0 = overlay_mask(img, mask, ignore_value=0, alpha=alpha)
    overlay_ignore_8 = overlay_mask(img, mask, ignore_value=8, alpha=alpha)

    left = img.copy()
    cv2.putText(
        left,
        "RGB",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    mid = draw_label_legend(overlay_ignore_0, ignore_value=0)
    right = draw_label_legend(overlay_ignore_8, ignore_value=8)

    return np.concatenate([left, mid, right], axis=1)


def main() -> None:
    args = parse_args()

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    img_files = collect_files(img_dir)
    mask_files = collect_files(mask_dir)
    matching_names = sorted(set(img_files) & set(mask_files))

    if not matching_names:
        raise ValueError(
            f"No matching filenames found between {img_dir} and {mask_dir}"
        )

    n = min(args.n, len(matching_names))
    chosen_names = rng.sample(matching_names, n)

    print(f"Found {len(matching_names)} matching image/mask pairs")
    print(f"Saving {n} random overlay panels to: {out_dir}")

    for name in chosen_names:
        img_path = img_files[name]
        mask_path = mask_files[name]

        img = load_rgb(img_path)
        mask = load_mask(mask_path)
        mask = resize_mask_if_needed(mask, img.shape[:2])

        uniques = np.unique(mask)
        print(f"{name}: unique mask values = {uniques}")

        panel = build_panel(img, mask, alpha=args.alpha)
        out_path = out_dir / f"{Path(name).stem}_overlay.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    print("Done.")


if __name__ == "__main__":
    main()
