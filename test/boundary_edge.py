import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cyclenet.eval.boundary_edge_align import (  # noqa: E402
    boundary_edge_alignment_stats,
    compute_mask_boundary,
    dilate_binary_mask,
    load_mask,
    load_rgb,
    pair_translated_with_masks,
    sobel_edge_magnitude,
)


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


def tensor_rgb_to_numpy(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return arr


def colorize_mask(mask: torch.Tensor, ignore_label: int | None) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy()
    color = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)

    for class_id, (_, rgb) in CLASS_INFO.items():
        color[mask_np == class_id] = np.array(rgb, dtype=np.uint8)

    if ignore_label is not None:
        color[mask_np == ignore_label] = np.array((255, 0, 255), dtype=np.uint8)

    return color


def binary_mask_to_rgb(mask: torch.Tensor, rgb: tuple[int, int, int]) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(bool)
    out = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    out[mask_np] = np.array(rgb, dtype=np.uint8)
    return out


def heatmap_from_tensor(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy().astype(np.float32)
    arr = arr - float(arr.min())
    denom = float(arr.max())
    if denom > 0.0:
        arr = arr / denom
    arr_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(arr_u8, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_mask(
    img: np.ndarray,
    mask: torch.Tensor,
    rgb: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(bool)
    out = img.astype(np.float32).copy()
    color = np.array(rgb, dtype=np.float32)
    out[mask_np] = ((1.0 - alpha) * out[mask_np]) + (alpha * color)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        title,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def add_footer(img: np.ndarray, lines: list[str]) -> np.ndarray:
    footer_h = 76
    footer = np.zeros((footer_h, img.shape[1], 3), dtype=np.uint8)
    y = 18
    for line in lines:
        cv2.putText(
            footer,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return np.concatenate([img, footer], axis=0)


def pad_to_same_height(images: list[np.ndarray]) -> list[np.ndarray]:
    max_h = max(img.shape[0] for img in images)
    padded: list[np.ndarray] = []
    for img in images:
        if img.shape[0] == max_h:
            padded.append(img)
            continue
        pad_h = max_h - img.shape[0]
        pad = np.zeros((pad_h, img.shape[1], 3), dtype=np.uint8)
        padded.append(np.concatenate([img, pad], axis=0))
    return padded


def make_row(images: list[np.ndarray]) -> np.ndarray:
    images = pad_to_same_height(images)
    return np.concatenate(images, axis=1)


def pad_to_same_width(images: list[np.ndarray]) -> list[np.ndarray]:
    max_w = max(img.shape[1] for img in images)
    padded: list[np.ndarray] = []
    for img in images:
        if img.shape[1] == max_w:
            padded.append(img)
            continue
        pad_w = max_w - img.shape[1]
        pad = np.zeros((img.shape[0], pad_w, 3), dtype=np.uint8)
        padded.append(np.concatenate([img, pad], axis=1))
    return padded


def build_sample_panel(
    translated_img: torch.Tensor,
    mask: torch.Tensor,
    boundary_radius: int,
    context_radius: int,
    ignore_label: int | None,
) -> tuple[np.ndarray, dict[str, float]]:
    boundary = compute_mask_boundary(mask, ignore_label=ignore_label)
    boundary_band = dilate_binary_mask(boundary, radius=boundary_radius)
    context_outer = dilate_binary_mask(boundary, radius=context_radius)
    context_band = context_outer & (~boundary_band)
    ignored = mask == ignore_label if ignore_label is not None else torch.zeros_like(mask, dtype=torch.bool)
    sobel_mag = sobel_edge_magnitude(translated_img)
    stats = boundary_edge_alignment_stats(
        translated_img=translated_img,
        mask=mask,
        boundary_radius=boundary_radius,
        context_radius=context_radius,
        ignore_label=ignore_label,
    )

    rgb = tensor_rgb_to_numpy(translated_img)
    mask_color = colorize_mask(mask, ignore_label=ignore_label)
    ignored_rgb = binary_mask_to_rgb(ignored, rgb=(255, 0, 255))
    boundary_rgb = binary_mask_to_rgb(boundary, rgb=(255, 255, 255))
    boundary_band_rgb = binary_mask_to_rgb(boundary_band, rgb=(0, 255, 0))
    context_outer_rgb = binary_mask_to_rgb(context_outer, rgb=(255, 165, 0))
    context_band_rgb = binary_mask_to_rgb(context_band, rgb=(0, 200, 255))
    sobel_heatmap = heatmap_from_tensor(sobel_mag)

    overlay_boundary = overlay_mask(rgb, boundary_band, rgb=(0, 255, 0), alpha=0.45)
    overlay_both = overlay_mask(overlay_boundary, context_band, rgb=(0, 200, 255), alpha=0.35)
    overlay_ignored = overlay_mask(rgb, ignored, rgb=(255, 0, 255), alpha=0.55)

    row1 = make_row(
        [
            add_title(rgb, "Translated RGB"),
            add_title(mask_color, "Mask (ignore=magenta)"),
            add_title(ignored_rgb, "Ignored Pixels"),
            add_title(boundary_rgb, "Raw Boundary Pixels"),
        ]
    )
    row2 = make_row(
        [
            add_title(boundary_band_rgb, f"Boundary Band r={boundary_radius}"),
            add_title(context_outer_rgb, f"Outer Dilation r={context_radius}"),
            add_title(context_band_rgb, "Context Band"),
            add_title(sobel_heatmap, "Sobel Magnitude"),
        ]
    )
    row3 = make_row(
        [
            add_title(overlay_boundary, "RGB + Boundary Band"),
            add_title(overlay_both, "RGB + Boundary/Context"),
            add_title(overlay_ignored, "RGB + Ignored Pixels"),
        ]
    )

    footer_lines = [
        (
            f"boundary_pixels={stats['boundary_pixels']}  "
            f"boundary_band_pixels={stats['boundary_band_pixels']}  "
            f"context_band_pixels={stats['context_band_pixels']}"
        ),
        (
            f"boundary_edge_mean={stats['boundary_edge_mean']:.5f}  "
            f"context_edge_mean={stats['context_edge_mean']:.5f}"
        ),
        (
            f"boundary_edge_ratio={stats['boundary_edge_ratio']:.5f}  "
            f"inverse_ratio={stats['boundary_edge_inverse_ratio']:.5f}  "
            f"contrast={stats['boundary_edge_contrast']:.5f}"
        ),
        (
            "Colors: raw boundary=white, boundary band=green, "
            "outer dilation=orange, context band=cyan, ignored=magenta"
        ),
    ]
    row1, row2, row3 = pad_to_same_width([row1, row2, row3])
    panel = np.concatenate([row1, row2, row3], axis=0)
    panel = add_footer(panel, footer_lines)
    return panel, stats


def main() -> None:
    # Directory containing one translated candidate set, for example `.../step-*/strength-*/cfg-*`.
    translated_dir = Path(
        "/cgi/data/nvesd/workspaces/logan/data/eval/cyclenet/remote_sensing/project_translated/example_candidate"
    )
    # Root directory where the corresponding source masks live.
    mask_root = Path(
        "/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/sim_subset"
    )
    # Output directory for the saved visualization panels.
    out_dir = Path("./boundary_edge_debug")
    # Number of random image/mask pairs to visualize.
    num_samples = 8
    # Random seed so repeated runs pick the same sample subset.
    random_seed = 42
    # Spatial size used to match the metric's resized image/mask computation.
    image_size = 256
    # Boundary-band dilation radius in pixels.
    boundary_radius = 2
    # Outer dilation radius in pixels used to form the context ring.
    context_radius = 5
    # Optional required parent directory name for mask discovery.
    mask_parent_dir = "gt_ss_mask"
    # Mask label to ignore when constructing class boundaries.
    ignore_label = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pair_translated_with_masks(
        translated_dir=translated_dir,
        mask_root=mask_root,
        mask_parent_dir=mask_parent_dir,
    )
    if not pairs:
        raise ValueError("No translated/mask pairs were found.")

    rng = random.Random(random_seed)
    sample_count = min(num_samples, len(pairs))
    chosen_pairs = rng.sample(pairs, sample_count)

    print(f"Found {len(pairs)} total translated/mask pairs")
    print(f"Saving {sample_count} random boundary-edge panels to: {out_dir}")

    for translated_path, mask_path in chosen_pairs:
        translated_img = load_rgb(translated_path, image_size=image_size)
        mask = load_mask(mask_path, image_size=image_size)

        try:
            panel, stats = build_sample_panel(
                translated_img=translated_img,
                mask=mask,
                boundary_radius=boundary_radius,
                context_radius=context_radius,
                ignore_label=ignore_label,
            )
        except ValueError as exc:
            print(f"Skipping {translated_path.name}: {exc}")
            continue

        out_path = out_dir / f"{translated_path.stem}_boundary_edge.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(
            f"{translated_path.name}: "
            f"ratio={stats['boundary_edge_ratio']:.5f}, "
            f"inverse_ratio={stats['boundary_edge_inverse_ratio']:.5f}, "
            f"contrast={stats['boundary_edge_contrast']:.5f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
