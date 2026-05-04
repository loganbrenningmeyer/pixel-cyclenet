import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_prefixed_float(name: str, prefix: str) -> float:
    if not name.startswith(prefix):
        raise ValueError(f"Expected '{name}' to start with '{prefix}'.")
    return float(name[len(prefix) :])


def parse_step_index(name: str) -> int:
    if not name.startswith("step-"):
        raise ValueError(f"Expected '{name}' to start with 'step-'.")
    return int(name.removeprefix("step-"))


def iter_candidate_dirs(step_dir: str | Path) -> list[tuple[int, float, float, Path]]:
    root = Path(step_dir)
    if not root.exists():
        raise FileNotFoundError(f"step_dir does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"step_dir must be a directory, got: {root}")
    if not root.name.startswith("step-"):
        raise ValueError(f"Expected step_dir name like 'step-*', got '{root.name}'")

    candidates: list[tuple[int, float, float, Path]] = []
    step = parse_step_index(root.name)
    strength_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("strength-")],
        key=lambda path: parse_prefixed_float(path.name, "strength-"),
    )
    for strength_dir in strength_dirs:
        noise_strength = parse_prefixed_float(strength_dir.name, "strength-")
        cfg_dirs = sorted(
            [path for path in strength_dir.iterdir() if path.is_dir() and path.name.startswith("cfg-")],
            key=lambda path: parse_prefixed_float(path.name, "cfg-"),
        )
        for cfg_dir in cfg_dirs:
            cfg_weight = parse_prefixed_float(cfg_dir.name, "cfg-")
            candidates.append((step, noise_strength, cfg_weight, cfg_dir))

    if not candidates:
        raise ValueError(
            f"No strength/cfg directories were found under {root}. "
            "Expected a layout like step-*/strength-*/cfg-*."
        )

    return candidates


def collect_images_by_name(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    images_by_name: dict[str, Path] = {}
    duplicate_names: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        name = path.name
        if name in images_by_name:
            duplicate_names.append(name)
            continue
        images_by_name[name] = path

    if duplicate_names:
        dupes = ", ".join(sorted(set(duplicate_names))[:10])
        raise ValueError(
            f"Found duplicate filenames under {root}. "
            f"Pairing by filename would be ambiguous. Examples: {dupes}"
        )

    if not images_by_name:
        raise ValueError(f"No images found under {root}")

    return images_by_name


def collect_masks_by_name(mask_root: str | Path, mask_parent_dir: str | None = None) -> dict[str, Path]:
    mask_root = Path(mask_root)
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask root does not exist: {mask_root}")

    masks_by_name: dict[str, Path] = {}
    duplicate_names: list[str] = []

    for path in sorted(mask_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if mask_parent_dir is not None and path.parent.name != mask_parent_dir:
            continue

        name = path.name
        if name in masks_by_name:
            duplicate_names.append(name)
            continue
        masks_by_name[name] = path

    if duplicate_names:
        dupes = ", ".join(sorted(set(duplicate_names))[:10])
        raise ValueError(
            f"Found duplicate mask filenames under {mask_root}. "
            f"Pairing by filename would be ambiguous. Examples: {dupes}"
        )

    if not masks_by_name:
        if mask_parent_dir is None:
            raise ValueError(f"No mask files found under {mask_root}")
        raise ValueError(
            f"No mask files found under {mask_root} with parent directory '{mask_parent_dir}'"
        )

    return masks_by_name


def pair_translated_with_masks(
    translated_dir: str | Path,
    mask_root: str | Path,
    mask_parent_dir: str | None = None,
) -> list[tuple[Path, Path]]:
    translated_images = collect_images_by_name(translated_dir)
    mask_images = collect_masks_by_name(mask_root, mask_parent_dir=mask_parent_dir)

    shared_names = sorted(set(translated_images) & set(mask_images))
    if not shared_names:
        raise ValueError(
            "No shared filenames were found between "
            f"{Path(translated_dir)} and {Path(mask_root)}"
        )

    return [(translated_images[name], mask_images[name]) for name in shared_names]


def load_rgb(path: str | Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_mask(path: str | Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
        arr = np.asarray(img, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy(arr)


def compute_mask_boundary(mask: torch.Tensor, ignore_label: int | None = 0) -> torch.Tensor:
    """
    Returns a boolean `(H, W)` boundary map where neighboring class ids differ.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask tensor, got shape {tuple(mask.shape)}")

    boundary = torch.zeros_like(mask, dtype=torch.bool)

    diff_down = mask[1:, :] != mask[:-1, :]
    diff_right = mask[:, 1:] != mask[:, :-1]

    if ignore_label is not None:
        valid_down = (mask[1:, :] != ignore_label) & (mask[:-1, :] != ignore_label)
        valid_right = (mask[:, 1:] != ignore_label) & (mask[:, :-1] != ignore_label)
        diff_down = diff_down & valid_down
        diff_right = diff_right & valid_right

    boundary[1:, :] |= diff_down
    boundary[:-1, :] |= diff_down
    boundary[:, 1:] |= diff_right
    boundary[:, :-1] |= diff_right

    return boundary


def dilate_binary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()

    x = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = 2 * radius + 1
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
    return y[0, 0] > 0


def sobel_edge_magnitude(img: torch.Tensor) -> torch.Tensor:
    """
    Computes Sobel edge magnitude for an RGB tensor `(3, H, W)` in `[0, 1]`.
    """
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor of shape (3, H, W), got {tuple(img.shape)}")

    gray = (
        0.2989 * img[0:1] +
        0.5870 * img[1:2] +
        0.1140 * img[2:3]
    ).unsqueeze(0)

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=gray.dtype,
    ).unsqueeze(0)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=gray.dtype,
    ).unsqueeze(0)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-12)
    return mag[0, 0]


def boundary_edge_alignment_stats(
    translated_img: torch.Tensor,
    mask: torch.Tensor,
    boundary_radius: int,
    context_radius: int,
    ignore_label: int | None = 0,
    eps: float = 1e-6,
) -> dict[str, float]:
    boundary = compute_mask_boundary(mask, ignore_label=ignore_label)
    boundary_band = dilate_binary_mask(boundary, radius=boundary_radius)
    context_outer = dilate_binary_mask(boundary, radius=context_radius)
    context_band = context_outer & (~boundary_band)

    edge_mag = sobel_edge_magnitude(translated_img)

    if boundary_band.sum().item() == 0:
        raise ValueError("Boundary band is empty; check mask contents and ignore_label.")
    if context_band.sum().item() == 0:
        raise ValueError("Context band is empty; increase context_radius.")

    boundary_mean = float(edge_mag[boundary_band].mean().item())
    context_mean = float(edge_mag[context_band].mean().item())
    ratio = boundary_mean / (context_mean + eps)
    inverse_ratio = context_mean / (boundary_mean + eps)
    contrast = boundary_mean - context_mean

    return {
        "boundary_pixels": int(boundary.sum().item()),
        "boundary_band_pixels": int(boundary_band.sum().item()),
        "context_band_pixels": int(context_band.sum().item()),
        "boundary_edge_mean": boundary_mean,
        "context_edge_mean": context_mean,
        "boundary_edge_ratio": ratio,
        "boundary_edge_inverse_ratio": inverse_ratio,
        "boundary_edge_contrast": contrast,
    }


def compute_boundary_alignment_for_dir(
    translated_dir: str | Path,
    mask_root: str | Path,
    image_size: int,
    boundary_radius: int,
    context_radius: int,
    mask_parent_dir: str | None = None,
    ignore_label: int | None = 0,
) -> tuple[dict[str, float], list[tuple[Path, Path]]]:
    pairs = pair_translated_with_masks(
        translated_dir=translated_dir,
        mask_root=mask_root,
        mask_parent_dir=mask_parent_dir,
    )

    per_image_stats: list[dict[str, float]] = []
    skipped_empty_boundary = 0
    skipped_empty_context = 0
    for translated_path, mask_path in pairs:
        translated_img = load_rgb(translated_path, image_size=image_size)
        mask = load_mask(mask_path, image_size=image_size)

        try:
            stats = boundary_edge_alignment_stats(
                translated_img=translated_img,
                mask=mask,
                boundary_radius=boundary_radius,
                context_radius=context_radius,
                ignore_label=ignore_label,
            )
        except ValueError as exc:
            msg = str(exc)
            if "Boundary band is empty" in msg:
                skipped_empty_boundary += 1
                continue
            if "Context band is empty" in msg:
                skipped_empty_context += 1
                continue
            raise

        per_image_stats.append(stats)

    if not per_image_stats:
        raise ValueError(
            "No valid boundary-alignment pairs remained after skipping images with "
            "empty boundary/context bands."
        )

    def summarize_metric(name: str) -> dict[str, float]:
        values = np.asarray([stats[name] for stats in per_image_stats], dtype=np.float64)
        return {
            f"{name}_mean": float(values.mean()),
            f"{name}_std": float(values.std()),
            f"{name}_min": float(values.min()),
            f"{name}_max": float(values.max()),
        }

    summary = {}
    for metric_name in [
        "boundary_pixels",
        "boundary_band_pixels",
        "context_band_pixels",
        "boundary_edge_mean",
        "context_edge_mean",
        "boundary_edge_ratio",
        "boundary_edge_inverse_ratio",
        "boundary_edge_contrast",
    ]:
        summary.update(summarize_metric(metric_name))

    summary["paired_images_total"] = int(len(pairs))
    summary["paired_images_used"] = int(len(per_image_stats))
    summary["skipped_empty_boundary"] = int(skipped_empty_boundary)
    summary["skipped_empty_context"] = int(skipped_empty_context)

    return summary, pairs


def boundary_edge_sweep(
    sim_dir: Path | str, 
    cyclenet_sim_dir: Path | str,
    steps: list[int],
    mask_parent_dir: str = "gt_ss_mask",  
):
    # Target spatial size used when loading translated RGB images and source masks.
    image_size = 256

    # Boundary-band dilation radius in pixels around the source segmentation boundary.
    boundary_radius = 2

    # Outer context-band radius in pixels. The context band is the area between
    # this radius and the narrower boundary band.
    context_radius = 5

    # Raw mask label to ignore when constructing class boundaries. Set to `None`
    # to include every label in the boundary map.
    ignore_label = 0

    for step in steps:
        step_dir = cyclenet_sim_dir / f"step-{step}"
        csv_out_path = step_dir / "boundary_edge_align_stats.csv"
        summary_rows: list[dict[str, object]] = []

        for step, noise_strength, cfg_weight, translated_dir in iter_candidate_dirs(step_dir):
            stats, pairs = compute_boundary_alignment_for_dir(
                translated_dir=translated_dir,
                mask_root=sim_dir,
                image_size=image_size,
                boundary_radius=boundary_radius,
                context_radius=context_radius,
                mask_parent_dir=mask_parent_dir,
                ignore_label=ignore_label,
            )

            print(
                f"step-{step} / strength-{noise_strength:.1f} / cfg-{cfg_weight:.1f}".center(50, "=")
            )
            print(f"paired_images: {len(pairs)}")
            print(f"[ Boundary Edge Ratio         ]: {stats['boundary_edge_ratio_mean']:.6f}")
            print(f"[ Boundary Edge Inverse Ratio ]: {stats['boundary_edge_inverse_ratio_mean']:.6f}")
            print(f"[ Boundary Edge Contrast      ]: {stats['boundary_edge_contrast_mean']:.6f}")

            summary_rows.append(
                {
                    "step": step,
                    "noise_strength": noise_strength,
                    "cfg_weight": cfg_weight,
                    "translated_dir": str(translated_dir),
                    "mask_root": str(sim_dir),
                    "mask_parent_dir": mask_parent_dir,
                    "paired_images": stats["paired_images_used"],
                    "image_size": image_size,
                    "boundary_radius": boundary_radius,
                    "context_radius": context_radius,
                    "ignore_label": ignore_label,
                    **stats,
                }
            )

        if not summary_rows:
            raise ValueError("No boundary-edge alignment stats were computed.")

        csv_out_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "noise_strength",
                    "cfg_weight",
                    "translated_dir",
                    "mask_root",
                    "mask_parent_dir",
                    "paired_images",
                    "paired_images_total",
                    "paired_images_used",
                    "skipped_empty_boundary",
                    "skipped_empty_context",
                    "image_size",
                    "boundary_radius",
                    "context_radius",
                    "ignore_label",
                    "boundary_pixels_mean",
                    "boundary_pixels_std",
                    "boundary_pixels_min",
                    "boundary_pixels_max",
                    "boundary_band_pixels_mean",
                    "boundary_band_pixels_std",
                    "boundary_band_pixels_min",
                    "boundary_band_pixels_max",
                    "context_band_pixels_mean",
                    "context_band_pixels_std",
                    "context_band_pixels_min",
                    "context_band_pixels_max",
                    "boundary_edge_mean_mean",
                    "boundary_edge_mean_std",
                    "boundary_edge_mean_min",
                    "boundary_edge_mean_max",
                    "context_edge_mean_mean",
                    "context_edge_mean_std",
                    "context_edge_mean_min",
                    "context_edge_mean_max",
                    "boundary_edge_ratio_mean",
                    "boundary_edge_ratio_std",
                    "boundary_edge_ratio_min",
                    "boundary_edge_ratio_max",
                    "boundary_edge_inverse_ratio_mean",
                    "boundary_edge_inverse_ratio_std",
                    "boundary_edge_inverse_ratio_min",
                    "boundary_edge_inverse_ratio_max",
                    "boundary_edge_contrast_mean",
                    "boundary_edge_contrast_std",
                    "boundary_edge_contrast_min",
                    "boundary_edge_contrast_max",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\nSaved boundary-edge alignment CSV to {csv_out_path}")
    