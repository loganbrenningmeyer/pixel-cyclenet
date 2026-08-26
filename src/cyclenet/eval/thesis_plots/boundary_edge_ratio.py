from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from cyclenet.data.utils import DEFAULT_SEG_PALETTE
from cyclenet.eval.boundary_edge_align import (
    collect_images_by_name,
    collect_masks_by_name,
    compute_mask_boundary,
    dilate_binary_mask,
    load_mask,
    load_rgb,
    sobel_edge_magnitude,
)
from cyclenet.eval.plotting.set_style import apply_style

apply_style()


def _colorize_label_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D label mask, got shape {mask.shape}")

    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask > 0
    if valid.any():
        class_idx = np.clip(mask[valid] - 1, a_min=0, a_max=DEFAULT_SEG_PALETTE.shape[0] - 1)
        color[valid] = DEFAULT_SEG_PALETTE[class_idx]
    return color


def _overlay_boolean_region(
    base_rgb: np.ndarray,
    region: np.ndarray,
    color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    overlay = base_rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32) * 255.0
    overlay[region] = (1.0 - alpha) * overlay[region] + alpha * color_arr
    return overlay.clip(0.0, 255.0).astype(np.uint8)


def _mute_background_with_region_highlight(
    base_rgb: np.ndarray,
    region: np.ndarray,
    region_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    background_scale: float = 0.30,
    highlight_alpha: float = 0.92,
) -> np.ndarray:
    muted = (base_rgb.astype(np.float32) * background_scale).clip(0.0, 255.0)
    region_color_arr = np.asarray(region_color, dtype=np.float32) * 255.0
    muted[region] = (
        (1.0 - highlight_alpha) * muted[region]
        + highlight_alpha * region_color_arr
    )
    return muted.clip(0.0, 255.0).astype(np.uint8)


def _overlay_two_regions_on_magnitude(
    edge_mag: np.ndarray,
    boundary_band: np.ndarray,
    context_band: np.ndarray,
    boundary_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    context_color: tuple[float, float, float] = (0.05, 0.95, 0.90),
    boundary_alpha: float = 0.38,
    context_alpha: float = 0.16,
) -> np.ndarray:
    vmax = float(np.percentile(edge_mag, 99.0))
    if vmax <= 1e-12:
        vmax = float(edge_mag.max()) if edge_mag.size > 0 else 1.0
    vmax = max(vmax, 1e-6)

    norm = np.clip(edge_mag / vmax, 0.0, 1.0)
    rgb = plt.get_cmap("magma")(norm)[..., :3]
    rgb_uint8 = (rgb * 255.0).round().astype(np.uint8)
    with_context = _overlay_boolean_region(
        base_rgb=rgb_uint8,
        region=context_band,
        color=context_color,
        alpha=context_alpha,
    )
    return _overlay_boolean_region(
        base_rgb=with_context,
        region=boundary_band,
        color=boundary_color,
        alpha=boundary_alpha,
    )


def _collect_triplets(
    sim_image_dir: str | Path,
    sim_label_dir: str | Path,
    translated_image_dir: str | Path,
    label_parent_dir: str | None,
) -> list[tuple[str, Path, Path, Path]]:
    sim_images = collect_images_by_name(sim_image_dir)
    sim_masks = collect_masks_by_name(sim_label_dir, mask_parent_dir=label_parent_dir)
    translated_images = collect_images_by_name(translated_image_dir)

    shared_names = sorted(set(sim_images) & set(sim_masks) & set(translated_images))
    if not shared_names:
        raise ValueError(
            "No shared filenames were found between the simulated image, "
            "label, and translated image directories."
        )

    return [
        (name, sim_images[name], sim_masks[name], translated_images[name])
        for name in shared_names
    ]


def plot_boundary_edge_alignment_samples(
    sim_image_dir: str | Path,
    sim_label_dir: str | Path,
    translated_image_dir: str | Path,
    save_dir: str | Path,
    num_samples: int,
    image_size: int = 256,
    boundary_radius: int = 2,
    context_radius: int = 5,
    label_parent_dir: str | None = None,
    ignore_label: int | None = 0,
    random_seed: int = 42,
) -> list[Path]:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    triplets = _collect_triplets(
        sim_image_dir=sim_image_dir,
        sim_label_dir=sim_label_dir,
        translated_image_dir=translated_image_dir,
        label_parent_dir=label_parent_dir,
    )
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(triplets))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for triplet_idx in order:
        name, sim_image_path, mask_path, translated_path = triplets[int(triplet_idx)]

        sim_img = load_rgb(sim_image_path, image_size=image_size)
        mask = load_mask(mask_path, image_size=image_size)
        translated_img = load_rgb(translated_path, image_size=image_size)

        boundary = compute_mask_boundary(mask, ignore_label=ignore_label)
        boundary_band = dilate_binary_mask(boundary, radius=boundary_radius)
        context_outer = dilate_binary_mask(boundary, radius=context_radius)
        context_band = context_outer & (~boundary_band)

        if boundary_band.sum().item() == 0 or context_band.sum().item() == 0:
            continue

        edge_mag = sobel_edge_magnitude(translated_img)
        boundary_mean = float(edge_mag[boundary_band].mean().item())
        context_mean = float(edge_mag[context_band].mean().item())
        ber = context_mean / max(boundary_mean, 1e-6)

        mask_np = mask.cpu().numpy().astype(np.int64)
        boundary_band_np = boundary_band.cpu().numpy().astype(bool)
        context_band_np = context_band.cpu().numpy().astype(bool)
        sim_rgb = sim_img.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        translated_rgb = translated_img.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
        edge_mag_np = edge_mag.cpu().numpy()

        mask_rgb = _colorize_label_mask(mask_np)
        mask_with_boundary = _mute_background_with_region_highlight(
            base_rgb=mask_rgb,
            region=boundary_band_np,
            region_color=(1.0, 1.0, 1.0),
            background_scale=0.48,
            highlight_alpha=0.94,
        )
        edge_overlay = _overlay_two_regions_on_magnitude(
            edge_mag=edge_mag_np,
            boundary_band=boundary_band_np,
            context_band=context_band_np,
        )

        fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0))
        axes = axes.ravel()

        axes[0].imshow(sim_rgb, interpolation="nearest")

        axes[1].imshow(mask_with_boundary, interpolation="nearest")

        axes[2].imshow(translated_rgb, interpolation="nearest")

        axes[3].imshow(edge_overlay, interpolation="nearest")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

        legend_handles = [
            mpatches.Patch(color=(1.0, 1.0, 1.0), label="Boundary band"),
            mpatches.Patch(color=(0.05, 0.95, 0.90), label="Surrounding band"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=2,
            frameon=True,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

        save_path = save_dir / f"{Path(name).stem}_boundary_edge_align.pdf"
        fig.savefig(save_path)
        plt.close(fig)
        saved_paths.append(save_path)

        if len(saved_paths) >= num_samples:
            break

    if not saved_paths:
        raise ValueError(
            "No valid samples were found after filtering for non-empty boundary "
            "and surrounding regions."
        )

    return saved_paths


def main() -> None:
    # Directory containing simulated source RGB images used for filename pairing.
    sim_image_dir = "/develop/data/remote_sensing/tiled/projection/sim_proj/opt"
    # Directory containing the corresponding simulated segmentation labels.
    sim_label_dir = "/develop/data/remote_sensing/tiled/projection/sim_proj/gt_ss_mask"
    # Directory containing translated RGB images to visualize.
    translated_image_dir = "/develop/data/remote_sensing/tiled/projection/cyclenet_sim_proj/seg/oem_only_seg_only/ema/step-40000/strength-0.5/cfg-1.0/opt"
    # Output directory where one 2x2 boundary-alignment figure per sample will be saved.
    save_dir = "/develop/code/eval/thesis/boundary_edge_align/oem_only_seg_only"
    # Number of random valid samples to visualize.
    num_samples = 6
    # Spatial size used when loading masks and translated images for plotting.
    image_size = 256
    # Boundary-band dilation radius in pixels around source segmentation boundaries.
    boundary_radius = 2
    # Surrounding-band dilation radius in pixels; must be larger than boundary_radius.
    context_radius = 5
    # Optional parent directory filter for segmentation masks. Use `None` to disable.
    label_parent_dir = None
    # Raw label value to ignore when constructing source-mask boundaries. Use `None` to include all labels.
    ignore_label = 0
    # Random seed for sample selection.
    random_seed = 42

    saved_paths = plot_boundary_edge_alignment_samples(
        sim_image_dir=sim_image_dir,
        sim_label_dir=sim_label_dir,
        translated_image_dir=translated_image_dir,
        save_dir=save_dir,
        num_samples=num_samples,
        image_size=image_size,
        boundary_radius=boundary_radius,
        context_radius=context_radius,
        label_parent_dir=label_parent_dir,
        ignore_label=ignore_label,
        random_seed=random_seed,
    )
    print(f"Saved {len(saved_paths)} boundary-edge alignment plots to {save_dir}")


if __name__ == "__main__":
    main()
