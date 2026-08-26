from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from cyclenet.data import TranslateSegDataset
from cyclenet.eval.plotting.set_style import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_NAMES,
    apply_style,
)

apply_style()


def tensor_to_numpy_image(img: torch.Tensor) -> np.ndarray:
    img = ((img.clamp(-1.0, 1.0) + 1.0) / 2.0).float().cpu()
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0)


def segmentation_to_color_image(seg: torch.Tensor) -> np.ndarray:
    if seg.ndim != 3:
        raise ValueError(f"Expected segmentation tensor with shape [C, H, W], got {tuple(seg.shape)}.")

    seg = seg.detach().cpu()
    valid = seg.sum(dim=0) > 0
    class_idx = seg.argmax(dim=0).numpy().astype(np.int64)

    palette = np.array(
        [to_rgb(CLASS_COLORS[CLASS_LABELS[label_id]]) for label_id in sorted(CLASS_LABELS)],
        dtype=np.float32,
    )
    color_image = np.ones((seg.shape[1], seg.shape[2], 3), dtype=np.float32)
    color_image[valid.numpy()] = palette[class_idx[valid.numpy()]]
    return color_image


def select_sample_indices(n_total: int, n_select: int, seed: int) -> list[int]:
    if n_total <= 0:
        raise ValueError("Cannot sample from an empty dataset.")
    if n_select <= 0:
        raise ValueError(f"n_select must be positive, got {n_select}.")
    if n_select > n_total:
        raise ValueError(f"Requested {n_select} samples from a dataset of size {n_total}.")

    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), k=n_select))


def load_random_domain_samples(
    data_dir: str | Path,
    image_size: int,
    n_samples: int,
    seed: int,
    rgb_parent_dirs: set[str],
    label_parent_dir: str,
    num_classes: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    dataset = TranslateSegDataset(
        src_dir=str(data_dir),
        image_size=image_size,
        num_classes=num_classes,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
    )
    sample_indices = select_sample_indices(len(dataset), n_samples, seed)

    images: list[np.ndarray] = []
    segmentations: list[np.ndarray] = []
    filepaths: list[str] = []
    for sample_idx in sample_indices:
        img, seg, filepath = dataset[sample_idx]
        images.append(tensor_to_numpy_image(img))
        segmentations.append(segmentation_to_color_image(seg))
        filepaths.append(filepath)

    return images, segmentations, filepaths


def build_segmentation_legend_handles() -> list[Patch]:
    handles: list[Patch] = []
    for label_id in sorted(CLASS_LABELS):
        class_slug = CLASS_LABELS[label_id]
        class_name = CLASS_NAMES[class_slug]
        handles.append(
            Patch(
                facecolor=CLASS_COLORS[class_slug],
                edgecolor="none",
                label=class_name,
            )
        )
    return handles


def plot_real_sim_seg_grid(
    real_dir: str | Path,
    sim_dir: str | Path,
    save_path: str | Path,
    n_samples: int,
    image_size: int,
    rgb_parent_dirs: set[str] | None = None,
    label_parent_dir: str = "gt_ss_mask",
    num_classes: int = 8,
    seed: int = 42,
    legend_ncol: int = 4,
    row_label_fontsize: float = 12.0,
    legend_fontsize: float = 10.0,
    legend_y_pad: float = 0.018,
    row_label_x: float = 0.035,
    row_label_rotation: float = 90.0,
    middle_gap: float = 0.012,
    divider_linewidth: float = 1.2,
    divider_color: str = "black",
) -> Path:
    rgb_parent_dirs = {"opt"} if rgb_parent_dirs is None else {str(parent) for parent in rgb_parent_dirs}

    real_images, real_segmentations, real_paths = load_random_domain_samples(
        data_dir=real_dir,
        image_size=image_size,
        n_samples=n_samples,
        seed=seed,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
        num_classes=num_classes,
    )
    sim_images, sim_segmentations, sim_paths = load_random_domain_samples(
        data_dir=sim_dir,
        image_size=image_size,
        n_samples=n_samples,
        seed=seed + 1,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
        num_classes=num_classes,
    )

    fig_w = max(1.8 * n_samples, 7.0)
    fig_h = 8.0
    fig, axes = plt.subplots(4, n_samples, figsize=(fig_w, fig_h), squeeze=False)

    row_images = [
        real_images,
        real_segmentations,
        sim_images,
        sim_segmentations,
    ]

    for row_idx, images in enumerate(row_images):
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, interpolation="nearest")
            ax.axis("off")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.08, 0.10, 1.0, 0.90))

    if middle_gap < 0.0:
        raise ValueError(f"middle_gap must be non-negative, got {middle_gap}.")

    if middle_gap > 0.0:
        row_shift = 0.5 * middle_gap
        for row_idx in (0, 1):
            for ax in axes[row_idx]:
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 + row_shift, pos.width, pos.height])
        for row_idx in (2, 3):
            for ax in axes[row_idx]:
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 - row_shift, pos.width, pos.height])

    real_group_y = 0.5 * (
        0.5 * (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1)
        + 0.5 * (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1)
    )
    sim_group_y = 0.5 * (
        0.5 * (axes[2, 0].get_position().y0 + axes[2, 0].get_position().y1)
        + 0.5 * (axes[3, 0].get_position().y0 + axes[3, 0].get_position().y1)
    )
    divider_y = 0.5 * (axes[1, 0].get_position().y0 + axes[2, 0].get_position().y1)
    grid_x0 = axes[0, 0].get_position().x0
    grid_x1 = axes[0, -1].get_position().x1
    divider_x0 = grid_x0
    divider_x1 = grid_x1
    legend_y = min(0.985, axes[0, 0].get_position().y1 + legend_y_pad)

    legend_handles = build_segmentation_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(grid_x0, legend_y, grid_x1 - grid_x0, 0.01),
        bbox_transform=fig.transFigure,
        ncol=legend_ncol,
        mode="expand",
        frameon=True,
        fontsize=legend_fontsize,
        borderaxespad=0.0,
    )

    fig.text(
        row_label_x,
        real_group_y,
        "Real\n(OEM)",
        rotation=row_label_rotation,
        ha="center",
        va="center",
        fontsize=row_label_fontsize,
    )
    fig.text(
        row_label_x,
        sim_group_y,
        "Simulated\n(SynRS3D)",
        rotation=row_label_rotation,
        ha="center",
        va="center",
        fontsize=row_label_fontsize,
    )
    fig.add_artist(
        Line2D(
            [divider_x0, divider_x1],
            [divider_y, divider_y],
            transform=fig.transFigure,
            color=divider_color,
            linewidth=divider_linewidth,
        )
    )

    fig.savefig(save_path)
    plt.close(fig)

    print("Saved real sample paths:")
    for filepath in real_paths:
        print(f"  {filepath}")

    print("Saved sim sample paths:")
    for filepath in sim_paths:
        print(f"  {filepath}")

    return save_path


def main() -> None:
    # Root directory containing the real RGB tiles and sibling segmentation masks.
    real_dir = "/develop/data/remote_sensing/tiled/projection/oem_proj"
    # Root directory containing the simulated RGB tiles and sibling segmentation masks.
    sim_dir = "/develop/data/remote_sensing/tiled/projection/sim_proj"
    # Output path for the saved thesis figure.
    save_path = "/develop/code/eval/thesis/real_sim_seg_grid/real_sim_seg_grid.pdf"
    # Number of random columns to sample from each domain.
    n_samples = 6
    # Spatial resolution applied to the loaded RGB images and segmentation masks.
    image_size = 224
    # Allowed RGB parent directories used when discovering source images.
    rgb_parent_dirs = {"opt"}
    # Name of the sibling directory containing segmentation masks.
    label_parent_dir = "gt_ss_mask"
    # Number of valid segmentation classes expected in each label mask.
    num_classes = 8
    # Random seed for reproducible real/sim sample selection.
    seed = 42
    # Number of legend columns for the segmentation-class legend.
    legend_ncol = 4
    # Font size for the repeated Real/Sim row labels.
    row_label_fontsize = 20.0
    # Font size for the segmentation-class legend entries.
    legend_fontsize = 16.0
    # Vertical gap between the top image row and the legend, in figure coordinates.
    legend_y_pad = 0.018
    # Horizontal figure coordinate for the large domain-group labels.
    row_label_x = 0.06
    # Rotation angle for the large domain-group labels.
    row_label_rotation = 90.0
    # Extra whitespace inserted around the divider between the real and simulated sections.
    middle_gap = 0.012
    # Line width of the divider separating real rows from simulated rows.
    divider_linewidth = 1.2
    # Color of the divider separating real rows from simulated rows.
    divider_color = "black"

    saved_path = plot_real_sim_seg_grid(
        real_dir=real_dir,
        sim_dir=sim_dir,
        save_path=save_path,
        n_samples=n_samples,
        image_size=image_size,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
        num_classes=num_classes,
        seed=seed,
        legend_ncol=legend_ncol,
        row_label_fontsize=row_label_fontsize,
        legend_fontsize=legend_fontsize,
        legend_y_pad=legend_y_pad,
        row_label_x=row_label_x,
        row_label_rotation=row_label_rotation,
        middle_gap=middle_gap,
        divider_linewidth=divider_linewidth,
        divider_color=divider_color,
    )
    print(f"Saved thesis real/sim segmentation grid to {saved_path}")


if __name__ == "__main__":
    main()
