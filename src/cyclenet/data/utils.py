from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Patch


DEFAULT_SEG_PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 50],
    ],
    dtype=np.uint8,
)


def _to_uint8_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().float()
        if image_np.ndim != 3:
            raise ValueError(f"Expected CHW image tensor, got shape {tuple(image_np.shape)}")

        if image_np.shape[0] != 3:
            raise ValueError(f"Expected 3-channel image tensor, got shape {tuple(image_np.shape)}")

        if float(image_np.min()) < 0.0:
            image_np = ((image_np.clamp(-1.0, 1.0) + 1.0) / 2.0).permute(1, 2, 0).numpy()
        else:
            image_np = image_np.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    else:
        image_np = np.asarray(image)
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected HWC image array, got shape {tuple(image_np.shape)}")

        if np.issubdtype(image_np.dtype, np.floating):
            if float(image_np.min()) < 0.0:
                image_np = np.clip((image_np + 1.0) / 2.0, 0.0, 1.0)
            else:
                image_np = np.clip(image_np, 0.0, 1.0)
        else:
            image_np = image_np.astype(np.float32) / 255.0

    return (image_np * 255.0).round().astype(np.uint8)


def _seg_to_class_index(seg_mask: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(seg_mask, torch.Tensor):
        seg_np = seg_mask.detach().cpu()

        if seg_np.ndim == 3:
            valid = (seg_np.sum(dim=0) > 0).numpy()
            class_idx = seg_np.argmax(dim=0).numpy().astype(np.int64)
            return class_idx, valid

        if seg_np.ndim == 2:
            seg_np = seg_np.long().numpy().astype(np.int64)
            valid = seg_np > 0
            class_idx = np.clip(seg_np - 1, a_min=0, a_max=None)
            return class_idx, valid

        raise ValueError(f"Expected CHW one-hot or HW label tensor, got shape {tuple(seg_np.shape)}")

    seg_np = np.asarray(seg_mask)
    if seg_np.ndim == 3:
        valid = seg_np.sum(axis=0) > 0
        class_idx = seg_np.argmax(axis=0).astype(np.int64)
        return class_idx, valid

    if seg_np.ndim == 2:
        seg_np = seg_np.astype(np.int64)
        valid = seg_np > 0
        class_idx = np.clip(seg_np - 1, a_min=0, a_max=None)
        return class_idx, valid

    raise ValueError(f"Expected CHW one-hot or HW label array, got shape {tuple(seg_np.shape)}")


def overlay_segmentation_mask(
    image: torch.Tensor | np.ndarray,
    seg_mask: torch.Tensor | np.ndarray,
    alpha: float = 0.45,
    palette: np.ndarray = DEFAULT_SEG_PALETTE,
) -> np.ndarray:
    image_np = _to_uint8_image(image)
    class_idx, valid = _seg_to_class_index(seg_mask)

    if class_idx.shape != image_np.shape[:2]:
        raise ValueError(
            "Image and segmentation mask must have the same spatial size, "
            f"got image={image_np.shape[:2]} mask={class_idx.shape}"
        )

    if valid.any() and palette.shape[0] <= int(class_idx[valid].max()):
        raise ValueError(
            f"Palette has {palette.shape[0]} colors, but found class index {int(class_idx[valid].max())}."
        )

    color_mask = np.zeros_like(image_np)
    color_mask[valid] = palette[class_idx[valid]]

    overlay = image_np.astype(np.float32).copy()
    overlay[valid] = (
        (1.0 - alpha) * image_np[valid].astype(np.float32)
        + alpha * color_mask[valid].astype(np.float32)
    )
    return overlay.clip(0, 255).astype(np.uint8)


def render_segmentation_overlay_comparison(
    source_images: torch.Tensor,
    seg_masks: torch.Tensor,
    translated_images: torch.Tensor,
    class_names: list[str],
    out_path: str | Path | None = None,
    alpha: float = 0.45,
    palette: np.ndarray = DEFAULT_SEG_PALETTE,
    source_title: str = "Source",
    source_overlay_title: str = "Source + Mask",
    translated_title: str = "Translated",
    translated_overlay_title: str = "Translated + Mask",
) -> torch.Tensor:
    if source_images.ndim != 4 or translated_images.ndim != 4:
        raise ValueError("Expected batched source and translated image tensors with shape [B, C, H, W].")

    if seg_masks.ndim != 4:
        raise ValueError("Expected batched segmentation tensor with shape [B, C, H, W].")

    if len(source_images) != len(seg_masks) or len(source_images) != len(translated_images):
        raise ValueError("Source images, segmentation masks, and translated images must have the same batch size.")

    if len(class_names) != int(seg_masks.shape[1]):
        raise ValueError(
            f"class_names length ({len(class_names)}) must match segmentation channels ({int(seg_masks.shape[1])})."
        )

    if palette.shape[0] < len(class_names):
        raise ValueError(f"Palette has {palette.shape[0]} colors, but {len(class_names)} class names were provided.")

    batch_size = int(source_images.shape[0])
    fig_height = max(2.8 * batch_size + 1.0, 4.0)
    fig = Figure(figsize=(13.0, fig_height), dpi=150)
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(batch_size, 4, squeeze=False)

    for row in range(batch_size):
        source = _to_uint8_image(source_images[row])
        source_overlay = overlay_segmentation_mask(
            image=source_images[row],
            seg_mask=seg_masks[row],
            alpha=alpha,
            palette=palette,
        )
        translated = _to_uint8_image(translated_images[row])
        translated_overlay = overlay_segmentation_mask(
            image=translated_images[row],
            seg_mask=seg_masks[row],
            alpha=alpha,
            palette=palette,
        )

        axes[row, 0].imshow(source, interpolation="nearest")
        axes[row, 1].imshow(source_overlay, interpolation="nearest")
        axes[row, 2].imshow(translated, interpolation="nearest")
        axes[row, 3].imshow(translated_overlay, interpolation="nearest")

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].set_frame_on(False)

    axes[0, 0].set_title(source_title)
    axes[0, 1].set_title(source_overlay_title)
    axes[0, 2].set_title(translated_title)
    axes[0, 3].set_title(translated_overlay_title)

    legend_handles = [
        Patch(facecolor=palette[idx] / 255.0, edgecolor="none", label=class_name)
        for idx, class_name in enumerate(class_names)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(class_names)),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    canvas.draw()
    width, height = fig.canvas.get_width_height()
    rendered = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)

    return torch.from_numpy(rendered.copy()).permute(2, 0, 1).float() / 255.0
