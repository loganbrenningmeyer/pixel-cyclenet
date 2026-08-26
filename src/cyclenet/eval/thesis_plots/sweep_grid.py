from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast

from cyclenet.data import TranslateDataset, TranslateSegDataset
from cyclenet.diffusion import DiffusionSchedule, cyclenet_ddim_loop, cyclenet_ddpm_loop
from cyclenet.eval.plotting.image_grid import plot_image_grid
from cyclenet.eval.plotting.set_style import apply_style
from cyclenet.models import CycleNet, UNet
from cyclenet.models.conditioning import (
    DomainEmbedding,
    build_condition_input,
    build_seg_modulation_input,
)
from cyclenet.models.controlnet import build_controlnet

apply_style()


def load_config(config_path: str | Path) -> DictConfig:
    return OmegaConf.load(str(config_path))


def cfg_select(config: DictConfig, key: str, default=None):
    value = OmegaConf.select(config, key)
    return default if value is None else value


def as_parent_dir_set(value) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(v) for v in value}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_numpy_image(img: torch.Tensor) -> np.ndarray:
    img = ((img.clamp(-1.0, 1.0) + 1.0) / 2.0).float().cpu()
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0)


def _checkpoint_training_dir(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix != ".ckpt":
        raise ValueError(f"Expected a `.ckpt` checkpoint path, got: {checkpoint_path}")
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError(
            f"Expected checkpoint to live under a `checkpoints/` directory, got: {checkpoint_path}"
        )
    return checkpoint_path.parent.parent


def _requires_segmentation(cyclenet_config: DictConfig) -> bool:
    cond_mode = str(cyclenet_config.model.cond_mode)
    use_spade = bool(cyclenet_config.model.use_spade)
    return cond_mode in {"seg", "rgb_seg"} or use_spade


def build_model(
    cyclenet_config: DictConfig,
    unet_config: DictConfig,
    device: torch.device,
) -> tuple[CycleNet, str, bool]:
    backbone = UNet(
        in_ch=3,
        base_ch=unet_config.model.base_ch,
        t_dim=unet_config.model.t_dim,
        d_dim=unet_config.model.d_dim,
        ch_mults=unet_config.model.ch_mults,
        num_res_blocks=unet_config.model.num_res_blocks,
        enc_heads=unet_config.model.enc_heads,
        mid_heads=unet_config.model.mid_heads,
        res_dropout=unet_config.model.res_dropout,
        attn_dropout=unet_config.model.attn_dropout,
        ffn_dropout=unet_config.model.ffn_dropout,
    ).to(device)

    cond_mode = str(cyclenet_config.model.cond_mode)
    use_spade = bool(cyclenet_config.model.use_spade)
    num_seg_classes = int(cfg_select(cyclenet_config, "model.num_seg_classes", 8))
    s_dim = OmegaConf.select(cyclenet_config, "model.s_dim", default=None)
    skip_block_mask = OmegaConf.select(cyclenet_config, "model.skip_block_mask", default=None)
    use_mid_skip = bool(OmegaConf.select(cyclenet_config, "model.use_mid_skip", default=True))
    if skip_block_mask is not None:
        skip_block_mask = [bool(v) for v in skip_block_mask]

    domain_emb = DomainEmbedding(d_dim=unet_config.model.d_dim).to(device)
    control = build_controlnet(
        backbone=backbone,
        cond_mode=cond_mode,
        num_seg_classes=num_seg_classes,
        use_spade=use_spade,
        s_dim=s_dim,
        skip_block_mask=skip_block_mask,
        use_mid_skip=use_mid_skip,
    ).to(device)

    model = CycleNet(
        backbone=backbone,
        control=control,
        domain_emb=domain_emb,
        t_dim=unet_config.model.t_dim,
        d_dim=unet_config.model.d_dim,
    ).to(device)
    return model, cond_mode, use_spade


def load_checkpoint(model: CycleNet, checkpoint_path: str | Path, model_key: str) -> None:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if model_key not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain key '{model_key}'.")
    model.load_state_dict(ckpt[model_key], strict=True)


def build_schedule(unet_config: DictConfig, device: torch.device) -> DiffusionSchedule:
    return DiffusionSchedule(
        schedule=unet_config.diffusion.schedule,
        T=unet_config.diffusion.T,
        beta_start=unet_config.diffusion.beta_start,
        beta_end=unet_config.diffusion.beta_end,
        device=device,
        s=unet_config.diffusion.s,
    )


def load_selected_sample(
    src_dir: str | Path,
    image_size: int,
    rgb_parent_dirs: set[str] | None,
    label_parent_dir: str,
    num_seg_classes: int,
    requires_segmentation: bool,
    sample_index: int,
    sample_rel_path: str | Path | None,
) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    src_dir = Path(src_dir)

    if requires_segmentation:
        dataset = TranslateSegDataset(
            src_dir=str(src_dir),
            image_size=image_size,
            num_classes=num_seg_classes,
            rgb_parent_dirs=rgb_parent_dirs if rgb_parent_dirs is not None else {"opt"},
            label_parent_dir=label_parent_dir,
        )
    else:
        dataset = TranslateDataset(
            src_dir=str(src_dir),
            image_size=image_size,
            rgb_parent_dirs=rgb_parent_dirs,
        )

    if len(dataset) == 0:
        raise ValueError(f"No source images found under {src_dir}.")

    if sample_rel_path is not None:
        target_rel_path = Path(sample_rel_path)
        for idx in range(len(dataset)):
            item = dataset[idx]
            filepath = item[-1]
            rel_path = Path(filepath).resolve().relative_to(src_dir.resolve())
            if rel_path == target_rel_path:
                if requires_segmentation:
                    x_src, seg_src, filepath = item
                    return x_src, seg_src, filepath
                x_src, filepath = item
                return x_src, None, filepath

        raise FileNotFoundError(
            f"Could not find sample_rel_path '{target_rel_path}' under filtered dataset rooted at {src_dir}."
        )

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} is out of range for dataset of size {len(dataset)}.")

    item = dataset[sample_index]
    if requires_segmentation:
        x_src, seg_src, filepath = item
        return x_src, seg_src, filepath
    x_src, filepath = item
    return x_src, None, filepath


def load_translate_dataset(
    src_dir: str | Path,
    image_size: int,
    rgb_parent_dirs: set[str] | None,
    label_parent_dir: str,
    num_seg_classes: int,
    requires_segmentation: bool,
):
    src_dir = Path(src_dir)
    if requires_segmentation:
        return TranslateSegDataset(
            src_dir=str(src_dir),
            image_size=image_size,
            num_classes=num_seg_classes,
            rgb_parent_dirs=rgb_parent_dirs if rgb_parent_dirs is not None else {"opt"},
            label_parent_dir=label_parent_dir,
        )
    return TranslateDataset(
        src_dir=str(src_dir),
        image_size=image_size,
        rgb_parent_dirs=rgb_parent_dirs,
    )


def _dataset_item_to_sample(
    item,
    requires_segmentation: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, str]:
    if requires_segmentation:
        x_src, seg_src, filepath = item
        return x_src, seg_src, filepath
    x_src, filepath = item
    return x_src, None, filepath


def _select_sample_requests(
    dataset,
    src_dir: str | Path,
    requires_segmentation: bool,
    sample_index: int,
    num_samples: int,
    sample_rel_path: str | Path | None,
) -> list[tuple[torch.Tensor, torch.Tensor | None, str]]:
    src_dir = Path(src_dir)
    if len(dataset) == 0:
        raise ValueError(f"No source images found under {src_dir}.")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")

    if sample_rel_path is not None:
        if num_samples != 1:
            raise ValueError("sample_rel_path can only be used when num_samples == 1.")
        target_rel_path = Path(sample_rel_path)
        for idx in range(len(dataset)):
            item = dataset[idx]
            filepath = item[-1]
            rel_path = Path(filepath).resolve().relative_to(src_dir.resolve())
            if rel_path == target_rel_path:
                return [_dataset_item_to_sample(item, requires_segmentation)]
        raise FileNotFoundError(
            f"Could not find sample_rel_path '{target_rel_path}' under filtered dataset rooted at {src_dir}."
        )

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} is out of range for dataset of size {len(dataset)}.")
    if sample_index + num_samples > len(dataset):
        raise IndexError(
            f"Requested samples [{sample_index}, {sample_index + num_samples}) from dataset of size {len(dataset)}."
        )

    return [
        _dataset_item_to_sample(dataset[idx], requires_segmentation)
        for idx in range(sample_index, sample_index + num_samples)
    ]


def translate_one(
    model: CycleNet,
    x_src: torch.Tensor,
    seg_src: torch.Tensor | None,
    device: torch.device,
    sched: DiffusionSchedule,
    src_domain_idx: int,
    sampler: str,
    cfg_weight: float,
    noise_strength: float,
    ddim_steps: int,
    eta: float,
    cond_mode: str,
    use_spade: bool,
) -> np.ndarray:
    x_src = x_src.unsqueeze(0).to(device, non_blocking=True)
    seg_src = None if seg_src is None else seg_src.unsqueeze(0).to(device, non_blocking=True)

    src_idx = torch.full((1,), src_domain_idx, device=device, dtype=torch.long)
    tgt_idx = torch.full((1,), 1 - src_domain_idx, device=device, dtype=torch.long)
    c_img = build_condition_input(x_src, seg_src, cond_mode)
    seg_mod = build_seg_modulation_input(seg_src, use_spade)

    with autocast(device_type=device.type, enabled=device.type == "cuda"):
        if sampler.lower() == "ddpm":
            samples, _ = cyclenet_ddpm_loop(
                model=model,
                x_src=x_src,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                c_img=c_img,
                seg=seg_mod,
                sched=sched,
                w=cfg_weight,
                strength=noise_strength,
            )
        elif sampler.lower() == "ddim":
            samples, _ = cyclenet_ddim_loop(
                model=model,
                x_src=x_src,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                c_img=c_img,
                seg=seg_mod,
                sched=sched,
                w=cfg_weight,
                strength=noise_strength,
                num_steps=ddim_steps,
                eta=eta,
            )
        else:
            raise ValueError("sampler must be 'ddpm' or 'ddim'.")

    return tensor_to_numpy_image(samples[0])


def _selected_grid_index(
    noise_strengths: list[float],
    cfg_weights: list[float],
    selected_noise_strength: float | None,
    selected_cfg_weight: float | None,
) -> int | None:
    if selected_noise_strength is None and selected_cfg_weight is None:
        return None
    if selected_noise_strength is None or selected_cfg_weight is None:
        raise ValueError(
            "selected_noise_strength and selected_cfg_weight must both be set to highlight an operating point."
        )

    row_idx = None
    for idx, value in enumerate(noise_strengths):
        if np.isclose(value, selected_noise_strength):
            row_idx = idx
            break
    if row_idx is None:
        raise ValueError(
            f"selected_noise_strength={selected_noise_strength} is not present in noise_strengths={noise_strengths}."
        )

    col_idx = None
    for idx, value in enumerate(cfg_weights):
        if np.isclose(value, selected_cfg_weight):
            col_idx = idx
            break
    if col_idx is None:
        raise ValueError(
            f"selected_cfg_weight={selected_cfg_weight} is not present in cfg_weights={cfg_weights}."
        )

    return row_idx * len(cfg_weights) + col_idx


def _add_image_border(
    image: np.ndarray,
    color: str,
    linewidth: float,
    inset_fraction: float,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, 3), got {image.shape}.")

    height, width = image.shape[:2]
    min_dim = max(1, min(height, width))
    pad_px = int(round(max(0.0, inset_fraction) * min_dim))
    border_px = max(1, int(round(max(0.5, linewidth))))
    border_color = np.asarray(to_rgb(color), dtype=np.float32)

    canvas_height = height + 2 * (border_px + pad_px)
    canvas_width = width + 2 * (border_px + pad_px)
    bordered = np.ones((canvas_height, canvas_width, 3), dtype=np.float32)

    y0 = border_px + pad_px
    x0 = border_px + pad_px
    bordered[y0 : y0 + height, x0 : x0 + width] = image

    if pad_px == 0:
        bordered[0:border_px, :] = border_color
        bordered[-border_px:, :] = border_color
        bordered[:, 0:border_px] = border_color
        bordered[:, -border_px:] = border_color
    else:
        top = pad_px
        bottom = canvas_height - pad_px
        left = pad_px
        right = canvas_width - pad_px
        bordered[top : top + border_px, left:right] = border_color
        bordered[bottom - border_px : bottom, left:right] = border_color
        bordered[top:bottom, left : left + border_px] = border_color
        bordered[top:bottom, right - border_px : right] = border_color

    return np.clip(bordered, 0.0, 1.0)


def _build_vertical_preview_panel(
    top_image: np.ndarray,
    bottom_image: np.ndarray,
    gap_fraction: float = 0.06,
    gap_color: str = "white",
) -> np.ndarray:
    if top_image.shape != bottom_image.shape:
        raise ValueError(
            f"Top and bottom preview images must have the same shape, got {top_image.shape} and {bottom_image.shape}."
        )

    height, width = top_image.shape[:2]
    gap_px = max(1, int(round(max(0.0, gap_fraction) * height)))
    gap_rgb = np.asarray(to_rgb(gap_color), dtype=np.float32)
    gap_strip = np.ones((gap_px, width, 3), dtype=np.float32) * gap_rgb
    return np.concatenate([top_image, gap_strip, bottom_image], axis=0)


def _plot_grid_with_stacked_preview(
    images: list[np.ndarray],
    n_cols: int,
    row_labels: list[str],
    col_labels: list[str],
    source_image: np.ndarray,
    translated_image: np.ndarray,
    source_label: str,
    translated_label: str | None,
    title: str | None,
    xlabel: str,
    ylabel: str,
    save_path: str | Path,
    dpi: int,
    scale: float,
    source_width_ratio: float,
    source_gap_ratio: float,
    grid_wspace: float,
    grid_hspace: float,
    show_source_divider: bool,
    source_divider_color: str,
    source_divider_linewidth: float,
    source_divider_gap_fraction: float,
    source_label_fontsize: float,
    source_label_pad: float,
    row_label_side: str,
    row_label_right_pad: float,
    row_label_fontsize: float,
    col_label_fontsize: float,
    title_fontsize: float,
    title_y: float,
    xlabel_fontsize: float,
    xlabel_y: float,
    ylabel_fontsize: float,
    ylabel_pad: float,
    ylabel_right_pad: float,
    ylabel_side: str,
    ylabel_rotation: float,
    subplot_top_with_title: float,
    subplot_top_without_title: float,
    subplot_bottom_with_xlabels: float,
    subplot_bottom_without_xlabels: float,
    subplot_left_with_ylabels: float,
    subplot_left_without_ylabels: float,
    subplot_right: float,
    save_pad_inches: float,
    image_interpolation: str,
    preview_gap_fraction: float,
    preview_height_fill_fraction: float,
    selected_border_color: str,
    selected_border_width: float,
    selected_index: int,
) -> None:
    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))
    H, W = images[0].shape[:2]

    top = subplot_top_with_title if title else subplot_top_without_title
    bottom = subplot_bottom_with_xlabels if col_labels or xlabel else subplot_bottom_without_xlabels
    left = subplot_left_with_ylabels if row_labels or ylabel else subplot_left_without_ylabels
    right = subplot_right

    grid_height_ratio = n_rows * (1.0 + grid_hspace * (n_rows - 1) / max(n_rows, 1))
    stacked_height_ratio = 2.0 + max(0.0, preview_gap_fraction)
    min_source_width_ratio = (
        min(max(preview_height_fill_fraction, 0.05), 1.0) * grid_height_ratio / stacked_height_ratio
    )
    source_width_ratio = max(source_width_ratio, min_source_width_ratio)

    grid_width_ratio = n_cols + grid_wspace * max(n_cols - 1, 0)
    outer_width_ratios = [source_width_ratio, source_gap_ratio, grid_width_ratio]
    axes_width_px = W * sum(outer_width_ratios)
    axes_height_px = H * grid_height_ratio
    fig_width = (axes_width_px / max(right - left, 1e-6)) / dpi * scale
    fig_height = (axes_height_px / max(top - bottom, 1e-6)) / dpi * scale

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=outer_width_ratios,
        wspace=0.0,
        hspace=grid_hspace,
    )
    source_gs = outer_gs[0, 0].subgridspec(
        3,
        1,
        height_ratios=[1.0, max(preview_gap_fraction, 0.02), 1.0],
        hspace=0.0,
    )
    grid_gs = outer_gs[0, 2].subgridspec(n_rows, n_cols, wspace=grid_wspace, hspace=grid_hspace)

    source_ax = fig.add_subplot(source_gs[0, 0])
    source_ax.imshow(source_image, interpolation=image_interpolation)
    source_ax.axis("off")
    if source_label:
        source_ax.set_title(source_label, fontsize=source_label_fontsize, pad=source_label_pad)

    gap_ax = fig.add_subplot(source_gs[1, 0])
    gap_ax.axis("off")

    translated_ax = fig.add_subplot(source_gs[2, 0])
    translated_ax.imshow(translated_image, interpolation=image_interpolation)
    translated_ax.axis("off")
    if translated_label:
        translated_ax.set_title(translated_label, fontsize=source_label_fontsize, pad=source_label_pad)
    translated_ax.add_patch(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=translated_ax.transAxes,
            fill=False,
            edgecolor=selected_border_color,
            linewidth=selected_border_width,
            clip_on=False,
            zorder=10,
        )
    )

    axs = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col] = fig.add_subplot(grid_gs[row, col])

    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axs[row, col]
        ax.imshow(img, interpolation=image_interpolation)
        ax.axis("off")

        if idx == selected_index:
            ax.add_patch(
                Rectangle(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor=selected_border_color,
                    linewidth=selected_border_width,
                    clip_on=False,
                    zorder=10,
                )
            )

    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axs[row, col].axis("off")

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    if row_labels is not None:
        row_label_side = row_label_side.lower()
        if row_label_side not in {"left", "right"}:
            raise ValueError(f"Unsupported row_label_side '{row_label_side}'.")
        for row, label in enumerate(row_labels[:n_rows]):
            target_ax = axs[row, -1]
            label_x = 1.0 + row_label_right_pad if row_label_side == "right" else -0.08
            ha = "left" if row_label_side == "right" else "right"
            target_ax.text(
                label_x,
                0.5,
                label,
                transform=target_ax.transAxes,
                ha=ha,
                va="center",
                rotation=0,
                clip_on=False,
                fontsize=row_label_fontsize,
            )

    if col_labels is not None:
        for col, label in enumerate(col_labels[:n_cols]):
            axs[-1, col].text(
                0.5,
                -0.08,
                label,
                transform=axs[-1, col].transAxes,
                ha="center",
                va="top",
                clip_on=False,
                fontsize=col_label_fontsize,
            )

    if show_source_divider:
        gap_left = translated_ax.get_position().x1
        gap_right = axs[0, 0].get_position().x0
        divider_x = gap_left + (gap_right - gap_left) * source_divider_gap_fraction
        divider_y0 = min(translated_ax.get_position().y0, axs[-1, 0].get_position().y0)
        divider_y1 = max(source_ax.get_position().y1, axs[0, 0].get_position().y1)
        fig.add_artist(
            Line2D(
                [divider_x, divider_x],
                [divider_y0, divider_y1],
                transform=fig.transFigure,
                color=source_divider_color,
                linewidth=source_divider_linewidth,
            )
        )

    if title:
        grid_left = axs[0, 0].get_position().x0
        grid_right = axs[0, -1].get_position().x1
        fig.text(
            (grid_left + grid_right) * 0.5,
            title_y,
            title,
            ha="center",
            va="top",
            fontsize=title_fontsize,
        )

    if xlabel:
        grid_left = axs[-1, 0].get_position().x0
        grid_right = axs[-1, -1].get_position().x1
        fig.text(
            (grid_left + grid_right) * 0.5,
            xlabel_y,
            xlabel,
            ha="center",
            va="top",
            fontsize=xlabel_fontsize,
        )

    if ylabel:
        grid_top = axs[0, 0].get_position().y1
        grid_bottom = axs[-1, 0].get_position().y0
        grid_right = axs[0, -1].get_position().x1
        if ylabel_side.lower() == "right":
            ylabel_x = grid_right + ylabel_right_pad
        else:
            ylabel_x = max(0.03, axs[0, 0].get_position().x0 - ylabel_pad)
        fig.text(
            ylabel_x,
            (grid_top + grid_bottom) * 0.5,
            ylabel,
            rotation=ylabel_rotation,
            ha="center",
            va="center",
            fontsize=ylabel_fontsize,
        )

    fig.savefig(save_path, bbox_inches="tight", pad_inches=save_pad_inches)
    plt.close(fig)


def _stacked_preview_source_width_ratio(
    n_rows: int,
    grid_hspace: float,
    preview_gap_fraction: float,
    preview_height_fill_fraction: float,
) -> float:
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}.")

    clamped_fill = min(max(preview_height_fill_fraction, 0.05), 1.0)
    grid_height_ratio = n_rows * (1.0 + grid_hspace * (n_rows - 1) / max(n_rows, 1))
    stacked_preview_height_ratio = 2.0 + max(0.0, preview_gap_fraction)
    return clamped_fill * grid_height_ratio / stacked_preview_height_ratio


def _sample_output_stem(src_dir: str | Path, filepath: str, sample_offset: int) -> str:
    rel_path = Path(filepath).resolve().relative_to(Path(src_dir).resolve())
    stem = "__".join(rel_path.with_suffix("").parts)
    safe_stem = stem.replace(" ", "_")
    return f"{sample_offset:03d}_{safe_stem}"


def _resolve_batch_save_path(
    save_path: str | Path,
    src_dir: str | Path,
    filepath: str,
    sample_offset: int,
    num_samples: int,
) -> Path:
    save_path = Path(save_path)
    if num_samples == 1 and save_path.suffix:
        return save_path

    if save_path.suffix:
        out_dir = save_path.parent / save_path.stem
        suffix = save_path.suffix
    else:
        out_dir = save_path
        suffix = ".pdf"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{_sample_output_stem(src_dir, filepath, sample_offset)}{suffix}"


def _plot_single_sample_grid(
    x_src: torch.Tensor,
    seg_src: torch.Tensor | None,
    filepath: str,
    src_dir: str | Path,
    save_path: str | Path,
    sample_offset: int,
    num_samples: int,
    model: CycleNet,
    cond_mode: str,
    use_spade: bool,
    device: torch.device,
    sched: DiffusionSchedule,
    cfg_weights: list[float],
    noise_strengths: list[float],
    src_domain_idx: int,
    sampler: str,
    ddim_steps: int,
    eta: float,
    translation_seed: int,
    title: str | None,
    source_label: str,
    translated_label: str | None,
    xlabel: str,
    ylabel: str,
    dpi: int,
    scale: float,
    source_mode: str,
    source_width_ratio: float,
    source_gap_ratio: float,
    grid_wspace: float,
    grid_hspace: float,
    show_source_divider: bool,
    source_divider_color: str,
    source_divider_linewidth: float,
    source_divider_gap_fraction: float,
    source_label_position: str,
    source_label_fontsize: float,
    source_label_pad: float,
    row_label_side: str,
    row_label_right_pad: float,
    row_label_fontsize: float,
    col_label_fontsize: float,
    title_fontsize: float,
    title_y: float,
    xlabel_fontsize: float,
    xlabel_y: float,
    ylabel_fontsize: float,
    ylabel_pad: float,
    ylabel_right_pad: float,
    ylabel_side: str,
    ylabel_rotation: float | None,
    subplot_top_with_title: float,
    subplot_bottom_with_xlabels: float,
    selected_cfg_weight: float | None,
    selected_noise_strength: float | None,
    selected_border_color: str,
    selected_border_width: float,
    selected_border_inset: float,
    show_selected_preview_below_source: bool,
    preview_gap_fraction: float,
    preview_gap_color: str,
    preview_height_fill_fraction: float,
) -> Path:
    images: list[np.ndarray] = []
    for noise_strength in noise_strengths:
        for cfg_weight in cfg_weights:
            set_seed(translation_seed)
            images.append(
                translate_one(
                    model=model,
                    x_src=x_src,
                    seg_src=seg_src,
                    device=device,
                    sched=sched,
                    src_domain_idx=src_domain_idx,
                    sampler=sampler,
                    cfg_weight=float(cfg_weight),
                    noise_strength=float(noise_strength),
                    ddim_steps=ddim_steps,
                    eta=eta,
                    cond_mode=cond_mode,
                    use_spade=use_spade,
                )
            )

    selected_index = _selected_grid_index(
        noise_strengths=[float(v) for v in noise_strengths],
        cfg_weights=[float(v) for v in cfg_weights],
        selected_noise_strength=selected_noise_strength,
        selected_cfg_weight=selected_cfg_weight,
    )

    default_ylabel_rotation = 270.0 if ylabel_side.lower() == "right" else 90.0
    ylabel_rotation = default_ylabel_rotation if ylabel_rotation is None else ylabel_rotation
    source_image = tensor_to_numpy_image(x_src)
    row_labels = [f"{float(strength):g}" for strength in noise_strengths]
    col_labels = [f"{float(cfg_weight):g}" for cfg_weight in cfg_weights]
    plot_images = list(images)
    resolved_save_path = _resolve_batch_save_path(
        save_path=save_path,
        src_dir=src_dir,
        filepath=filepath,
        sample_offset=sample_offset,
        num_samples=num_samples,
    )
    resolved_save_path.parent.mkdir(parents=True, exist_ok=True)

    if show_selected_preview_below_source and selected_index is not None:
        _plot_grid_with_stacked_preview(
            images=plot_images,
            n_cols=len(cfg_weights),
            row_labels=row_labels,
            col_labels=col_labels,
            source_image=source_image,
            translated_image=plot_images[selected_index],
            source_label=source_label,
            translated_label=translated_label,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=resolved_save_path,
            dpi=dpi,
            scale=scale,
            source_width_ratio=source_width_ratio,
            source_gap_ratio=source_gap_ratio,
            grid_wspace=grid_wspace,
            grid_hspace=grid_hspace,
            show_source_divider=show_source_divider,
            source_divider_color=source_divider_color,
            source_divider_linewidth=source_divider_linewidth,
            source_divider_gap_fraction=source_divider_gap_fraction,
            source_label_fontsize=source_label_fontsize,
            source_label_pad=source_label_pad,
            row_label_side=row_label_side,
            row_label_right_pad=row_label_right_pad,
            row_label_fontsize=row_label_fontsize,
            col_label_fontsize=col_label_fontsize,
            title_fontsize=title_fontsize,
            title_y=title_y,
            xlabel_fontsize=xlabel_fontsize,
            xlabel_y=xlabel_y,
            ylabel_fontsize=ylabel_fontsize,
            ylabel_pad=ylabel_pad,
            ylabel_right_pad=ylabel_right_pad,
            ylabel_side=ylabel_side,
            ylabel_rotation=ylabel_rotation,
            subplot_top_with_title=subplot_top_with_title,
            subplot_top_without_title=0.98,
            subplot_bottom_with_xlabels=subplot_bottom_with_xlabels,
            subplot_bottom_without_xlabels=0.04,
            subplot_left_with_ylabels=0.12,
            subplot_left_without_ylabels=0.03,
            subplot_right=0.995,
            save_pad_inches=0.03,
            image_interpolation="nearest",
            preview_gap_fraction=preview_gap_fraction,
            preview_height_fill_fraction=preview_height_fill_fraction,
            selected_border_color=selected_border_color,
            selected_border_width=selected_border_width,
            selected_index=selected_index,
        )
        return resolved_save_path

    if selected_index is not None:
        plot_images[selected_index] = _add_image_border(
            image=plot_images[selected_index],
            color=selected_border_color,
            linewidth=selected_border_width,
            inset_fraction=selected_border_inset,
        )

    source_panel_image = source_image
    source_mode_for_plot = source_mode
    if selected_index is not None:
        source_panel_image = _build_vertical_preview_panel(
            top_image=source_image,
            bottom_image=plot_images[selected_index],
            gap_fraction=preview_gap_fraction,
            gap_color=preview_gap_color,
        )
        source_mode_for_plot = "separate"
        source_width_ratio = max(
            source_width_ratio,
            _stacked_preview_source_width_ratio(
                n_rows=len(noise_strengths),
                grid_hspace=grid_hspace,
                preview_gap_fraction=preview_gap_fraction,
                preview_height_fill_fraction=preview_height_fill_fraction,
            ),
        )

    plot_image_grid(
        images=plot_images,
        n_cols=len(cfg_weights),
        row_labels=row_labels,
        col_labels=col_labels,
        title=title,
        source_image=source_panel_image,
        source_label=source_label,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=resolved_save_path,
        dpi=dpi,
        scale=scale,
        source_mode=source_mode_for_plot,
        source_width_ratio=source_width_ratio,
        source_gap_ratio=source_gap_ratio,
        grid_wspace=grid_wspace,
        grid_hspace=grid_hspace,
        show_source_divider=show_source_divider,
        source_divider_color=source_divider_color,
        source_divider_linewidth=source_divider_linewidth,
        source_divider_gap_fraction=source_divider_gap_fraction,
        source_label_position=source_label_position,
        source_label_fontsize=source_label_fontsize,
        source_label_pad=source_label_pad,
        row_label_side=row_label_side,
        row_label_right_pad=row_label_right_pad,
        row_label_fontsize=row_label_fontsize,
        col_label_fontsize=col_label_fontsize,
        title_span="full",
        title_fontsize=title_fontsize,
        title_y=title_y,
        xlabel_fontsize=xlabel_fontsize,
        xlabel_y=xlabel_y,
        ylabel_fontsize=ylabel_fontsize,
        ylabel_pad=ylabel_pad,
        ylabel_right_pad=ylabel_right_pad,
        ylabel_side=ylabel_side,
        ylabel_rotation=ylabel_rotation,
        ylabel_multiline=False,
        subplot_top_with_title=subplot_top_with_title,
        subplot_bottom_with_xlabels=subplot_bottom_with_xlabels,
    )
    return resolved_save_path


def plot_checkpoint_sweep_grid(
    checkpoint_path: str | Path,
    src_dir: str | Path,
    save_path: str | Path,
    cfg_weights: list[float],
    noise_strengths: list[float],
    sample_index: int = 0,
    num_samples: int = 1,
    sample_rel_path: str | Path | None = None,
    image_size: int | None = None,
    rgb_parent_dirs: set[str] | str | list[str] | None = None,
    label_parent_dir: str | None = None,
    src_domain_idx: int = 0,
    sampler: str = "ddim",
    ddim_steps: int = 100,
    eta: float = 0.0,
    model_key: str = "ema_model",
    seed: int = 42,
    translation_seed: int | None = None,
    title: str | None = None,
    source_label: str = "Source (Sim)",
    translated_label: str | None = "Translated",
    xlabel: str = "CFG weight ($w$)",
    ylabel: str = "Noise strength ($\\gamma$)",
    dpi: int = 300,
    scale: float = 1.0,
    source_mode: str = "grid_column",
    source_width_ratio: float = 1.15,
    source_gap_ratio: float = 0.10,
    grid_wspace: float = 0.045,
    grid_hspace: float = 0.045,
    show_source_divider: bool = True,
    source_divider_color: str = "black",
    source_divider_linewidth: float = 0.8,
    source_divider_gap_fraction: float = 0.5,
    source_label_position: str = "top",
    source_label_fontsize: float = 10.0,
    source_label_pad: float = 3.0,
    row_label_side: str = "right",
    row_label_right_pad: float = 0.01,
    row_label_fontsize: float = 11.0,
    col_label_fontsize: float = 11.0,
    title_fontsize: float = 11.0,
    title_y: float = 0.955,
    xlabel_fontsize: float = 12.0,
    xlabel_y: float = 0.035,
    ylabel_fontsize: float = 12.0,
    ylabel_pad: float = 0.08,
    ylabel_right_pad: float = 0.07,
    ylabel_side: str = "right",
    ylabel_rotation: float | None = None,
    subplot_top_with_title: float = 0.93,
    subplot_bottom_with_xlabels: float = 0.10,
    selected_cfg_weight: float | None = None,
    selected_noise_strength: float | None = None,
    selected_border_color: str = "#C84C1A",
    selected_border_width: float = 2.2,
    selected_border_inset: float = 0.01,
    show_selected_preview_below_source: bool = True,
    preview_gap_fraction: float = 0.06,
    preview_gap_color: str = "white",
    preview_height_fill_fraction: float = 0.78,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_dir = _checkpoint_training_dir(checkpoint_path)
    cyclenet_config = load_config(train_dir / "config.yaml")

    unet_train_dir = Path(cyclenet_config.run.unet_ckpt).parent.parent
    unet_config = load_config(unet_train_dir / "config.yaml")

    image_size = int(image_size or cfg_select(cyclenet_config, "data.image_size", 224))
    rgb_parent_dirs = as_parent_dir_set(
        rgb_parent_dirs if rgb_parent_dirs is not None else cfg_select(cyclenet_config, "data.rgb_parent_dirs", None)
    )
    label_parent_dir = str(
        label_parent_dir
        if label_parent_dir is not None
        else cfg_select(cyclenet_config, "data.label_parent_dir", "gt_ss_mask")
    )
    num_seg_classes = int(cfg_select(cyclenet_config, "model.num_seg_classes", 8))
    requires_segmentation = _requires_segmentation(cyclenet_config)

    set_seed(seed)
    if translation_seed is None:
        translation_seed = seed

    dataset = load_translate_dataset(
        src_dir=src_dir,
        image_size=image_size,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
        num_seg_classes=num_seg_classes,
        requires_segmentation=requires_segmentation,
    )
    selected_samples = _select_sample_requests(
        dataset=dataset,
        src_dir=src_dir,
        requires_segmentation=requires_segmentation,
        sample_index=sample_index,
        num_samples=num_samples,
        sample_rel_path=sample_rel_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cond_mode, use_spade = build_model(cyclenet_config, unet_config, device)
    load_checkpoint(model, checkpoint_path, model_key=model_key)
    model.eval()
    sched = build_schedule(unet_config, device)

    saved_paths: list[Path] = []
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        for sample_offset, (x_src, seg_src, filepath) in enumerate(selected_samples):
            saved_path = _plot_single_sample_grid(
                x_src=x_src,
                seg_src=seg_src,
                filepath=filepath,
                src_dir=src_dir,
                save_path=save_path,
                sample_offset=sample_offset,
                num_samples=num_samples,
                model=model,
                cond_mode=cond_mode,
                use_spade=use_spade,
                device=device,
                sched=sched,
                cfg_weights=cfg_weights,
                noise_strengths=noise_strengths,
                src_domain_idx=src_domain_idx,
                sampler=sampler,
                ddim_steps=ddim_steps,
                eta=eta,
                translation_seed=translation_seed,
                title=title,
                source_label=source_label,
                translated_label=translated_label,
                xlabel=xlabel,
                ylabel=ylabel,
                dpi=dpi,
                scale=scale,
                source_mode=source_mode,
                source_width_ratio=source_width_ratio,
                source_gap_ratio=source_gap_ratio,
                grid_wspace=grid_wspace,
                grid_hspace=grid_hspace,
                show_source_divider=show_source_divider,
                source_divider_color=source_divider_color,
                source_divider_linewidth=source_divider_linewidth,
                source_divider_gap_fraction=source_divider_gap_fraction,
                source_label_position=source_label_position,
                source_label_fontsize=source_label_fontsize,
                source_label_pad=source_label_pad,
                row_label_side=row_label_side,
                row_label_right_pad=row_label_right_pad,
                row_label_fontsize=row_label_fontsize,
                col_label_fontsize=col_label_fontsize,
                title_fontsize=title_fontsize,
                title_y=title_y,
                xlabel_fontsize=xlabel_fontsize,
                xlabel_y=xlabel_y,
                ylabel_fontsize=ylabel_fontsize,
                ylabel_pad=ylabel_pad,
                ylabel_right_pad=ylabel_right_pad,
                ylabel_side=ylabel_side,
                ylabel_rotation=ylabel_rotation,
                subplot_top_with_title=subplot_top_with_title,
                subplot_bottom_with_xlabels=subplot_bottom_with_xlabels,
                selected_cfg_weight=selected_cfg_weight,
                selected_noise_strength=selected_noise_strength,
                selected_border_color=selected_border_color,
                selected_border_width=selected_border_width,
                selected_border_inset=selected_border_inset,
                show_selected_preview_below_source=show_selected_preview_below_source,
                preview_gap_fraction=preview_gap_fraction,
                preview_gap_color=preview_gap_color,
                preview_height_fill_fraction=preview_height_fill_fraction,
            )
            saved_paths.append(saved_path)
            rel_path = Path(filepath).resolve().relative_to(Path(src_dir).resolve())
            print(f"Saved CFG/strength sweep grid for {rel_path} to {saved_path}")

    if len(saved_paths) == 1:
        return saved_paths[0]

    save_path = Path(save_path)
    if save_path.suffix:
        return save_path.parent / save_path.stem
    return save_path


def main() -> None:
    # CycleNet checkpoint to sample directly for the thesis sweep grid.
    checkpoint_path = "/develop/code/runs/cyclenet/remote_sensing/seg/oem_only_seg_only/training/checkpoints/step-40000.ckpt"
    # Source-image root directory used to build the translate dataset.
    src_dir = "/develop/data/remote_sensing/tiled/projection/sim_proj"
    # Output PDF path for the rendered CFG/noise-strength grid.
    save_path = "/develop/code/eval/thesis/sweep_grid/sweep_grid.pdf"
    # CFG values to sweep across the grid columns.
    cfg_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Noise-strength values to sweep across the grid rows.
    noise_strengths = [0.1, 0.2, 0.3, 0.4, 0.5]
    # Dataset index of the source image to plot when `sample_rel_path` is not set.
    sample_index = 0
    # Number of consecutive dataset samples to process starting from `sample_index`.
    num_samples = 10
    # Relative path under `src_dir` for an exact source image to plot. Use `None` to use `sample_index`.
    sample_rel_path = None
    # Optional image-size override for the source dataset. Use `None` to inherit from the training config.
    image_size = None
    # Optional allowed RGB parent directories for source-image discovery. Use `None` to inherit from config.
    rgb_parent_dirs = None
    # Optional label parent directory override for segmentation-conditioned checkpoints.
    label_parent_dir = None
    # Source-domain index used during translation. Sim-to-real runs typically use `0`.
    src_domain_idx = 0
    # Diffusion sampler. Supported values: `ddim` and `ddpm`.
    sampler = "ddim"
    # Number of DDIM steps when `sampler='ddim'`.
    ddim_steps = 50
    # DDIM eta parameter when `sampler='ddim'`.
    eta = 0.0
    # Checkpoint state-dict key to load from the `.ckpt` file.
    model_key = "ema_model"
    # Global random seed for reproducibility.
    seed = 42
    # Translation noise seed reset before each cell so the sweep varies only by CFG/strength.
    translation_seed = 42
    # Optional figure title.
    title = None
    # Label shown above the source-image column.
    source_label = "Simulated Source"
    # Label shown above the selected translated preview image.
    translated_label = "Translated"
    # X-axis label for the grid.
    xlabel = "CFG weight ($w$)"
    # Y-axis label for the grid.
    ylabel = "Noise strength ($s$)"
    # Saved figure DPI.
    dpi = 300
    # Figure scale multiplier applied on top of the image-driven base size.
    scale = 1.0
    # Source-image layout mode. Supported values: `grid_column` and `separate`.
    source_mode = "grid_column"
    # Relative width of the source-image column.
    source_width_ratio = 1.15
    # Horizontal gap between the source-image column and the sweep grid.
    source_gap_ratio = 0.10
    # Horizontal spacing between grid cells.
    grid_wspace = 0.045
    # Vertical spacing between grid cells.
    grid_hspace = 0.045
    # Whether to draw a divider between the source image and the sweep grid.
    show_source_divider = True
    # Divider color between the source image and the sweep grid.
    source_divider_color = "black"
    # Divider linewidth between the source image and the sweep grid.
    source_divider_linewidth = 0.8
    # Divider x-position within the source/grid gap, as a fraction of the gap width.
    source_divider_gap_fraction = 0.5
    # Whether the source label appears at the `top` or `bottom` of the source column.
    source_label_position = "top"
    # Font size for the source-image label.
    source_label_fontsize = 10.0
    # Padding above the source-image label.
    source_label_pad = 3.0
    # Side on which row labels are drawn. Supported values: `left` and `right`.
    row_label_side = "right"
    # Padding between the grid and right-side row labels.
    row_label_right_pad = 0.03
    # Font size for the row labels.
    row_label_fontsize = 11.0
    # Font size for the column labels.
    col_label_fontsize = 11.0
    # Font size for the title.
    title_fontsize = 11.0
    # Vertical figure coordinate for the title.
    title_y = 0.955
    # Font size for the x-axis label.
    xlabel_fontsize = 12.0
    # Vertical figure coordinate for the x-axis label.
    xlabel_y = 0.035
    # Font size for the y-axis label.
    ylabel_fontsize = 12.0
    # Padding between the grid and a left-side y-axis label.
    ylabel_pad = 0.08
    # Padding between the grid and a right-side y-axis label.
    ylabel_right_pad = 0.07
    # Side on which the y-axis label is drawn. Supported values: `left` and `right`.
    ylabel_side = "right"
    # Rotation for the y-axis label. Use `None` for the side-dependent default.
    ylabel_rotation = None
    # Top subplot margin when a title is present.
    subplot_top_with_title = 0.93
    # Bottom subplot margin when x-axis labels are present.
    subplot_bottom_with_xlabels = 0.10
    # Selected CFG value whose translated image should receive a border.
    selected_cfg_weight = 1.0
    # Selected noise strength whose translated image should receive a border.
    selected_noise_strength = 0.5
    # Border color for the selected operating-point cell.
    selected_border_color = "#C84C1A"
    # Border linewidth for the selected operating-point cell.
    selected_border_width = 2
    # Inset fraction used to keep the selected-cell border slightly inside the axes bounds.
    selected_border_inset = 0.0
    # Whether to show a stacked left preview with source on top and selected translated output on bottom.
    show_selected_preview_below_source = True
    # Vertical gap between the source and selected preview images as a fraction of one image height.
    preview_gap_fraction = 0.01
    # Color of the gap between the source and selected preview images.
    preview_gap_color = "white"
    # Target fraction of the full left preview-panel height occupied by the stacked source/translated images.
    preview_height_fill_fraction = 0.78

    saved_path = plot_checkpoint_sweep_grid(
        checkpoint_path=checkpoint_path,
        src_dir=src_dir,
        save_path=save_path,
        cfg_weights=cfg_weights,
        noise_strengths=noise_strengths,
        sample_index=sample_index,
        num_samples=num_samples,
        sample_rel_path=sample_rel_path,
        image_size=image_size,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
        src_domain_idx=src_domain_idx,
        sampler=sampler,
        ddim_steps=ddim_steps,
        eta=eta,
        model_key=model_key,
        seed=seed,
        translation_seed=translation_seed,
        title=title,
        source_label=source_label,
        translated_label=translated_label,
        xlabel=xlabel,
        ylabel=ylabel,
        dpi=dpi,
        scale=scale,
        source_mode=source_mode,
        source_width_ratio=source_width_ratio,
        source_gap_ratio=source_gap_ratio,
        grid_wspace=grid_wspace,
        grid_hspace=grid_hspace,
        show_source_divider=show_source_divider,
        source_divider_color=source_divider_color,
        source_divider_linewidth=source_divider_linewidth,
        source_divider_gap_fraction=source_divider_gap_fraction,
        source_label_position=source_label_position,
        source_label_fontsize=source_label_fontsize,
        source_label_pad=source_label_pad,
        row_label_side=row_label_side,
        row_label_right_pad=row_label_right_pad,
        row_label_fontsize=row_label_fontsize,
        col_label_fontsize=col_label_fontsize,
        title_fontsize=title_fontsize,
        title_y=title_y,
        xlabel_fontsize=xlabel_fontsize,
        xlabel_y=xlabel_y,
        ylabel_fontsize=ylabel_fontsize,
        ylabel_pad=ylabel_pad,
        ylabel_right_pad=ylabel_right_pad,
        ylabel_side=ylabel_side,
        ylabel_rotation=ylabel_rotation,
        subplot_top_with_title=subplot_top_with_title,
        subplot_bottom_with_xlabels=subplot_bottom_with_xlabels,
        selected_cfg_weight=selected_cfg_weight,
        selected_noise_strength=selected_noise_strength,
        selected_border_color=selected_border_color,
        selected_border_width=selected_border_width,
        selected_border_inset=selected_border_inset,
        show_selected_preview_below_source=show_selected_preview_below_source,
        preview_gap_fraction=preview_gap_fraction,
        preview_gap_color=preview_gap_color,
        preview_height_fill_fraction=preview_height_fill_fraction,
    )
    print(f"Saved thesis sweep grid to {saved_path}")


if __name__ == "__main__":
    main()
