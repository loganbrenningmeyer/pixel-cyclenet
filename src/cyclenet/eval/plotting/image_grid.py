import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import torch

from .set_style import apply_style
from cyclenet.data import TranslateDataset
from cyclenet.diffusion import DiffusionSchedule, q_sample
from cyclenet.diffusion.sampling import ddim_steps_from_strength

apply_style()


def plot_image_grid(
    images: list[np.ndarray],
    n_cols: int,
    row_labels: list[str],
    col_labels: list[str],
    title: str | None = None,
    source_image: np.ndarray | None = None,
    source_label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | None = None,
    dpi: int = 300,
    scale: float = 1.0,
    source_mode: str = "separate",
    source_width_ratio: float = 1.15,
    source_gap_ratio: float = 0.25,
    grid_wspace: float = 0.045,
    grid_hspace: float = 0.045,
    show_source_divider: bool = False,
    source_divider_color: str = "black",
    source_divider_linewidth: float = 0.8,
    source_divider_gap_fraction: float = 0.5,
    source_label_position: str = "bottom",
    source_label_fontsize: float = 8,
    source_label_pad: float = 3,
    row_label_side: str = "right",
    row_label_x: float = -0.08,
    row_label_right_pad: float = 0.01,
    row_label_fontsize: float | None = None,
    col_label_y: float = -0.08,
    col_label_fontsize: float | None = None,
    title_span: str = "grid",
    title_fontsize: float = 9,
    title_y: float = 0.965,
    xlabel_fontsize: float | None = None,
    xlabel_y: float = 0.055,
    ylabel_fontsize: float | None = None,
    ylabel_pad: float = 0.10,
    ylabel_right_pad: float = 0.07,
    ylabel_side: str = "left",
    ylabel_rotation: float = 0.0,
    ylabel_multiline: bool = True,
    subplot_top_with_title: float = 0.90,
    subplot_top_without_title: float = 0.98,
    subplot_bottom_with_xlabels: float = 0.14,
    subplot_bottom_without_xlabels: float = 0.04,
    subplot_left_with_ylabels: float = 0.12,
    subplot_left_without_ylabels: float = 0.03,
    subplot_right: float = 0.995,
    save_pad_inches: float = 0.03,
    image_interpolation: str = "nearest",
):
    """
    Plots image grid with specified n_cols and axes labels
        - cfg/strength images
        - denoising trajectory images
    """
    # -------------------------
    # Determine n_rows / figure size
    # -------------------------
    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))
    source_mode = source_mode.lower()
    if source_mode not in {"separate", "grid_column"}:
        raise ValueError(f"Unsupported source_mode '{source_mode}'.")
    source_label_position = source_label_position.lower()
    if source_label_position not in {"top", "bottom"}:
        raise ValueError(f"Unsupported source_label_position '{source_label_position}'.")
    title_span = title_span.lower()
    if title_span not in {"grid", "full"}:
        raise ValueError(f"Unsupported title_span '{title_span}'.")

    H, W = images[0].shape[:2]
    top = subplot_top_with_title if title else subplot_top_without_title
    bottom = subplot_bottom_with_xlabels if col_labels is not None or xlabel else subplot_bottom_without_xlabels
    left = subplot_left_with_ylabels if row_labels is not None or ylabel else subplot_left_without_ylabels
    right = subplot_right

    if source_image is not None and source_mode == "separate":
        grid_width_ratio = n_cols + grid_wspace * max(n_cols - 1, 0)
        outer_width_ratios = [source_width_ratio, source_gap_ratio, grid_width_ratio]
        axes_width_px = W * sum(outer_width_ratios)
    elif source_image is not None and source_mode == "grid_column":
        grid_width_ratio = n_cols + grid_wspace * max(n_cols - 1, 0)
        outer_width_ratios = [source_width_ratio, source_gap_ratio, grid_width_ratio]
        axes_width_px = W * sum(outer_width_ratios)
    else:
        width_ratios = [1.0] * n_cols
        total_cols = len(width_ratios)
        axes_width_px = W * sum(width_ratios) * (1.0 + grid_wspace * (total_cols - 1) / max(total_cols, 1))

    # Size the figure from the desired panel size after reserving outer margins.
    # This prevents apparent extra horizontal spacing caused by the axes being
    # squeezed by title/label padding rather than by true GridSpec gaps.
    axes_height_px = H * n_rows * (1.0 + grid_hspace * (n_rows - 1) / max(n_rows, 1))
    fig_width = (axes_width_px / max(right - left, 1e-6)) / dpi * scale
    fig_height = (axes_height_px / max(top - bottom, 1e-6)) / dpi * scale

    # -------------------------
    # Plot image grid
    # -------------------------
    fig = plt.figure(figsize=(fig_width, fig_height))
    total_cols = n_cols + 1 if source_image is not None and source_mode == "grid_column" else n_cols
    axs = np.empty((n_rows, total_cols), dtype=object)

    source_ax = None
    if source_image is not None and source_mode == "separate":
        outer_gs = GridSpec(
            1,
            3,
            figure=fig,
            width_ratios=outer_width_ratios,
            wspace=0.0,
            hspace=grid_hspace,
        )
        grid_gs = outer_gs[0, 2].subgridspec(n_rows, n_cols, wspace=grid_wspace, hspace=grid_hspace)
        source_ax = fig.add_subplot(outer_gs[0, 0])
        source_ax.imshow(source_image, interpolation=image_interpolation)
        source_ax.axis("off")
        if source_label:
            source_ax.set_title(source_label, fontsize=source_label_fontsize, pad=source_label_pad)
        for row in range(n_rows):
            for col in range(n_cols):
                axs[row, col] = fig.add_subplot(grid_gs[row, col])
    elif source_image is not None and source_mode == "grid_column":
        outer_gs = GridSpec(
            1,
            3,
            figure=fig,
            width_ratios=outer_width_ratios,
            wspace=0.0,
            hspace=grid_hspace,
        )
        source_gs = outer_gs[0, 0].subgridspec(n_rows, 1, hspace=grid_hspace)
        grid_gs = outer_gs[0, 2].subgridspec(n_rows, n_cols, wspace=grid_wspace, hspace=grid_hspace)
        for row in range(n_rows):
            axs[row, 0] = fig.add_subplot(source_gs[row, 0])
            axs[row, 0].imshow(source_image, interpolation=image_interpolation)
            axs[row, 0].axis("off")
            for col in range(n_cols):
                axs[row, col + 1] = fig.add_subplot(grid_gs[row, col])
        if source_label and source_label_position == "top":
            axs[0, 0].set_title(source_label, fontsize=source_label_fontsize, pad=source_label_pad)
    else:
        gs = GridSpec(
            n_rows,
            n_cols,
            figure=fig,
            wspace=grid_wspace,
            hspace=grid_hspace,
        )
        for row in range(n_rows):
            for col in range(n_cols):
                axs[row, col] = fig.add_subplot(gs[row, col])

    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        if source_image is not None and source_mode == "grid_column":
            col += 1

        ax = axs[row, col]
        ax.imshow(img, interpolation=image_interpolation)
        ax.axis("off")

    # -- Hide unused axes
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        if source_image is not None and source_mode == "grid_column":
            col += 1
        axs[row, col].axis("off")

    if row_labels is not None:
        row_label_side = row_label_side.lower()
        if row_label_side not in {"left", "right"}:
            raise ValueError(f"Unsupported row_label_side '{row_label_side}'.")
        for row, label in enumerate(row_labels[:n_rows]):
            target_ax = axs[row, -1]
            label_x = row_label_x
            ha = "right"
            if row_label_side == "left":
                target_ax = axs[row, 0]
            elif source_image is not None and source_mode in {"separate", "grid_column"}:
                label_x = 1.0 + row_label_right_pad
                ha = "left"
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
        display_col_labels = list(col_labels[:n_cols])
        label_col_offset = 0
        if source_image is not None and source_mode == "grid_column":
            if source_label_position == "bottom":
                display_col_labels = [source_label or "Source"] + display_col_labels
            else:
                label_col_offset = 1
        for col, label in enumerate(display_col_labels[: total_cols - label_col_offset]):
            axs[-1, col + label_col_offset].text(
                0.5,
                col_label_y,
                label,
                transform=axs[-1, col + label_col_offset].transAxes,
                ha="center",
                va="top",
                clip_on=False,
                fontsize=col_label_fontsize,
            )

    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=0.0 if source_image is not None and source_mode in {"separate", "grid_column"} else grid_wspace,
        hspace=grid_hspace,
    )

    if source_image is not None and show_source_divider and source_mode in {"separate", "grid_column"}:
        if source_mode == "separate":
            gap_left = source_ax.get_position().x1
            gap_right = axs[0, 0].get_position().x0
            divider_y0 = min(source_ax.get_position().y0, axs[-1, 0].get_position().y0)
            divider_y1 = max(source_ax.get_position().y1, axs[0, 0].get_position().y1)
        else:
            source_positions = [axs[row, 0].get_position() for row in range(n_rows)]
            gap_left = max(pos.x1 for pos in source_positions)
            gap_right = axs[0, 1].get_position().x0
            divider_y0 = min(source_positions[-1].y0, axs[-1, 1].get_position().y0)
            divider_y1 = max(source_positions[0].y1, axs[0, 1].get_position().y1)

        divider_x = gap_left + (gap_right - gap_left) * source_divider_gap_fraction
        fig.add_artist(
            Line2D(
                [divider_x, divider_x],
                [divider_y0, divider_y1],
                transform=fig.transFigure,
                color=source_divider_color,
                linewidth=source_divider_linewidth,
            )
        )

    first_grid_col = 1 if source_image is not None and source_mode == "grid_column" else 0
    if title:
        title_left_col = 0 if title_span == "full" else first_grid_col
        grid_left = axs[0, title_left_col].get_position().x0
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
        grid_left = axs[-1, first_grid_col].get_position().x0
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
        ylabel_side = ylabel_side.lower()
        if ylabel_side not in {"left", "right"}:
            raise ValueError(f"Unsupported ylabel_side '{ylabel_side}'.")
        grid_top = axs[0, 0].get_position().y1
        grid_bottom = axs[-1, 0].get_position().y0
        grid_right = axs[0, -1].get_position().x1
        if ylabel_side == "right":
            ylabel_x = grid_right + ylabel_right_pad
        else:
            ylabel_x = max(0.03, axs[0, 0].get_position().x0 - ylabel_pad)
        ylabel_text = ylabel.replace(" ", "\n") if ylabel_multiline else ylabel
        fig.text(
            ylabel_x,
            (grid_top + grid_bottom) * 0.5,
            ylabel_text,
            rotation=ylabel_rotation,
            ha="center",
            va="center",
            multialignment="center",
            fontsize=ylabel_fontsize,
        )

    # -------------------------
    # Save figure
    # -------------------------
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=save_pad_inches)
    else:
        fig.show()

    plt.close(fig)
