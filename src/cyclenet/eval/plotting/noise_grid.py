from pathlib import Path
import random

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import torch
import numpy as np

from cyclenet.diffusion import DiffusionSchedule, q_sample
from cyclenet.diffusion.sampling import ddim_steps_from_strength
from cyclenet.data import TranslateDataset
from cyclenet.eval.plotting.set_style import apply_style

apply_style()


def tensor_to_numpy_image(img: torch.Tensor) -> np.ndarray:
    img = ((img.clamp(-1.0, 1.0) + 1.0) / 2.0).float().cpu()
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_sample_indices(n_total: int, n_select: int | None, seed: int) -> list[int]:
    if n_total <= 0:
        return []
    if n_select is None or n_select <= 0 or n_select >= n_total:
        return list(range(n_total))

    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), k=n_select))


def as_parent_dir_set(value: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(v) for v in value}


def q_sample_from_strength(
    x_src: torch.Tensor,
    eps: torch.Tensor,
    strength: float,
    sched: DiffusionSchedule,
    ddim_steps: int,
) -> torch.Tensor:
    strength = float(max(0.0, min(1.0, strength)))
    if strength == 0.0:
        return x_src

    t_steps = ddim_steps_from_strength(sched, ddim_steps, strength)
    t_noise = torch.full((x_src.shape[0],), t_steps[-1], device=x_src.device, dtype=torch.long)
    return q_sample(x_src, t_noise, eps, sched)


def plot_noise_grid(
    source_image: np.ndarray,
    noised_images: list[np.ndarray],
    col_labels: list[str],
    title: str | None = None,
    source_label: str | None = None,
    source_bottom_label: str | None = None,
    xlabel: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    scale: float = 1.0,
    source_width_ratio: float = 1.0,  # Relative width of the source-image column.
    source_gap_ratio: float = 0.25,  # Gap between the source image and the noised-image grid, in panel-width units.
    grid_wspace: float = 0.06,  # Horizontal spacing between image panels.
    grid_hspace: float = 0.06,  # Vertical spacing between image panels.
    show_source_divider: bool = False,  # Whether to draw a divider centered in the source-to-grid gap.
    source_divider_color: str = "black",  # Color of the divider between source and noised images.
    source_divider_linewidth: float = 0.8,  # Line width of the divider between source and noised images.
    source_label_fontsize: float = 8,  # Font size for the source-image title.
    source_label_pad: float = 4,  # Gap between the source image and its title.
    col_label_fontsize: float | None = None,  # Font size for per-column labels; None uses Matplotlib defaults.
    col_label_y: float = -0.08,  # Vertical offset for per-column labels in axes coordinates.
    title_fontsize: float = 9,  # Font size for the main title above the noised-image grid.
    title_y: float = 0.985,  # Vertical position of the main title in figure coordinates.
    xlabel_fontsize: float | None = None,  # Font size for the bottom x-label; None uses Matplotlib defaults.
    xlabel_y: float = 0.03,  # Vertical position of the bottom x-label in figure coordinates.
    subplot_left: float = 0.03,  # Left outer margin for the full figure.
    subplot_right: float = 0.995,  # Right outer margin for the full figure.
    subplot_top_with_title: float = 0.93,  # Top margin when a title is present.
    subplot_top_without_title: float = 0.98,  # Top margin when no title is present.
    subplot_bottom_with_xlabel: float = 0.12,  # Bottom margin when bottom labels/x-label are present.
    subplot_bottom_without_xlabel: float = 0.06,  # Bottom margin when no x-label is present.
    save_pad_inches: float = 0.03,  # Extra whitespace around the saved figure.
):
    n_cols = len(noised_images)
    if n_cols == 0:
        raise ValueError("plot_noise_grid requires at least one noised image.")

    H, W = noised_images[0].shape[:2]
    has_top_annotations = bool(title or source_label)
    top = subplot_top_with_title if has_top_annotations else subplot_top_without_title
    has_bottom_annotations = bool(source_bottom_label or xlabel or any(col_labels[:n_cols]))
    bottom = subplot_bottom_with_xlabel if has_bottom_annotations else subplot_bottom_without_xlabel
    grid_width_ratio = n_cols + grid_wspace * max(n_cols - 1, 0)
    outer_width_ratios = [source_width_ratio, source_gap_ratio, grid_width_ratio]

    # Size the figure from the desired image-panel size rather than only from the
    # raw pixel width. This keeps square images square after reserving room for
    # titles/labels, keeps source-gap spacing independent from grid spacing, and
    # makes grid_wspace=0 produce touching noised-image panels.
    axes_width_px = W * sum(outer_width_ratios)
    axes_height_px = H
    fig_width = (axes_width_px / max(subplot_right - subplot_left, 1e-6)) / dpi * scale
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
    grid_gs = outer_gs[0, 2].subgridspec(1, n_cols, wspace=grid_wspace, hspace=grid_hspace)

    source_ax = fig.add_subplot(outer_gs[0, 0])
    source_ax.imshow(source_image, interpolation="nearest")
    source_ax.axis("off")
    if source_label:
        source_ax.set_title(
            source_label,
            fontsize=source_label_fontsize,
            pad=source_label_pad,
        )

    if source_bottom_label:
        source_ax.text(
            0.5,
            col_label_y,
            source_bottom_label,
            transform=source_ax.transAxes,
            ha="center",
            va="top",
            clip_on=False,
            fontsize=col_label_fontsize,
        )

    axs = []
    for col, img in enumerate(noised_images):
        ax = fig.add_subplot(grid_gs[0, col])
        ax.imshow(img, interpolation="nearest")
        ax.axis("off")
        axs.append(ax)

    for col, label in enumerate(col_labels[:n_cols]):
        axs[col].text(
            0.5,
            col_label_y,
            label,
            transform=axs[col].transAxes,
            ha="center",
            va="top",
            clip_on=False,
            fontsize=col_label_fontsize,
        )

    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        top=top,
        bottom=bottom,
        wspace=0.0,
        hspace=grid_hspace,
    )

    if show_source_divider:
        divider_x = (source_ax.get_position().x1 + axs[0].get_position().x0) * 0.5
        divider_y0 = min(source_ax.get_position().y0, axs[0].get_position().y0)
        divider_y1 = max(source_ax.get_position().y1, axs[0].get_position().y1)
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
        fig.text(
            0.5,
            title_y,
            title,
            ha="center",
            va="top",
            fontsize=title_fontsize,
        )

    if xlabel:
        fig.text(
            0.5,
            xlabel_y,
            xlabel,
            ha="center",
            va="top",
            fontsize=xlabel_fontsize,
        )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=save_pad_inches)
    else:
        fig.show()

    plt.close(fig)


def plot_noise_grid_rows(
    source_images: list[np.ndarray],
    noised_image_rows: list[list[np.ndarray]],
    col_labels: list[str],
    title: str | None = None,
    source_label: str | None = None,
    source_bottom_label: str | None = None,
    xlabel: str | None = None,
    row_labels: list[str] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    scale: float = 1.0,
    source_width_ratio: float = 1.0,  # Relative width of the source-image column.
    source_gap_ratio: float = 0.25,  # Gap between the source image and the noised-image grid, in panel-width units.
    grid_wspace: float = 0.06,  # Horizontal spacing between image panels.
    grid_hspace: float = 0.06,  # Vertical spacing between image panels.
    show_source_divider: bool = False,  # Whether to draw a divider centered in the source-to-grid gap.
    source_divider_color: str = "black",  # Color of the divider between source and noised images.
    source_divider_linewidth: float = 0.8,  # Line width of the divider between source and noised images.
    source_label_fontsize: float = 8,  # Font size for the source-image column title.
    source_label_pad: float = 4,  # Gap between the source image and its title.
    col_label_fontsize: float | None = None,  # Font size for per-column labels; None uses Matplotlib defaults.
    col_label_y: float = -0.08,  # Vertical offset for per-column labels in bottom-row axes coordinates.
    title_fontsize: float = 9,  # Font size for the main title above the noised-image grid.
    title_y: float = 0.985,  # Vertical position of the main title in figure coordinates.
    xlabel_fontsize: float | None = None,  # Font size for the bottom x-label; None uses Matplotlib defaults.
    xlabel_y: float = 0.03,  # Vertical position of the bottom x-label in figure coordinates.
    row_label_fontsize: float = 7,  # Font size for per-row sample labels.
    row_label_x: float = -0.04,  # Horizontal offset for row labels in source-axes coordinates.
    subplot_left: float = 0.03,  # Left outer margin for the full figure.
    subplot_right: float = 0.995,  # Right outer margin for the full figure.
    subplot_top_with_title: float = 0.93,  # Top margin when a title is present.
    subplot_top_without_title: float = 0.98,  # Top margin when no title is present.
    subplot_bottom_with_xlabel: float = 0.12,  # Bottom margin when bottom labels/x-label are present.
    subplot_bottom_without_xlabel: float = 0.06,  # Bottom margin when no x-label is present.
    save_pad_inches: float = 0.03,  # Extra whitespace around the saved figure.
):
    n_rows = len(source_images)
    if n_rows == 0:
        raise ValueError("plot_noise_grid_rows requires at least one source image.")
    if len(noised_image_rows) != n_rows:
        raise ValueError("source_images and noised_image_rows must have the same length.")
    if row_labels is not None and len(row_labels) != n_rows:
        raise ValueError("row_labels must match the number of source images.")

    n_cols = len(col_labels)
    if n_cols == 0:
        raise ValueError("plot_noise_grid_rows requires at least one column label.")

    for row_images in noised_image_rows:
        if len(row_images) != n_cols:
            raise ValueError("Each noised-image row must match the number of column labels.")

    H, W = noised_image_rows[0][0].shape[:2]
    has_top_annotations = bool(title or source_label)
    top = subplot_top_with_title if has_top_annotations else subplot_top_without_title
    has_bottom_annotations = bool(source_bottom_label or xlabel or any(col_labels))
    bottom = subplot_bottom_with_xlabel if has_bottom_annotations else subplot_bottom_without_xlabel
    grid_width_ratio = n_cols + grid_wspace * max(n_cols - 1, 0)
    grid_height_ratio = n_rows + grid_hspace * max(n_rows - 1, 0)
    outer_width_ratios = [source_width_ratio, source_gap_ratio, grid_width_ratio]

    axes_width_px = W * sum(outer_width_ratios)
    axes_height_px = H * grid_height_ratio
    fig_width = (axes_width_px / max(subplot_right - subplot_left, 1e-6)) / dpi * scale
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
    source_gs = outer_gs[0, 0].subgridspec(n_rows, 1, hspace=grid_hspace)
    grid_gs = outer_gs[0, 2].subgridspec(n_rows, n_cols, wspace=grid_wspace, hspace=grid_hspace)

    source_axes = []
    grid_axes = []
    for row, (source_image, row_images) in enumerate(zip(source_images, noised_image_rows)):
        source_ax = fig.add_subplot(source_gs[row, 0])
        source_ax.imshow(source_image, interpolation="nearest")
        source_ax.axis("off")
        source_axes.append(source_ax)

        row_axes = []
        for col, img in enumerate(row_images):
            ax = fig.add_subplot(grid_gs[row, col])
            ax.imshow(img, interpolation="nearest")
            ax.axis("off")
            row_axes.append(ax)
        grid_axes.append(row_axes)

    if source_label:
        source_axes[0].set_title(
            source_label,
            fontsize=source_label_fontsize,
            pad=source_label_pad,
        )

    if source_bottom_label:
        source_axes[-1].text(
            0.5,
            col_label_y,
            source_bottom_label,
            transform=source_axes[-1].transAxes,
            ha="center",
            va="top",
            clip_on=False,
            fontsize=col_label_fontsize,
        )

    for col, label in enumerate(col_labels):
        grid_axes[-1][col].text(
            0.5,
            col_label_y,
            label,
            transform=grid_axes[-1][col].transAxes,
            ha="center",
            va="top",
            clip_on=False,
            fontsize=col_label_fontsize,
        )

    if row_labels is not None:
        for source_ax, row_label in zip(source_axes, row_labels):
            source_ax.text(
                row_label_x,
                0.5,
                row_label,
                transform=source_ax.transAxes,
                ha="right",
                va="center",
                clip_on=False,
                fontsize=row_label_fontsize,
            )

    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        top=top,
        bottom=bottom,
        wspace=0.0,
        hspace=grid_hspace,
    )

    if show_source_divider:
        divider_x = (source_axes[0].get_position().x1 + grid_axes[0][0].get_position().x0) * 0.5
        divider_y0 = min(source_axes[-1].get_position().y0, grid_axes[-1][0].get_position().y0)
        divider_y1 = max(source_axes[0].get_position().y1, grid_axes[0][0].get_position().y1)
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
        fig.text(
            0.5,
            title_y,
            title,
            ha="center",
            va="top",
            fontsize=title_fontsize,
        )

    if xlabel:
        fig.text(
            0.5,
            xlabel_y,
            xlabel,
            ha="center",
            va="top",
            fontsize=xlabel_fontsize,
        )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=save_pad_inches)
    else:
        fig.show()

    plt.close(fig)


def plot_noises(
    sim_dir: str | Path,
    out_dir: str | Path,
    num_samples: int,
    noise_strengths: list[float],
    seed: int,
    samples_as_rows: bool = False,
    combined_filename: str = "noise_grid_rows.pdf",
    show_row_labels: bool = False,
    image_size: int = 256,
    rgb_parent_dirs: str | list[str] | tuple[str, ...] | set[str] | None = ("opt",),
    schedule: str = "linear",
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    s: float = 0.008,
    ddim_steps: int = 50,
    title: str | None = "Forward Noising Across Strengths",
    source_label: str = "Source (Sim)",
    source_bottom_label: str = "0",
    xlabel: str = "Noise Strength",
    dpi: int = 300,
    scale: float = 1.0,
    source_width_ratio: float = 1.0,  # Relative width of the source-image column.
    source_gap_ratio: float = 0.25,  # Gap between the source image and the noised-image grid, in panel-width units.
    grid_wspace: float = 0.06,  # Horizontal spacing between image panels.
    grid_hspace: float = 0.06,  # Vertical spacing between image panels.
    show_source_divider: bool = False,  # Whether to draw a divider centered in the source-to-grid gap.
    source_divider_color: str = "black",  # Color of the divider between source and noised images.
    source_divider_linewidth: float = 0.8,  # Line width of the divider between source and noised images.
    source_label_fontsize: float = 8,  # Font size for the source-image title.
    source_label_pad: float = 4,  # Gap between the source image and its title.
    col_label_fontsize: float | None = None,  # Font size for per-column labels; None uses Matplotlib defaults.
    col_label_y: float = -0.08,  # Vertical offset for per-column labels in axes coordinates.
    title_fontsize: float = 9,  # Font size for the main title above the noised-image grid.
    title_y: float = 0.985,  # Vertical position of the main title in figure coordinates.
    xlabel_fontsize: float | None = None,  # Font size for the bottom x-label; None uses Matplotlib defaults.
    xlabel_y: float = 0.03,  # Vertical position of the bottom x-label in figure coordinates.
    subplot_left: float = 0.03,  # Left outer margin for the full figure.
    subplot_right: float = 0.995,  # Right outer margin for the full figure.
    subplot_top_with_title: float = 0.93,  # Top margin when a title is present.
    subplot_top_without_title: float = 0.98,  # Top margin when no title is present.
    subplot_bottom_with_xlabel: float = 0.12,  # Bottom margin when bottom labels/x-label are present.
    subplot_bottom_without_xlabel: float = 0.06,  # Bottom margin when no x-label is present.
    save_pad_inches: float = 0.03,  # Extra whitespace around the saved figure.
):
    set_seed(seed)

    sim_dir = Path(sim_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TranslateDataset(
        src_dir=str(sim_dir),
        rgb_parent_dirs=as_parent_dir_set(rgb_parent_dirs),
        image_size=image_size,
    )
    if len(dataset) == 0:
        raise ValueError(f"No source images found under {sim_dir}.")

    sample_indices = select_sample_indices(len(dataset), num_samples, seed)
    if not sample_indices:
        raise ValueError("No source images selected for noising plot.")

    device = torch.device("cpu")
    sched = DiffusionSchedule(
        schedule=schedule,
        T=T,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
        s=s,
    )

    col_labels = [f"{strength:g}" for strength in noise_strengths]
    source_images = []
    noised_image_rows = []
    row_labels = []

    for sample_idx, dataset_idx in enumerate(sample_indices):
        x_src, filepath = dataset[dataset_idx]
        x_src = x_src.unsqueeze(0).to(device)

        # Use one fixed epsilon per source image so changes across columns isolate strength.
        sample_gen = torch.Generator(device=device)
        sample_gen.manual_seed(seed + sample_idx)
        eps = torch.randn(x_src.shape, generator=sample_gen, device=device, dtype=x_src.dtype)

        noised_images = []
        for strength in noise_strengths:
            x_t = q_sample_from_strength(
                x_src=x_src,
                eps=eps,
                strength=strength,
                sched=sched,
                ddim_steps=ddim_steps,
            )
            noised_images.append(tensor_to_numpy_image(x_t[0]))

        sample_name = f"{sample_idx:03d}_{Path(filepath).stem}"
        source_image = tensor_to_numpy_image(x_src[0])
        if samples_as_rows:
            source_images.append(source_image)
            noised_image_rows.append(noised_images)
            row_labels.append(sample_name)
            continue

        plot_noise_grid(
            source_image=source_image,
            noised_images=noised_images,
            col_labels=col_labels,
            title=title,
            source_label=source_label,
            source_bottom_label=source_bottom_label,
            xlabel=xlabel,
            save_path=out_dir / f"{sample_name}.pdf",
            dpi=dpi,
            scale=scale,
            source_width_ratio=source_width_ratio,
            source_gap_ratio=source_gap_ratio,
            grid_wspace=grid_wspace,
            grid_hspace=grid_hspace,
            show_source_divider=show_source_divider,
            source_divider_color=source_divider_color,
            source_divider_linewidth=source_divider_linewidth,
            source_label_fontsize=source_label_fontsize,
            source_label_pad=source_label_pad,
            col_label_fontsize=col_label_fontsize,
            col_label_y=col_label_y,
            title_fontsize=title_fontsize,
            title_y=title_y,
            xlabel_fontsize=xlabel_fontsize,
            xlabel_y=xlabel_y,
            subplot_left=subplot_left,
            subplot_right=subplot_right,
            subplot_top_with_title=subplot_top_with_title,
            subplot_top_without_title=subplot_top_without_title,
            subplot_bottom_with_xlabel=subplot_bottom_with_xlabel,
            subplot_bottom_without_xlabel=subplot_bottom_without_xlabel,
            save_pad_inches=save_pad_inches,
        )

    if samples_as_rows:
        plot_noise_grid_rows(
            source_images=source_images,
            noised_image_rows=noised_image_rows,
            col_labels=col_labels,
            title=title,
            source_label=source_label,
            source_bottom_label=source_bottom_label,
            xlabel=xlabel,
            row_labels=row_labels if show_row_labels else None,
            save_path=out_dir / combined_filename,
            dpi=dpi,
            scale=scale,
            source_width_ratio=source_width_ratio,
            source_gap_ratio=source_gap_ratio,
            grid_wspace=grid_wspace,
            grid_hspace=grid_hspace,
            show_source_divider=show_source_divider,
            source_divider_color=source_divider_color,
            source_divider_linewidth=source_divider_linewidth,
            source_label_fontsize=source_label_fontsize,
            source_label_pad=source_label_pad,
            col_label_fontsize=col_label_fontsize,
            col_label_y=col_label_y,
            title_fontsize=title_fontsize,
            title_y=title_y,
            xlabel_fontsize=xlabel_fontsize,
            xlabel_y=xlabel_y,
            subplot_left=subplot_left,
            subplot_right=subplot_right,
            subplot_top_with_title=subplot_top_with_title,
            subplot_top_without_title=subplot_top_without_title,
            subplot_bottom_with_xlabel=subplot_bottom_with_xlabel,
            subplot_bottom_without_xlabel=subplot_bottom_without_xlabel,
            save_pad_inches=save_pad_inches,
        )


def main():
    sim_dir = Path("/develop/data/remote_sensing/tiled/sim_subset/sim_test")
    out_dir = Path("/develop/code/eval/cyclenet/remote_sensing/noise_grids")
    num_samples = 3
    noise_strengths = [0.1, 0.2, 0.3, 0.4, 0.5]
    seed = 1
    samples_as_rows = True  # When True, write one multi-row grid instead of one PDF per sample.
    combined_filename = "noise_grid_rows.pdf"  # Output filename used when samples_as_rows=True.
    show_row_labels = False  # Whether to annotate each row with the sample name in multi-row mode.

    image_size = 256
    rgb_parent_dirs = ("opt",)
    schedule = "linear"
    T = 1000
    beta_start = 1e-4
    beta_end = 2e-2
    s = 0.008
    ddim_steps = 50

    # -------------------------
    # Adjust plotting
    # -------------------------
    title = "Forward Noising Across Strengths"
    source_label = "Source (Sim)"
    source_bottom_label = "0"
    xlabel = f"Noise Strength ($\\gamma$)"
    
    dpi = 300
    scale = 1.0  # Overall scale multiplier for the full figure size.

    source_width_ratio = 1.0  # Relative width of the source-image column.
    source_gap_ratio = 0.25  # Gap between the source image and the noised-image grid, in panel-width units.

    grid_wspace = 0.06  # Horizontal spacing between image panels.
    grid_hspace = 0.06  # Vertical spacing between image panels.

    show_source_divider = True  # Whether to draw a divider centered in the source-to-grid gap.
    source_divider_color = "black"  # Color of the divider between source and noised images.
    source_divider_linewidth = 0.5  # Line width of the divider between source and noised images.

    source_label_fontsize = 8  # Font size for the source-image title.
    source_label_pad = 4  # Gap between the source image and its title.

    col_label_fontsize = 8  # Font size for per-column labels; None uses Matplotlib defaults.
    col_label_y = -0.08  # Vertical offset for per-column labels in axes coordinates.

    title_fontsize = 9  # Font size for the main title above the noised-image grid.
    title_y = 1  # Vertical position of the main title in figure coordinates.

    xlabel_fontsize = 9  # Font size for the bottom x-label; None uses Matplotlib defaults.
    xlabel_y = 0.055  # Vertical position of the bottom x-label in figure coordinates.

    subplot_left = 0.03  # Left outer margin for the full figure.
    subplot_right = 0.995  # Right outer margin for the full figure.
    subplot_top_with_title = 0.93  # Top margin when a title is present.
    subplot_top_without_title = 0.98  # Top margin when no title is present.
    subplot_bottom_with_xlabel = 0.12  # Bottom margin when bottom labels/x-label are present.
    subplot_bottom_without_xlabel = 0.06  # Bottom margin when no x-label is present.

    save_pad_inches = 0.03  # Extra whitespace around the saved figure.

    plot_noises(
        sim_dir=sim_dir,
        out_dir=out_dir,
        num_samples=num_samples,
        noise_strengths=noise_strengths,
        seed=seed,
        samples_as_rows=samples_as_rows,
        combined_filename=combined_filename,
        show_row_labels=show_row_labels,
        image_size=image_size,
        rgb_parent_dirs=rgb_parent_dirs,
        schedule=schedule,
        T=T,
        beta_start=beta_start,
        beta_end=beta_end,
        s=s,
        ddim_steps=ddim_steps,
        title=title,
        source_label=source_label,
        source_bottom_label=source_bottom_label,
        xlabel=xlabel,
        dpi=dpi,
        scale=scale,
        source_width_ratio=source_width_ratio,
        source_gap_ratio=source_gap_ratio,
        grid_wspace=grid_wspace,
        grid_hspace=grid_hspace,
        show_source_divider=show_source_divider,
        source_divider_color=source_divider_color,
        source_divider_linewidth=source_divider_linewidth,
        source_label_fontsize=source_label_fontsize,
        source_label_pad=source_label_pad,
        col_label_fontsize=col_label_fontsize,
        col_label_y=col_label_y,
        title_fontsize=title_fontsize,
        title_y=title_y,
        xlabel_fontsize=xlabel_fontsize,
        xlabel_y=xlabel_y,
        subplot_left=subplot_left,
        subplot_right=subplot_right,
        subplot_top_with_title=subplot_top_with_title,
        subplot_top_without_title=subplot_top_without_title,
        subplot_bottom_with_xlabel=subplot_bottom_with_xlabel,
        subplot_bottom_without_xlabel=subplot_bottom_without_xlabel,
        save_pad_inches=save_pad_inches,
    )


if __name__ == "__main__":
    main()
