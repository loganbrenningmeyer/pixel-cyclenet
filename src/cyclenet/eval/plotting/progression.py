from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .set_style import apply_style

apply_style()


TITLE_FONTSIZE = 18
COLUMN_LABEL_FONTSIZE = 14
GRID_LABEL_FONTSIZE = 15
GRID_GAP_INCHES = 0.12
SOURCE_GRID_GAP_INCHES = 0.28
GRID_LEFT = 0.02
GRID_RIGHT = 0.985
GRID_BOTTOM = 0.16
GRID_TOP = 0.89


def find_tag_dir(model_root: Path, img_tag: str) -> Path:
    """
    Resolve:
    model_root / "translated_samples" / "{img_tag}_*"
    """
    translated_dir = model_root / "translated_samples"
    tag_matches = sorted(translated_dir.glob(f"{img_tag}_*"))

    if not tag_matches:
        raise FileNotFoundError(
            f"No directory matching '{img_tag}_*' found under: {translated_dir}"
        )

    if len(tag_matches) > 1:
        print(
            f"Warning: multiple translated matches for tag '{img_tag}' under "
            f"{translated_dir}. Using: {tag_matches[0].name}"
        )

    return tag_matches[0]


def find_translated_image_path(
    model_root: Path,
    img_tag: str,
    noise_strength: float,
    cfg_weight: float,
) -> Path:
    """
    Resolve:
    model_root / "translated_samples" / "{img_tag}_*" /
    "strength-{noise_strength:.2f}" / "cfg-{cfg_weight:.1f}" / "img.png"
    """
    tag_dir = find_tag_dir(model_root, img_tag)

    img_path = (
        tag_dir
        / f"strength-{noise_strength:.2f}"
        / f"cfg-{cfg_weight:.1f}"
        / "img.png"
    )

    if not img_path.exists():
        raise FileNotFoundError(f"Missing translated image: {img_path}")

    return img_path


def find_source_image_path(model_root: Path, img_tag: str) -> Path:
    """
    Resolve the source image from:
    model_root / "source_samples"

    Uses prefix matching:
    {img_tag}_*.png

    This is more robust than assuming the full translated directory stem
    exactly matches the source filename stem.
    """
    source_dir = model_root / "source_samples"
    source_matches = sorted(source_dir.glob(f"{img_tag}_*.png"))

    if not source_matches:
        raise FileNotFoundError(
            f"No source image matching '{img_tag}_*.png' found under: {source_dir}"
        )

    if len(source_matches) > 1:
        print(
            f"Warning: multiple source matches for tag '{img_tag}' under "
            f"{source_dir}. Using: {source_matches[0].name}"
        )

    return source_matches[0]


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def format_step_label(step: int) -> str:
    if step >= 1000:
        value = step / 1000.0
        if value.is_integer():
            return f"{int(value)}k"
        return f"{value:g}k"
    return str(step)


def build_axes_grid(fig, n_rows: int, n_cols: int) -> np.ndarray:
    axes = np.empty((n_rows, n_cols), dtype=object)

    fig_width, fig_height = fig.get_size_inches()
    available_width = (GRID_RIGHT - GRID_LEFT) * fig_width
    available_height = (GRID_TOP - GRID_BOTTOM) * fig_height
    total_horizontal_gap = (
        SOURCE_GRID_GAP_INCHES + GRID_GAP_INCHES * max(n_cols - 2, 0)
    )

    axis_size = min(
        (available_width - total_horizontal_gap) / n_cols,
        (available_height - GRID_GAP_INCHES * (n_rows - 1)) / n_rows,
    )

    grid_width = n_cols * axis_size + total_horizontal_gap
    grid_height = n_rows * axis_size + (n_rows - 1) * GRID_GAP_INCHES
    start_x = GRID_LEFT + 0.5 * (available_width - grid_width) / fig_width
    start_y = GRID_BOTTOM + 0.5 * (available_height - grid_height) / fig_height
    axis_width = axis_size / fig_width
    axis_height = axis_size / fig_height
    gap_y = GRID_GAP_INCHES / fig_height

    for row_idx in range(n_rows):
        current_x0 = start_x
        for col_idx in range(n_cols):
            x0 = current_x0
            y0 = start_y + (n_rows - 1 - row_idx) * (axis_height + gap_y)
            axes[row_idx, col_idx] = fig.add_axes([x0, y0, axis_width, axis_height])
            if col_idx == 0:
                current_x0 += axis_width + SOURCE_GRID_GAP_INCHES / fig_width
            else:
                current_x0 += axis_width + GRID_GAP_INCHES / fig_width

    return axes


def main():
    cfg_weight = 3.0
    noise_strength = 0.30

    grid_dir = Path("/develop/code/eval/cyclenet/remote_sensing/cfg_str_grid")

    model_grid_dirs = {
        2500: grid_dir / "oem_only/step-2500/ema",
        10000: grid_dir / "all_real/step-10000/ema",
        20000: grid_dir / "all_real/step-20000/ema",
        30000: grid_dir / "all_real/step-30000/ema",
    }

    img_tags = ["007", "013", "012"]

    model_steps = list(model_grid_dirs.keys())
    n_rows = len(img_tags)
    n_model_cols = len(model_steps)
    n_cols = 1 + n_model_cols  # left source column + model columns

    fig = plt.figure(figsize=(2.8 * n_cols, 2.8 * n_rows + 0.6))
    axes = build_axes_grid(fig=fig, n_rows=n_rows, n_cols=n_cols)

    # Use first model root to resolve source images.
    # Assumes source_samples is shared / duplicated consistently alongside translated_samples.
    reference_model_root = next(iter(model_grid_dirs.values()))

    for row_idx, img_tag in enumerate(img_tags):
        # ---- source image column ----
        source_ax = axes[row_idx, 0]
        source_img_path = find_source_image_path(
            model_root=reference_model_root,
            img_tag=img_tag,
        )
        source_img = load_image(source_img_path)

        source_ax.imshow(source_img)
        source_ax.set_xticks([])
        source_ax.set_yticks([])
        if row_idx == n_rows - 1:
            source_ax.set_xlabel(
                "Source",
                fontsize=COLUMN_LABEL_FONTSIZE,
                labelpad=10,
            )

        # ---- translated image columns ----
        for model_col_idx, step in enumerate(model_steps, start=1):
            ax = axes[row_idx, model_col_idx]
            model_root = model_grid_dirs[step]

            img_path = find_translated_image_path(
                model_root=model_root,
                img_tag=img_tag,
                noise_strength=noise_strength,
                cfg_weight=cfg_weight,
            )
            img = load_image(img_path)

            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == n_rows - 1:
                ax.set_xlabel(
                    format_step_label(step),
                    fontsize=COLUMN_LABEL_FONTSIZE,
                    labelpad=10,
                )

    for ax_row in axes:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle(
        f"Source and translated samples at noise strength {noise_strength:.2f}, CFG {cfg_weight:.1f}",
        fontsize=TITLE_FONTSIZE,
        y=0.97,
    )

    source_bbox = axes[0, 0].get_position()
    translated_bbox = axes[0, 1].get_position()
    separator_x = 0.5 * (source_bbox.x1 + translated_bbox.x0)
    top = axes[0, 0].get_position().y1
    bottom = axes[-1, 0].get_position().y0
    fig.add_artist(
        mlines.Line2D(
            [separator_x, separator_x],
            [bottom, top],
            transform=fig.transFigure,
            color="black",
            linewidth=2.0,
        )
    )

    translated_left = axes[-1, 1].get_position().x0
    translated_right = axes[-1, -1].get_position().x1
    translated_bottom = axes[-1, 1].get_position().y0
    fig.text(
        0.5 * (translated_left + translated_right),
        translated_bottom - 0.07,
        "Model Checkpoint",
        ha="center",
        va="top",
        fontsize=GRID_LABEL_FONTSIZE,
    )

    fig.savefig("/develop/code/eval/cyclenet/remote_sensing/progression/samples.pdf")


if __name__ == "__main__":
    main()
