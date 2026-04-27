from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .set_style import apply_style

apply_style()


TITLE_FONTSIZE = 18
COLUMN_LABEL_FONTSIZE = 16
MODEL_LABEL_FONTSIZE = 13
ROW_LABEL_FONTSIZE = 16

PANEL_SIZE_INCHES = 2.25
PAIR_GAP_INCHES = 0.08
GROUP_GAP_INCHES = 0.34
ROW_GAP_INCHES = 0.18
ROW_LABEL_GAP_INCHES = 0.46

LEFT_MARGIN_INCHES = 0.10
RIGHT_MARGIN_INCHES = 0.10
TOP_MARGIN_INCHES = 0.78
BOTTOM_MARGIN_INCHES = 0.86

IMAGE_BORDER_LINEWIDTH = 0.8


@dataclass(frozen=True)
class ModelComparisonSpec:
    label: str
    model_root: Path
    noise_strength: float
    cfg_weight: float


def find_tag_dir(model_root: Path, img_tag: str) -> Path:
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


def _compute_figure_size(
    n_rows: int,
    n_models: int,
    row_gap_inches: float,
) -> tuple[float, float]:
    grid_width = (
        2 * n_models * PANEL_SIZE_INCHES
        + n_models * PAIR_GAP_INCHES
        + max(n_models - 1, 0) * GROUP_GAP_INCHES
    )
    grid_height = (
        n_rows * PANEL_SIZE_INCHES
        + max(n_rows - 1, 0) * row_gap_inches
    )

    fig_width = LEFT_MARGIN_INCHES + grid_width + RIGHT_MARGIN_INCHES
    fig_height = TOP_MARGIN_INCHES + grid_height + BOTTOM_MARGIN_INCHES
    return fig_width, fig_height


def build_axes_grid(fig, n_rows: int, n_models: int, row_gap_inches: float) -> np.ndarray:
    axes = np.empty((n_rows, n_models, 2), dtype=object)

    fig_width, fig_height = fig.get_size_inches()
    available_width = fig_width - LEFT_MARGIN_INCHES - RIGHT_MARGIN_INCHES
    available_height = fig_height - TOP_MARGIN_INCHES - BOTTOM_MARGIN_INCHES

    total_horizontal_gap = (
        n_models * PAIR_GAP_INCHES
        + max(n_models - 1, 0) * GROUP_GAP_INCHES
    )
    axis_size = min(
        (available_width - total_horizontal_gap) / max(2 * n_models, 1),
        (available_height - row_gap_inches * max(n_rows - 1, 0)) / max(n_rows, 1),
    )

    grid_width = 2 * n_models * axis_size + total_horizontal_gap
    grid_height = n_rows * axis_size + max(n_rows - 1, 0) * row_gap_inches
    start_x = (LEFT_MARGIN_INCHES + 0.5 * (available_width - grid_width)) / fig_width
    start_y = (BOTTOM_MARGIN_INCHES + 0.5 * (available_height - grid_height)) / fig_height
    axis_width = axis_size / fig_width
    axis_height = axis_size / fig_height
    pair_gap = PAIR_GAP_INCHES / fig_width
    group_gap = GROUP_GAP_INCHES / fig_width
    row_gap = row_gap_inches / fig_height

    for row_idx in range(n_rows):
        y0 = start_y + (n_rows - 1 - row_idx) * (axis_height + row_gap)
        current_x0 = start_x
        for model_idx in range(n_models):
            for pair_idx in range(2):
                axes[row_idx, model_idx, pair_idx] = fig.add_axes(
                    [current_x0, y0, axis_width, axis_height]
                )
                current_x0 += axis_width
                if pair_idx == 0:
                    current_x0 += pair_gap
            if model_idx < n_models - 1:
                current_x0 += group_gap

    return axes


def _style_image_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(IMAGE_BORDER_LINEWIDTH)
        spine.set_edgecolor("black")


def _model_label_text(spec: ModelComparisonSpec, include_settings: bool) -> str:
    if not include_settings:
        return spec.label
    return (
        f"{spec.label}\n"
        f"$\\gamma={spec.noise_strength:g}$, $w={spec.cfg_weight:g}$"
    )


def plot_model_comparison_grid(
    model_specs: list[ModelComparisonSpec],
    img_tags: list[str],
    save_path: str | Path,
    row_labels: list[str] | None = None,
    title: str | None = None,
    include_model_settings: bool = True,
) -> Path:
    if not model_specs:
        raise ValueError("model_specs must contain at least one model.")
    if not img_tags:
        raise ValueError("img_tags must contain at least one image tag.")
    if row_labels is not None and len(row_labels) != len(img_tags):
        raise ValueError("row_labels must have the same length as img_tags.")

    row_gap_inches = ROW_LABEL_GAP_INCHES if row_labels is not None else ROW_GAP_INCHES
    fig_width, fig_height = _compute_figure_size(
        n_rows=len(img_tags),
        n_models=len(model_specs),
        row_gap_inches=row_gap_inches,
    )
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes = build_axes_grid(
        fig=fig,
        n_rows=len(img_tags),
        n_models=len(model_specs),
        row_gap_inches=row_gap_inches,
    )

    for model_idx, spec in enumerate(model_specs):
        axes[0, model_idx, 0].set_title("Input", fontsize=COLUMN_LABEL_FONTSIZE, pad=8)
        axes[0, model_idx, 1].set_title("Output", fontsize=COLUMN_LABEL_FONTSIZE, pad=8)

        for row_idx, img_tag in enumerate(img_tags):
            source_img = load_image(find_source_image_path(spec.model_root, img_tag))
            translated_img = load_image(
                find_translated_image_path(
                    model_root=spec.model_root,
                    img_tag=img_tag,
                    noise_strength=spec.noise_strength,
                    cfg_weight=spec.cfg_weight,
                )
            )

            source_ax = axes[row_idx, model_idx, 0]
            translated_ax = axes[row_idx, model_idx, 1]
            source_ax.imshow(source_img)
            translated_ax.imshow(translated_img)
            _style_image_axis(source_ax)
            _style_image_axis(translated_ax)

    if title is not None:
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, y=0.985)

    for model_idx, spec in enumerate(model_specs):
        left_bbox = axes[-1, model_idx, 0].get_position()
        right_bbox = axes[-1, model_idx, 1].get_position()
        x_center = 0.5 * (left_bbox.x0 + right_bbox.x1)
        y = min(left_bbox.y0, right_bbox.y0) - 0.035
        fig.text(
            x_center,
            y,
            _model_label_text(spec, include_settings=include_model_settings),
            ha="center",
            va="top",
            fontsize=MODEL_LABEL_FONTSIZE,
        )

    if row_labels is not None:
        left = axes[0, 0, 0].get_position().x0
        right = axes[0, -1, 1].get_position().x1
        x_center = 0.5 * (left + right)

        for row_idx, row_label in enumerate(row_labels):
            current_bottom = axes[row_idx, 0, 0].get_position().y0
            if row_idx < len(img_tags) - 1:
                next_top = axes[row_idx + 1, 0, 0].get_position().y1
                y = 0.5 * (current_bottom + next_top)
            else:
                y = current_bottom - 0.09
            fig.text(
                x_center,
                y,
                row_label,
                ha="center",
                va="center",
                fontsize=ROW_LABEL_FONTSIZE,
            )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def main() -> None:
    # Output figure path for the side-by-side comparison plot.
    save_path = Path("/develop/code/eval/cyclenet/remote_sensing/model_comparison/samples.pdf")

    # Model columns to compare. Each entry defines the display label, the grid
    # directory that contains `source_samples/` and `translated_samples/`, and
    # the single `(noise_strength, cfg_weight)` pair to show for that model.
    model_specs = [
        ModelComparisonSpec(
            label="Step 2.5k",
            model_root=Path("/develop/code/eval/cyclenet/remote_sensing/cfg_str_grid/oem_only/step-2500/ema"),
            noise_strength=0.30,
            cfg_weight=3.0,
        ),
        ModelComparisonSpec(
            label="Step 10k",
            model_root=Path("/develop/code/eval/cyclenet/remote_sensing/cfg_str_grid/all_real/step-10000/ema"),
            noise_strength=0.35,
            cfg_weight=3.0,
        ),
        ModelComparisonSpec(
            label="Step 30k",
            model_root=Path("/develop/code/eval/cyclenet/remote_sensing/cfg_str_grid/all_real/step-30000/ema"),
            noise_strength=0.30,
            cfg_weight=4.0,
        ),
    ]

    # Image tags to plot. Each tag is matched via `{img_tag}_*` under the
    # model's `source_samples/` and `translated_samples/` directories.
    img_tags = ["007", "013", "012"]

    # Optional centered labels placed beneath each image row.
    row_labels = None

    # Optional overall figure title.
    title = None

    # Whether to append the chosen `(noise_strength, cfg_weight)` values below
    # each model label.
    include_model_settings = True

    saved_path = plot_model_comparison_grid(
        model_specs=model_specs,
        img_tags=img_tags,
        save_path=save_path,
        row_labels=row_labels,
        title=title,
        include_model_settings=include_model_settings,
    )
    print(f"Saved model comparison plot to {saved_path}")


if __name__ == "__main__":
    main()
