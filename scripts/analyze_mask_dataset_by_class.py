#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from cyclenet.data.dataset import load_label_mask

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTS = {".tif", ".tiff"}
CLASS_INFO = {
    0: ("Ignore", "ignore"),
    1: ("Bareland", "bareland"),
    2: ("Rangeland", "rangeland"),
    3: ("Developed Space", "developed_space"),
    4: ("Road", "road"),
    5: ("Trees", "trees"),
    6: ("Water", "water"),
    7: ("Agriculture land", "agriculture_land"),
    8: ("Buildings", "buildings"),
}
DATASET_COLORS = {
    "sim": "#6b7280",
    "real": "#dc2626",
}
NON_IGNORE_LABEL_IDS = [label_id for label_id in sorted(CLASS_INFO) if label_id != 0]


def collect_mask_paths(root: Path, label_parent_dir: str, file_exts: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.parent.name == label_parent_dir
        and path.suffix.lower() in file_exts
    )


def label_name(label_id: int) -> str:
    return CLASS_INFO.get(label_id, (f"Unknown({label_id})", f"unknown_{label_id}"))[0]


def label_slug(label_id: int) -> str:
    return CLASS_INFO.get(label_id, (f"Unknown({label_id})", f"unknown_{label_id}"))[1]


def safe_stat(values: list[float], fn) -> float:
    if not values:
        return float("nan")
    return float(fn(np.asarray(values, dtype=np.float64)))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def boundary_pixel_count(binary_mask: np.ndarray) -> int:
    if not binary_mask.any():
        return 0

    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return int((boundary > 0).sum())


def component_rows_for_label(
    binary_mask: np.ndarray,
    dataset_name: str,
    mask_path: Path,
    mask_rel_path: str,
    label_id: int,
    min_component_area: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not binary_mask.any():
        return rows

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8),
        connectivity=8,
    )
    height, width = binary_mask.shape

    for component_index in range(1, num_labels):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        left = int(stats[component_index, cv2.CC_STAT_LEFT])
        top = int(stats[component_index, cv2.CC_STAT_TOP])
        bbox_width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        bbox_height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        touches_border = (
            left == 0
            or top == 0
            or left + bbox_width >= width
            or top + bbox_height >= height
        )

        rows.append(
            {
                "dataset": dataset_name,
                "mask_path": str(mask_path),
                "mask_rel_path": mask_rel_path,
                "label_id": label_id,
                "label_name": label_name(label_id),
                "component_index": component_index,
                "component_area": area,
                "bbox_left": left,
                "bbox_top": top,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_aspect_ratio": bbox_width / bbox_height if bbox_height > 0 else float("nan"),
                "equivalent_diameter": float(np.sqrt((4.0 * area) / np.pi)),
                "touches_border": int(touches_border),
            }
        )

    return rows


def analyze_mask(
    mask_path: Path,
    dataset_name: str,
    dataset_root: Path,
    min_component_area: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    mask = np.asarray(load_label_mask(mask_path), dtype=np.int32)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask for {mask_path}, got shape {mask.shape}")

    image_height, image_width = mask.shape
    total_pixels = int(mask.size)
    ignore_pixels = int((mask == 0).sum())
    non_ignore_pixels = total_pixels - ignore_pixels
    mask_rel_path = str(mask_path.relative_to(dataset_root))

    valid_binary = (mask != 0).astype(np.uint8)
    total_boundary_pixels = boundary_pixel_count(valid_binary)
    per_image_row: dict[str, object] = {
        "dataset": dataset_name,
        "mask_path": str(mask_path),
        "mask_rel_path": mask_rel_path,
        "image_height": image_height,
        "image_width": image_width,
        "total_pixels": total_pixels,
        "ignore_pixels": ignore_pixels,
        "ignore_fraction": ignore_pixels / total_pixels if total_pixels else 0.0,
        "non_ignore_pixels": non_ignore_pixels,
        "total_boundary_pixels": total_boundary_pixels,
        "total_boundary_density": total_boundary_pixels / total_pixels if total_pixels else 0.0,
    }

    component_rows: list[dict[str, object]] = []
    for label_id in NON_IGNORE_LABEL_IDS:
        slug = label_slug(label_id)
        binary_mask = (mask == label_id).astype(np.uint8)
        pixel_count = int(binary_mask.sum())
        area_fraction = pixel_count / total_pixels if total_pixels else 0.0
        boundary_pixels = boundary_pixel_count(binary_mask)

        label_component_rows = component_rows_for_label(
            binary_mask=binary_mask,
            dataset_name=dataset_name,
            mask_path=mask_path,
            mask_rel_path=mask_rel_path,
            label_id=label_id,
            min_component_area=min_component_area,
        )
        component_rows.extend(label_component_rows)

        per_image_row[f"pixel_count_{slug}"] = pixel_count
        per_image_row[f"area_fraction_{slug}"] = area_fraction
        per_image_row[f"component_count_{slug}"] = len(label_component_rows)
        per_image_row[f"boundary_pixels_{slug}"] = boundary_pixels
        per_image_row[f"boundary_density_{slug}"] = boundary_pixels / total_pixels if total_pixels else 0.0

    return per_image_row, component_rows


def analyze_dataset(
    root: Path,
    dataset_name: str,
    label_parent_dir: str,
    min_component_area: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mask_paths = collect_mask_paths(root, label_parent_dir=label_parent_dir, file_exts=IMAGE_EXTS)
    if not mask_paths:
        raise FileNotFoundError(f"No mask files found under '{label_parent_dir}' in {root}")

    print(f"[{dataset_name}] analyzing {len(mask_paths)} masks from {root}")

    per_image_rows: list[dict[str, object]] = []
    per_component_rows: list[dict[str, object]] = []

    for mask_index, mask_path in enumerate(mask_paths, start=1):
        if mask_index % 100 == 0 or mask_index == len(mask_paths):
            print(f"[{dataset_name}] {mask_index}/{len(mask_paths)}")

        per_image_row, component_rows = analyze_mask(
            mask_path=mask_path,
            dataset_name=dataset_name,
            dataset_root=root,
            min_component_area=min_component_area,
        )
        per_image_rows.append(per_image_row)
        per_component_rows.extend(component_rows)

    return per_image_rows, per_component_rows


def build_summary_rows(
    per_image_rows: list[dict[str, object]],
    per_component_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    dataset_names = sorted({str(row["dataset"]) for row in per_image_rows})
    summary_rows: list[dict[str, object]] = []

    for dataset_name in dataset_names:
        dataset_image_rows = [row for row in per_image_rows if row["dataset"] == dataset_name]
        dataset_component_rows = [row for row in per_component_rows if row["dataset"] == dataset_name]
        num_images = len(dataset_image_rows)

        for label_id in NON_IGNORE_LABEL_IDS:
            slug = label_slug(label_id)
            pixel_fractions = [float(row[f"area_fraction_{slug}"]) for row in dataset_image_rows]
            component_counts = [float(row[f"component_count_{slug}"]) for row in dataset_image_rows]
            boundary_densities = [float(row[f"boundary_density_{slug}"]) for row in dataset_image_rows]
            label_components = [row for row in dataset_component_rows if row["label_id"] == label_id]

            component_areas = [float(row["component_area"]) for row in label_components]
            bbox_widths = [float(row["bbox_width"]) for row in label_components]
            bbox_heights = [float(row["bbox_height"]) for row in label_components]
            equivalent_diameters = [float(row["equivalent_diameter"]) for row in label_components]
            border_touching = [float(row["touches_border"]) for row in label_components]

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "label_id": label_id,
                    "label_name": label_name(label_id),
                    "num_images": num_images,
                    "images_with_class": int(sum(value > 0 for value in pixel_fractions)),
                    "images_with_class_fraction": (
                        sum(value > 0 for value in pixel_fractions) / num_images if num_images else float("nan")
                    ),
                    "pixel_fraction_mean": safe_stat(pixel_fractions, np.mean),
                    "pixel_fraction_median": safe_stat(pixel_fractions, np.median),
                    "pixel_fraction_p90": quantile(pixel_fractions, 0.90),
                    "component_count_mean": safe_stat(component_counts, np.mean),
                    "component_count_median": safe_stat(component_counts, np.median),
                    "component_count_p90": quantile(component_counts, 0.90),
                    "boundary_density_mean": safe_stat(boundary_densities, np.mean),
                    "boundary_density_median": safe_stat(boundary_densities, np.median),
                    "boundary_density_p90": quantile(boundary_densities, 0.90),
                    "num_components": len(label_components),
                    "component_area_mean": safe_stat(component_areas, np.mean),
                    "component_area_median": safe_stat(component_areas, np.median),
                    "component_area_p90": quantile(component_areas, 0.90),
                    "bbox_width_mean": safe_stat(bbox_widths, np.mean),
                    "bbox_width_median": safe_stat(bbox_widths, np.median),
                    "bbox_width_p90": quantile(bbox_widths, 0.90),
                    "bbox_height_mean": safe_stat(bbox_heights, np.mean),
                    "bbox_height_median": safe_stat(bbox_heights, np.median),
                    "bbox_height_p90": quantile(bbox_heights, 0.90),
                    "equivalent_diameter_mean": safe_stat(equivalent_diameters, np.mean),
                    "equivalent_diameter_median": safe_stat(equivalent_diameters, np.median),
                    "equivalent_diameter_p90": quantile(equivalent_diameters, 0.90),
                    "touches_border_fraction": safe_stat(border_touching, np.mean),
                }
            )

    return summary_rows


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary_metric(
    summary_rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    use_log_scale: bool = False,
) -> None:
    datasets = sorted({str(row["dataset"]) for row in summary_rows})
    label_ids = NON_IGNORE_LABEL_IDS
    labels = [label_name(label_id) for label_id in label_ids]
    x = np.arange(len(label_ids), dtype=np.float64)
    width = 0.36 if len(datasets) <= 2 else 0.8 / max(len(datasets), 1)

    fig, ax = plt.subplots(figsize=(12.0, 5.5))

    for dataset_index, dataset_name in enumerate(datasets):
        dataset_rows = {
            int(row["label_id"]): row
            for row in summary_rows
            if row["dataset"] == dataset_name
        }
        values = np.array(
            [float(dataset_rows.get(label_id, {}).get(metric, np.nan)) for label_id in label_ids],
            dtype=np.float64,
        )
        if use_log_scale:
            values[~np.isfinite(values) | (values <= 0)] = np.nan
        offsets = x + (dataset_index - (len(datasets) - 1) / 2.0) * width
        ax.bar(
            offsets,
            values,
            width=width,
            label=dataset_name.capitalize(),
            color=DATASET_COLORS.get(dataset_name, None),
        )

    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if use_log_scale:
        ax.set_yscale("log")
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_console_summary(summary_rows: list[dict[str, object]]) -> None:
    print("\n[dataset mask summary]")
    print(
        "dataset\tlabel\tpixel_frac_mean\tcomponent_count_mean\t"
        "component_area_median\tbbox_width_median\tbbox_height_median"
    )
    for row in summary_rows:
        print(
            f"{row['dataset']}\t"
            f"{row['label_name']}\t"
            f"{float(row['pixel_fraction_mean']):.6f}\t"
            f"{float(row['component_count_mean']):.3f}\t"
            f"{float(row['component_area_median']):.3f}\t"
            f"{float(row['bbox_width_median']):.3f}\t"
            f"{float(row['bbox_height_median']):.3f}"
        )


def main() -> None:
    # Root directory for the simulated dataset that contains mask files under `gt_ss_mask`.
    sim_root = Path("/path/to/sim_root")
    # Root directory for the real dataset that contains mask files under `gt_ss_mask`.
    real_root = Path("/path/to/real_root")
    # Name of the parent directory that contains segmentation masks.
    label_parent_dir = "gt_ss_mask"
    # Minimum connected-component area in pixels to keep in per-object statistics.
    min_component_area = 8
    # Output directory for CSV summaries and plots.
    out_dir = Path("/tmp/mask_dataset_analysis")

    if not sim_root.exists():
        raise FileNotFoundError(f"sim_root does not exist: {sim_root}")
    if not real_root.exists():
        raise FileNotFoundError(f"real_root does not exist: {real_root}")

    sim_per_image_rows, sim_per_component_rows = analyze_dataset(
        root=sim_root,
        dataset_name="sim",
        label_parent_dir=label_parent_dir,
        min_component_area=min_component_area,
    )
    real_per_image_rows, real_per_component_rows = analyze_dataset(
        root=real_root,
        dataset_name="real",
        label_parent_dir=label_parent_dir,
        min_component_area=min_component_area,
    )

    per_image_rows = sim_per_image_rows + real_per_image_rows
    per_component_rows = sim_per_component_rows + real_per_component_rows
    summary_rows = build_summary_rows(per_image_rows, per_component_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(per_image_rows, out_dir / "per_image_metrics.csv")
    write_csv(per_component_rows, out_dir / "per_component_metrics.csv")
    write_csv(summary_rows, out_dir / "dataset_summary.csv")

    plots_dir = out_dir / "plots"
    plot_summary_metric(
        summary_rows=summary_rows,
        metric="pixel_fraction_mean",
        ylabel="Mean class area fraction",
        title="Per-class Area Fraction",
        out_path=plots_dir / "class_area_fraction_mean.png",
    )
    plot_summary_metric(
        summary_rows=summary_rows,
        metric="component_count_mean",
        ylabel="Mean components per image",
        title="Per-class Connected Components Per Image",
        out_path=plots_dir / "component_count_mean.png",
        use_log_scale=True,
    )
    plot_summary_metric(
        summary_rows=summary_rows,
        metric="component_area_median",
        ylabel="Median component area (pixels)",
        title="Per-class Median Component Area",
        out_path=plots_dir / "component_area_median.png",
        use_log_scale=True,
    )
    plot_summary_metric(
        summary_rows=summary_rows,
        metric="bbox_width_median",
        ylabel="Median component bbox width (pixels)",
        title="Per-class Median Component Width",
        out_path=plots_dir / "bbox_width_median.png",
        use_log_scale=True,
    )
    plot_summary_metric(
        summary_rows=summary_rows,
        metric="bbox_height_median",
        ylabel="Median component bbox height (pixels)",
        title="Per-class Median Component Height",
        out_path=plots_dir / "bbox_height_median.png",
        use_log_scale=True,
    )
    plot_summary_metric(
        summary_rows=summary_rows,
        metric="boundary_density_mean",
        ylabel="Mean boundary density",
        title="Per-class Boundary Density",
        out_path=plots_dir / "boundary_density_mean.png",
    )

    print_console_summary(summary_rows)
    print(f"\nSaved mask analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()
