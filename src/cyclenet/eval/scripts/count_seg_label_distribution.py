from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cyclenet.data.dataset import load_label_mask
from cyclenet.eval.plotting.set_style import COLORS, apply_style


IMAGE_EXTS = {".tif", ".tiff"}
CLASS_NAMES = {
    0: "Ignore",
    1: "Bareland",
    2: "Rangeland",
    3: "Developed Space",
    4: "Road",
    5: "Trees",
    6: "Water",
    7: "Agriculture land",
    8: "Buildings",
}

apply_style()


def collect_mask_paths(root: Path, label_parent_dir: str, file_exts: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.parent.name == label_parent_dir
        and path.suffix.lower() in file_exts
    )


def count_labels(mask_paths: list[Path]) -> Counter[int]:
    counts: Counter[int] = Counter()

    for mask_path in mask_paths:
        mask = np.asarray(load_label_mask(mask_path), dtype=np.int64)
        values, value_counts = np.unique(mask, return_counts=True)
        for value, value_count in zip(values.tolist(), value_counts.tolist(), strict=True):
            counts[int(value)] += int(value_count)

    return counts


def summarize_domain(root: Path, domain_name: str, label_parent_dir: str) -> dict[str, object]:
    mask_paths = collect_mask_paths(root, label_parent_dir=label_parent_dir, file_exts=IMAGE_EXTS)
    if not mask_paths:
        raise FileNotFoundError(
            f"No mask files found under parent dir '{label_parent_dir}' in {root}"
        )

    counts = count_labels(mask_paths)
    total_pixels = sum(counts.values())
    non_ignore_pixels = total_pixels - counts.get(0, 0)

    return {
        "domain_name": domain_name,
        "root": root,
        "mask_paths": mask_paths,
        "counts": counts,
        "total_pixels": total_pixels,
        "non_ignore_pixels": non_ignore_pixels,
    }


def format_label_name(label_id: int) -> str:
    return CLASS_NAMES.get(label_id, f"Unknown({label_id})")


def summary_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    domain_name = str(summary["domain_name"])
    counts: Counter[int] = summary["counts"]  # type: ignore[assignment]
    total_pixels = int(summary["total_pixels"])
    non_ignore_pixels = int(summary["non_ignore_pixels"])

    rows: list[dict[str, object]] = []
    for label_id in sorted(counts):
        count = counts[label_id]
        frac_all = count / total_pixels if total_pixels else 0.0
        frac_non_ignore = 0.0
        if label_id != 0 and non_ignore_pixels:
            frac_non_ignore = count / non_ignore_pixels

        rows.append(
            {
                "domain": domain_name,
                "label_id": label_id,
                "label_name": format_label_name(label_id),
                "pixel_count": count,
                "frac_all": frac_all,
                "frac_non_ignore": frac_non_ignore,
                "num_masks": int(len(summary["mask_paths"])),
                "total_pixels": total_pixels,
                "non_ignore_pixels": non_ignore_pixels,
            }
        )

    return rows


def comparison_rows(
    sim_summary: dict[str, object],
    real_summary: dict[str, object],
) -> list[dict[str, object]]:
    sim_counts: Counter[int] = sim_summary["counts"]  # type: ignore[assignment]
    real_counts: Counter[int] = real_summary["counts"]  # type: ignore[assignment]
    sim_non_ignore = int(sim_summary["non_ignore_pixels"])
    real_non_ignore = int(real_summary["non_ignore_pixels"])

    rows: list[dict[str, object]] = []
    for label_id in sorted(set(sim_counts) | set(real_counts)):
        if label_id == 0:
            continue

        sim_count = sim_counts.get(label_id, 0)
        real_count = real_counts.get(label_id, 0)
        sim_frac = sim_count / sim_non_ignore if sim_non_ignore else 0.0
        real_frac = real_count / real_non_ignore if real_non_ignore else 0.0

        rows.append(
            {
                "label_id": label_id,
                "label_name": format_label_name(label_id),
                "sim_pixel_count": sim_count,
                "real_pixel_count": real_count,
                "sim_frac_non_ignore": sim_frac,
                "real_frac_non_ignore": real_frac,
                "real_minus_sim": real_frac - sim_frac,
            }
        )

    return rows


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_label_count_bars(
    sim_summary: dict[str, object],
    real_summary: dict[str, object],
    out_path: Path,
) -> None:
    sim_counts: Counter[int] = sim_summary["counts"]  # type: ignore[assignment]
    real_counts: Counter[int] = real_summary["counts"]  # type: ignore[assignment]
    label_ids = sorted(label_id for label_id in set(sim_counts) | set(real_counts) if label_id != 0)
    labels = [format_label_name(label_id) for label_id in label_ids]

    sim_values = np.array([sim_counts.get(label_id, 0) for label_id in label_ids], dtype=np.float64)
    real_values = np.array([real_counts.get(label_id, 0) for label_id in label_ids], dtype=np.float64)

    x = np.arange(len(label_ids), dtype=np.float64)
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    ax.bar(x - width / 2, sim_values, width=width, color=COLORS["sim"], label="Sim")
    ax.bar(x + width / 2, real_values, width=width, color=COLORS["real"], label="Real")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Pixel count")
    ax.set_title("Segmentation Label Pixel Counts")
    ax.set_yscale("log")
    ax.grid(axis="y")
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_label_fraction_bars(
    sim_summary: dict[str, object],
    real_summary: dict[str, object],
    out_path: Path,
) -> None:
    sim_counts: Counter[int] = sim_summary["counts"]  # type: ignore[assignment]
    real_counts: Counter[int] = real_summary["counts"]  # type: ignore[assignment]
    sim_non_ignore = int(sim_summary["non_ignore_pixels"])
    real_non_ignore = int(real_summary["non_ignore_pixels"])

    label_ids = sorted(label_id for label_id in set(sim_counts) | set(real_counts) if label_id != 0)
    labels = [format_label_name(label_id) for label_id in label_ids]

    sim_values = np.array(
        [sim_counts.get(label_id, 0) / sim_non_ignore if sim_non_ignore else 0.0 for label_id in label_ids],
        dtype=np.float64,
    )
    real_values = np.array(
        [real_counts.get(label_id, 0) / real_non_ignore if real_non_ignore else 0.0 for label_id in label_ids],
        dtype=np.float64,
    )

    x = np.arange(len(label_ids), dtype=np.float64)
    width = 0.38

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    ax.bar(x - width / 2, sim_values, width=width, color=COLORS["sim"], label="Sim")
    ax.bar(x + width / 2, real_values, width=width, color=COLORS["real"], label="Real")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Fraction of non-ignore pixels")
    ax.set_title("Segmentation Label Distribution")
    ax.grid(axis="y")
    ax.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def print_summary(summary: dict[str, object]) -> None:
    domain_name = str(summary["domain_name"])
    root = Path(summary["root"])
    mask_paths = list(summary["mask_paths"])
    counts: Counter[int] = summary["counts"]  # type: ignore[assignment]
    total_pixels = int(summary["total_pixels"])
    non_ignore_pixels = int(summary["non_ignore_pixels"])

    labels = sorted(counts)

    print(f"\n[{domain_name}]")
    print(f"root: {root}")
    print(f"masks: {len(mask_paths)}")
    print(f"total pixels: {total_pixels}")
    print(f"non-ignore pixels: {non_ignore_pixels}")
    print("label_id\tlabel_name\tpixel_count\tfrac_all\tfrac_non_ignore")

    for label_id in labels:
        count = counts[label_id]
        frac_all = count / total_pixels if total_pixels else 0.0
        if label_id == 0:
            frac_non_ignore = 0.0
        else:
            frac_non_ignore = count / non_ignore_pixels if non_ignore_pixels else 0.0

        print(
            f"{label_id}\t"
            f"{format_label_name(label_id)}\t"
            f"{count}\t"
            f"{frac_all:.6f}\t"
            f"{frac_non_ignore:.6f}"
        )


def print_comparison(sim_summary: dict[str, object], real_summary: dict[str, object]) -> None:
    print("\n[sim vs real comparison]")
    print("label_id\tlabel_name\tsim_frac_non_ignore\treal_frac_non_ignore\treal_minus_sim")

    for row in comparison_rows(sim_summary, real_summary):
        print(
            f"{row['label_id']}\t"
            f"{row['label_name']}\t"
            f"{row['sim_frac_non_ignore']:.6f}\t"
            f"{row['real_frac_non_ignore']:.6f}\t"
            f"{row['real_minus_sim']:.6f}"
        )


def main():
    # Root directory for the simulated dataset to scan for mask .tif files.
    sim_root = Path("/path/to/sim_root")
    # Root directory for the real dataset to scan for mask .tif files.
    real_root = Path("/path/to/real_root")
    # Name of the parent directory that contains segmentation masks.
    label_parent_dir = "gt_ss_mask"
    # Directory where CSV summaries and comparison plots will be written.
    out_dir = Path("/tmp/seg_label_distribution")

    if not sim_root.exists():
        raise FileNotFoundError(f"sim_root does not exist: {sim_root}")
    if not real_root.exists():
        raise FileNotFoundError(f"real_root does not exist: {real_root}")

    sim_summary = summarize_domain(
        root=sim_root,
        domain_name="sim",
        label_parent_dir=label_parent_dir,
    )
    real_summary = summarize_domain(
        root=real_root,
        domain_name="real",
        label_parent_dir=label_parent_dir,
    )

    print_summary(sim_summary)
    print_summary(real_summary)
    print_comparison(sim_summary, real_summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = summary_rows(sim_summary) + summary_rows(real_summary)
    comparison = comparison_rows(sim_summary, real_summary)

    write_csv(all_rows, out_dir / "label_distribution_summary.csv")
    write_csv(comparison, out_dir / "label_distribution_comparison.csv")
    plot_label_count_bars(sim_summary, real_summary, out_dir / "label_pixel_counts.png")
    plot_label_fraction_bars(sim_summary, real_summary, out_dir / "label_fraction_non_ignore.png")

    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
