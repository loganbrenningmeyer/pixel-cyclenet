from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cyclenet.eval.plotting.set_style import (
    COLORS,
    MODEL_COLORS,
    MODEL_NAMES,
    apply_style,
)

apply_style()

REQUIRED_COLUMNS = {
    "model_name",
    "checkpoint",
    "noise_strength",
    "cfg_weight",
    "deeplab_translated_embed_path",
}


def _display_model_name(model_name: str) -> str:
    display_name = MODEL_NAMES.get(model_name, "")
    display_name = display_name if display_name else model_name.replace("_", " ")
    if display_name == "RGB + SPADE (BN Only)":
        return "RGB + SPADE\n(BN Only)"
    return display_name


def _model_color(model_name: str) -> str:
    color = MODEL_COLORS.get(model_name, "")
    return color if color else COLORS["translated"]


def _sample_coords(coords: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or len(coords) <= max_points:
        return coords

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords), size=max_points, replace=False)
    return coords[idx]


def _coerce_numeric_array(path: str | Path) -> np.ndarray:
    arr = np.load(Path(path))
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected a 2D embedding/coordinate array at {path}, got shape {arr.shape}")
    return np.asarray(arr, dtype=float)


def _load_selected_models(selected_models_csv: str | Path) -> pd.DataFrame:
    csv_path = Path(selected_models_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Selected models CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"Selected models CSV is missing required columns: {', '.join(missing)}"
        )

    if df.empty:
        raise ValueError(f"Selected models CSV is empty: {csv_path}")

    return df


def _load_reference_cache(reference_cache_dir: str | Path) -> tuple[object, np.ndarray, np.ndarray]:
    cache_dir = Path(reference_cache_dir)
    projector_path = cache_dir / "pca_projector.pkl"
    sim_coords_path = cache_dir / "pca_sim_coords.npy"
    real_coords_path = cache_dir / "pca_real_coords.npy"

    for path in (projector_path, sim_coords_path, real_coords_path):
        if not path.exists():
            raise FileNotFoundError(f"Required PCA reference-cache file does not exist: {path}")

    with projector_path.open("rb") as f:
        projector = pickle.load(f)

    sim_coords = _coerce_numeric_array(sim_coords_path)[:, :2]
    real_coords = _coerce_numeric_array(real_coords_path)[:, :2]
    return projector, sim_coords, real_coords


def _compute_axis_limits(
    coords_groups: list[np.ndarray],
    pad_frac: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    all_coords = np.concatenate(coords_groups, axis=0)
    x_min = float(all_coords[:, 0].min())
    x_max = float(all_coords[:, 0].max())
    y_min = float(all_coords[:, 1].min())
    y_max = float(all_coords[:, 1].max())

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_pad = x_span * pad_frac
    y_pad = y_span * pad_frac
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_deeplab_pca_selected_row(
    selected_models_csv: str | Path,
    reference_cache_dir: str | Path,
    save_path: str | Path,
    max_reference_points: int | None = 5000,
    max_translated_points: int | None = None,
    random_seed: int = 42,
    show_centroid: bool = True,
    model_title_fontsize: float = 16.0,
    axis_label_fontsize: float = 14.0,
    legend_fontsize: float = 14.0,
    rasterized: bool = False,
    reference_size: int = 16,
    translated_size: int = 20,
    reference_alpha: float = 0.3,
    translated_alpha: float = 0.45,
) -> Path:
    selected_df = _load_selected_models(selected_models_csv)
    projector, sim_coords, real_coords = _load_reference_cache(reference_cache_dir)

    translated_rows: list[dict[str, object]] = []
    for row_idx, row in selected_df.iterrows():
        embed_path = Path(str(row["deeplab_translated_embed_path"]))
        if not embed_path.exists():
            raise FileNotFoundError(
                f"Translated DeepLab embedding does not exist for model "
                f"'{row['model_name']}': {embed_path}"
            )

        translated_emb = _coerce_numeric_array(embed_path)
        translated_coords = np.asarray(projector.transform(translated_emb), dtype=float)
        if translated_coords.ndim != 2 or translated_coords.shape[1] < 2:
            raise ValueError(
                f"Expected projector.transform() to return 2D coords for {embed_path}, "
                f"got shape {translated_coords.shape}"
            )
        translated_rows.append(
            {
                "row": row,
                "coords": translated_coords[:, :2],
                "seed_offset": row_idx * 100,
            }
        )

    xlim, ylim = _compute_axis_limits(
        [sim_coords, real_coords] + [entry["coords"] for entry in translated_rows],
        pad_frac=0.05,
    )

    n_models = len(translated_rows)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(3.65 * n_models, 3.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_row = list(axes[0])

    for idx, (entry, ax) in enumerate(zip(translated_rows, axes_row, strict=True)):
        row = entry["row"]
        translated_coords = np.asarray(entry["coords"], dtype=float)
        step = int(row["checkpoint"])
        noise_strength = float(row["noise_strength"])
        cfg_weight = float(row["cfg_weight"])
        model_name = str(row["model_name"])
        model_color = _model_color(model_name)

        sim_points = _sample_coords(sim_coords, max_reference_points, random_seed + idx * 10)
        real_points = _sample_coords(real_coords, max_reference_points, random_seed + idx * 10 + 1)
        translated_points = _sample_coords(
            translated_coords,
            max_translated_points,
            random_seed + int(entry["seed_offset"]) + 2,
        )

        # ref_rasterized = len(sim_points) + len(real_points) > 4000
        # translated_rasterized = len(translated_points) > 4000

        ax.scatter(
            sim_points[:, 0],
            sim_points[:, 1],
            s=reference_size,
            alpha=reference_alpha,
            color=COLORS["sim"],
            edgecolors="none",
            rasterized=rasterized,
            zorder=1,
        )
        ax.scatter(
            real_points[:, 0],
            real_points[:, 1],
            s=reference_size,
            alpha=reference_alpha,
            color=COLORS["real"],
            edgecolors="none",
            rasterized=rasterized,
            zorder=1,
        )
        ax.scatter(
            translated_points[:, 0],
            translated_points[:, 1],
            s=translated_size,
            alpha=translated_alpha,
            color=model_color,
            marker="o",
            edgecolors="none",
            linewidths=0.8,
            rasterized=rasterized,
            zorder=2,
        )

        if show_centroid:
            centroid = translated_coords.mean(axis=0)
            ax.scatter(
                [centroid[0]],
                [centroid[1]],
                s=48,
                marker="o",
                color=model_color,
                facecolors=model_color,
                edgecolors="black",
                linewidths=1.0,
                zorder=3,
            )

        ax.set_title(_display_model_name(model_name), fontsize=model_title_fontsize)
        ax.grid(alpha=0.25)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1.0)
        if idx == 0:
            ax.set_ylabel("PCA 2", fontsize=axis_label_fontsize)
        ax.set_xlabel("PCA 1", fontsize=axis_label_fontsize)

    handles = [
        mlines.Line2D(
            [],
            [],
            color=COLORS["sim"],
            marker="o",
            linestyle="None",
            markersize=5.5,
            label="Sim embeddings",
        ),
        mlines.Line2D(
            [],
            [],
            color=COLORS["real"],
            marker="o",
            linestyle="None",
            markersize=5.5,
            label="Real embeddings",
        ),
        mlines.Line2D(
            [],
            [],
            color=COLORS["translated"],
            marker="o",
            markerfacecolor="white",
            markeredgecolor=COLORS["translated"],
            linestyle="None",
            markersize=5.5,
            label="Translated embeddings",
        ),
    ]
    if show_centroid:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=COLORS["translated"],
                marker="o",
                markerfacecolor="#dddddd",
                markeredgecolor="black",
                linestyle="None",
                markersize=6.5,
                label="Translated centroid",
            )
        )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 1.15),
        fontsize=legend_fontsize,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def main() -> None:
    # CSV listing the selected thesis models and their chosen checkpoint / CFG / gamma settings.
    selected_models_csv = "/develop/code/eval/thesis/selected_models.csv"
    # DeepLab reference-cache directory containing `pca_projector.pkl`,
    # `pca_sim_coords.npy`, and `pca_real_coords.npy`.
    reference_cache_dir = "/develop/code/eval/thesis/reference_cache/deeplab"
    # Output path for the thesis row figure.
    save_path = "/develop/code/eval/thesis/pca/deeplab_selected_models_pca_row.pdf"
    # Optional cap on plotted sim/real reference points per panel.
    max_reference_points = 150
    # Optional cap on plotted translated points per model panel. Use `None` for all points.
    max_translated_points = 150
    # Random seed used for point subsampling when max-point caps are active.
    random_seed = 42
    # Whether to draw a highlighted centroid marker for each translated cloud.
    show_centroid = True
    # Font size for each model-name subplot title.
    model_title_fontsize = 18.0
    # Font size for the x/y-axis labels.
    axis_label_fontsize = 16.0
    # Font size for the legend labels.
    legend_fontsize = 16.0
    # Rasterize points
    rasterized = False
    # Sim/real + translated point size
    reference_size = 20
    translated_size = 28
    # Sim/real + translated point transparency
    reference_alpha = 0.3
    translated_alpha = 0.45

    saved_path = plot_deeplab_pca_selected_row(
        selected_models_csv=selected_models_csv,
        reference_cache_dir=reference_cache_dir,
        save_path=save_path,
        max_reference_points=max_reference_points,
        max_translated_points=max_translated_points,
        random_seed=random_seed,
        show_centroid=show_centroid,
        model_title_fontsize=model_title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        legend_fontsize=legend_fontsize,
        rasterized=rasterized,
        reference_size=reference_size,
        translated_size=translated_size,
        reference_alpha=reference_alpha,
        translated_alpha=translated_alpha,
    )
    print(f"Saved PCA row plot to {saved_path}")


if __name__ == "__main__":
    main()
