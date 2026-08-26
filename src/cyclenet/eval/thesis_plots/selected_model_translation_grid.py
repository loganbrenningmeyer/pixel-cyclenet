from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cyclenet.eval.plotting.set_style import MODEL_NAMES, apply_style

apply_style()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
REQUIRED_SELECTED_MODEL_COLUMNS = {"model_name", "image_dir"}


@dataclass(frozen=True)
class SelectedModel:
    model_name: str
    image_dir: Path


@dataclass
class ImageLookup:
    root: Path
    exact_rel_map: dict[str, Path]
    stem_map: dict[str, Path]


def _display_model_name(model_name: str) -> str:
    display_name = MODEL_NAMES.get(model_name, model_name.replace("_", " "))
    if display_name == "RGB + SPADE (BN Only)":
        return "RGB + SPADE\n(BN Only)"
    return display_name


def _collect_image_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    image_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not image_paths:
        raise ValueError(f"No image files found under {root}")
    return image_paths


def _load_selected_models(
    selected_models_csv: str | Path,
    selected_model_names: list[str] | None = None,
) -> list[SelectedModel]:
    csv_path = Path(selected_models_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Selected models CSV does not exist: {csv_path}")

    with csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Selected models CSV is empty: {csv_path}")

    fieldnames = set(rows[0].keys())
    missing = sorted(REQUIRED_SELECTED_MODEL_COLUMNS - fieldnames)
    if missing:
        raise ValueError(
            f"Selected models CSV is missing required columns: {', '.join(missing)}"
        )

    models = [
        SelectedModel(
            model_name=str(row["model_name"]).strip(),
            image_dir=Path(str(row["image_dir"])).resolve(),
        )
        for row in rows
    ]

    seen: set[str] = set()
    duplicates: set[str] = set()
    for model in models:
        if model.model_name in seen:
            duplicates.add(model.model_name)
        seen.add(model.model_name)
    if duplicates:
        raise ValueError(
            f"Selected models CSV must have unique model_name values, found duplicates: {sorted(duplicates)}"
        )

    if selected_model_names is None:
        return models

    by_name = {model.model_name: model for model in models}
    missing_names = [name for name in selected_model_names if name not in by_name]
    if missing_names:
        raise ValueError(
            f"Requested model names are missing from {csv_path}: {missing_names}"
        )

    return [by_name[name] for name in selected_model_names]


def _build_image_lookup(root: Path) -> ImageLookup:
    exact_rel_map: dict[str, Path] = {}
    stem_paths: dict[str, list[Path]] = defaultdict(list)

    for image_path in _collect_image_paths(root):
        rel_key = image_path.relative_to(root).as_posix()
        exact_rel_map[rel_key] = image_path
        stem_paths[image_path.stem].append(image_path)

    stem_map: dict[str, Path] = {}
    for stem, paths in stem_paths.items():
        if len(paths) == 1:
            stem_map[stem] = paths[0]

    return ImageLookup(root=root, exact_rel_map=exact_rel_map, stem_map=stem_map)


def _resolve_matching_translated_path(
    sim_image_path: Path,
    sim_image_root: Path,
    lookup: ImageLookup,
) -> Path:
    rel_path = sim_image_path.relative_to(sim_image_root)
    exact_key = rel_path.as_posix()
    exact_match = lookup.exact_rel_map.get(exact_key)
    if exact_match is not None:
        return exact_match

    candidate_parent = lookup.root / rel_path.parent
    if candidate_parent.is_dir():
        parent_matches = [
            path
            for path in sorted(candidate_parent.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.stem == rel_path.stem
        ]
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            raise RuntimeError(
                f"Multiple translated files match '{rel_path.stem}' under {candidate_parent}: {parent_matches}"
            )

    stem_match = lookup.stem_map.get(rel_path.stem)
    if stem_match is not None:
        return stem_match

    raise FileNotFoundError(
        f"Could not match simulated image {sim_image_path} within translated image root {lookup.root}"
    )


def _select_sample_paths(
    sim_image_root: Path,
    num_samples: int,
    random_seed: int,
) -> list[Path]:
    image_paths = _collect_image_paths(sim_image_root)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_samples > len(image_paths):
        raise ValueError(
            f"Requested {num_samples} samples from {sim_image_root}, but only found {len(image_paths)} images"
        )

    rng = random.Random(random_seed)
    sample_indices = sorted(rng.sample(range(len(image_paths)), k=num_samples))
    return [image_paths[idx] for idx in sample_indices]


def _load_rgb_image(image_path: Path, image_size: int | None) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image_size is not None:
        image = cv2.resize(
            image,
            (int(image_size), int(image_size)),
            interpolation=cv2.INTER_LINEAR,
        )

    return np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)


def plot_selected_model_translation_grid(
    selected_models_csv: str | Path,
    sim_image_root: str | Path,
    save_path: str | Path,
    num_samples: int = 6,
    image_size: int | None = 224,
    random_seed: int = 42,
    selected_model_names: list[str] | None = None,
    figure_title: str | None = None,
    show_row_labels: bool = False,
    title_fontsize: float = 13.0,
    column_title_fontsize: float = 12.0,
    row_label_fontsize: float = 8.0,
    dpi: int = 200,
    column_width: float = 2.15,
    row_height: float = 2.15,
) -> Path:
    sim_image_root = Path(sim_image_root).resolve()
    save_path = Path(save_path)

    selected_models = _load_selected_models(
        selected_models_csv=selected_models_csv,
        selected_model_names=selected_model_names,
    )
    if not selected_models:
        raise ValueError("No selected models were resolved for the translation grid.")

    sim_sample_paths = _select_sample_paths(
        sim_image_root=sim_image_root,
        num_samples=num_samples,
        random_seed=random_seed,
    )
    lookups = {
        model.model_name: _build_image_lookup(model.image_dir)
        for model in selected_models
    }

    n_rows = len(sim_sample_paths)
    n_cols = 1 + len(selected_models)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(column_width * n_cols, row_height * n_rows),
        dpi=dpi,
        squeeze=False,
        constrained_layout=True,
    )

    if figure_title:
        fig.suptitle(figure_title, fontsize=title_fontsize)

    axes[0, 0].set_title("Sim", fontsize=column_title_fontsize)
    for col_idx, model in enumerate(selected_models, start=1):
        axes[0, col_idx].set_title(
            _display_model_name(model.model_name),
            fontsize=column_title_fontsize,
        )

    for row_idx, sim_path in enumerate(sim_sample_paths):
        sim_image = _load_rgb_image(sim_path, image_size=image_size)
        axes[row_idx, 0].imshow(sim_image, interpolation="nearest")
        axes[row_idx, 0].axis("off")

        if show_row_labels:
            axes[row_idx, 0].set_ylabel(
                sim_path.stem,
                rotation=0,
                ha="right",
                va="center",
                labelpad=24,
                fontsize=row_label_fontsize,
            )

        for col_idx, model in enumerate(selected_models, start=1):
            translated_path = _resolve_matching_translated_path(
                sim_image_path=sim_path,
                sim_image_root=sim_image_root,
                lookup=lookups[model.model_name],
            )
            translated_image = _load_rgb_image(translated_path, image_size=image_size)
            axes[row_idx, col_idx].imshow(translated_image, interpolation="nearest")
            axes[row_idx, col_idx].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    print("Selected simulated source images:")
    for sim_path in sim_sample_paths:
        print(f"  {sim_path}")
    print(f"Saved selected-model translation grid to {save_path}")

    return save_path


def main() -> None:
    # CSV listing the thesis-selected translation models and their translated image directories.
    selected_models_csv = "/home/logan/projects/pixel-cyclenet/eval/thesis/selected_models.csv"
    # Root directory containing the simulated source RGB images to sample from.
    sim_image_root = "/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/projection/sim_proj/opt"
    # Output path for the saved 6x6 thesis grid figure.
    save_path = "/home/logan/projects/pixel-cyclenet/eval/thesis/figs/grids/selected_model_translation_grid.pdf"
    # Number of simulated source images to sample as rows in the grid.
    num_samples = 6
    # Spatial resolution used when loading the simulated and translated images for plotting.
    image_size = 224
    # Random seed for reproducible simulated-image sampling.
    random_seed = 42
    # Optional explicit model ordering; set to None to use every row from selected_models_csv in CSV order.
    selected_model_names = None
    # Optional figure title shown above the grid.
    figure_title = None
    # Whether to annotate each row with the sampled source filename stem.
    show_row_labels = False

    plot_selected_model_translation_grid(
        selected_models_csv=selected_models_csv,
        sim_image_root=sim_image_root,
        save_path=save_path,
        num_samples=num_samples,
        image_size=image_size,
        random_seed=random_seed,
        selected_model_names=selected_model_names,
        figure_title=figure_title,
        show_row_labels=show_row_labels,
    )


if __name__ == "__main__":
    main()
