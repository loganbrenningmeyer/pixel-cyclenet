from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cyclenet.eval.embed import DeepLabEmbedder


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
REQUIRED_SELECTED_MODEL_COLUMNS = {"model_name", "image_dir", "label_dir"}


def as_parent_dir_set(value: str | list[str] | tuple[str, ...] | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(v) for v in value}


def collect_rgb_paths(root: Path, rgb_parent_dirs: set[str] | None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")

    rgb_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and (rgb_parent_dirs is None or path.parent.name in rgb_parent_dirs)
    ]
    if not rgb_paths:
        if rgb_parent_dirs is None:
            raise ValueError(f"No RGB files found under {root}")
        raise ValueError(f"No RGB files found under {root} with parent dirs {sorted(rgb_parent_dirs)}")
    return rgb_paths


def resolve_matching_file(parent: Path, stem: str) -> Path:
    matches = [
        path
        for path in sorted(parent.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.stem == stem
    ]
    if not matches:
        raise FileNotFoundError(f"No file with stem '{stem}' found under {parent}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files with stem '{stem}' found under {parent}: {matches}")
    return matches[0]


def derive_label_path(
    rgb_path: Path,
    image_root: Path,
    label_root: Path,
    rgb_parent_dirs: set[str] | None,
    label_parent_dir: str | None,
) -> Path:
    try:
        rel_path = rgb_path.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(f"RGB path is outside image_root: {rgb_path}") from exc

    if rgb_parent_dirs is not None and rgb_path.parent.name not in rgb_parent_dirs:
        raise ValueError(
            f"Expected RGB file under one of {sorted(rgb_parent_dirs)}, got '{rgb_path.parent.name}': {rgb_path}"
        )

    rel_parts = list(rel_path.parts)
    if label_parent_dir is not None and len(rel_parts) >= 2:
        rel_parts[-2] = label_parent_dir
    label_path = label_root.joinpath(*rel_parts)
    if label_path.exists():
        return label_path.resolve()

    if label_parent_dir is None:
        flat_label_path = label_root / rgb_path.name
        if flat_label_path.exists():
            return flat_label_path.resolve()

    label_parent = label_path.parent
    if label_parent.exists():
        return resolve_matching_file(label_parent, rgb_path.stem).resolve()

    raise FileNotFoundError(f"Missing label for {rgb_path}: expected {label_path}")


def collect_rgb_label_pairs(
    image_root: Path,
    label_root: Path,
    rgb_parent_dirs: set[str] | None,
    label_parent_dir: str | None,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for rgb_path in collect_rgb_paths(image_root, rgb_parent_dirs):
        label_path = derive_label_path(
            rgb_path=rgb_path,
            image_root=image_root,
            label_root=label_root,
            rgb_parent_dirs=rgb_parent_dirs,
            label_parent_dir=label_parent_dir,
        )
        pairs.append((rgb_path.resolve(), label_path))
    return pairs


def sample_pairs(
    pairs: list[tuple[Path, Path]],
    max_count: int | None,
    seed: int,
) -> list[tuple[Path, Path]]:
    if max_count is None:
        return pairs
    if max_count <= 0:
        raise ValueError(f"max_count must be positive, got {max_count}")
    if len(pairs) <= max_count:
        return pairs

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(pairs), size=max_count, replace=False))
    return [pairs[i] for i in idx]


def load_or_create_pair_manifest(
    image_root: Path,
    label_root: Path,
    rgb_parent_dirs: set[str] | None,
    label_parent_dir: str | None,
    manifest_path: Path,
    max_count: int | None,
    seed: int,
) -> list[tuple[Path, Path]]:
    if manifest_path.exists():
        rows = list(csv.DictReader(manifest_path.open("r", newline="")))
        pairs = [(Path(row["img_path"]), Path(row["label_path"])) for row in rows]
        missing = [pair for pair in pairs if not pair[0].exists() or not pair[1].exists()]
        if missing:
            first = missing[0]
            raise FileNotFoundError(
                f"Cached manifest contains missing files: image={first[0]}, label={first[1]}"
            )
        return pairs

    pairs = collect_rgb_label_pairs(
        image_root=image_root,
        label_root=label_root,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
    )
    pairs = sample_pairs(pairs, max_count=max_count, seed=seed)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["img_path", "label_path"])
        writer.writeheader()
        for img_path, label_path in pairs:
            writer.writerow({"img_path": str(img_path), "label_path": str(label_path)})

    return pairs


def write_metadata(metadata: dict[str, object], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def validate_or_refresh_metadata(
    metadata: dict[str, object],
    metadata_path: Path,
    compare_keys: list[str],
) -> None:
    if not metadata_path.exists():
        write_metadata(metadata, metadata_path)
        return

    existing = json.loads(metadata_path.read_text())
    mismatched_keys = [
        key for key in compare_keys if existing.get(key) != metadata.get(key)
    ]
    if mismatched_keys:
        raise ValueError(
            "Cache metadata does not match the current script configuration for keys "
            f"{mismatched_keys}. Either use a different cache directory or remove the stale "
            f"cache at {metadata_path}."
        )

    if existing != metadata:
        write_metadata(metadata, metadata_path)


def load_class_embeddings(cache_path: Path) -> dict[int, np.ndarray]:
    bundle = np.load(cache_path)
    return {
        int(name.removeprefix("class_")): bundle[name]
        for name in sorted(bundle.files)
        if name.startswith("class_")
    }


def load_or_compute_class_embeddings(
    embedder: DeepLabEmbedder,
    pairs: list[tuple[Path, Path]],
    batch_size: int,
    cache_path: Path,
) -> dict[int, np.ndarray]:
    if cache_path.exists():
        return load_class_embeddings(cache_path)

    return embedder.embed_by_class(
        img_paths=[str(img_path) for img_path, _ in pairs],
        label_paths=[str(label_path) for _, label_path in pairs],
        batch_size=batch_size,
        save_path=cache_path,
    )


def summarize_embeddings(dataset_name: str, feats_by_class: dict[int, np.ndarray]) -> None:
    print(f"\n[{dataset_name}] class feature counts")
    print("class_id\tcount\tfeature_dim")
    for class_id in sorted(feats_by_class):
        feats = feats_by_class[class_id]
        print(f"{class_id}\t{len(feats)}\t{feats.shape[1] if feats.ndim == 2 else 0}")


def load_selected_models(selected_models_csv: str | Path) -> pd.DataFrame:
    csv_path = Path(selected_models_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Selected models CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    missing = sorted(REQUIRED_SELECTED_MODEL_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"Selected models CSV is missing required columns: {', '.join(missing)}"
        )
    if df.empty:
        raise ValueError(f"Selected models CSV is empty: {csv_path}")
    if df["model_name"].duplicated().any():
        duplicates = sorted(df.loc[df["model_name"].duplicated(), "model_name"].unique())
        raise ValueError(
            f"Selected models CSV must have unique model_name values, found duplicates: {duplicates}"
        )
    return df


def cache_dataset(
    *,
    dataset_name: str,
    image_root: Path,
    label_root: Path,
    cache_root: Path,
    rgb_parent_dirs: set[str] | None,
    label_parent_dir: str | None,
    max_samples_per_dataset: int | None,
    seed: int,
    embedder: DeepLabEmbedder,
    batch_size: int,
) -> dict[int, np.ndarray]:
    dataset_cache_dir = cache_root / dataset_name
    pairs = load_or_create_pair_manifest(
        image_root=image_root,
        label_root=label_root,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
        manifest_path=dataset_cache_dir / "pairs.csv",
        max_count=max_samples_per_dataset,
        seed=seed,
    )
    feats_by_class = load_or_compute_class_embeddings(
        embedder=embedder,
        pairs=pairs,
        batch_size=batch_size,
        cache_path=dataset_cache_dir / "deeplab_class_features.npz",
    )
    summarize_embeddings(dataset_name, feats_by_class)
    return feats_by_class


def cache_selected_model_class_features(
    selected_models_csv: str | Path,
    sim_image_root: str | Path,
    sim_label_root: str | Path,
    real_image_root: str | Path,
    real_label_root: str | Path,
    cache_dir: str | Path,
    deeplab_ckpt_path: str | Path,
    feature_layer: str = "prelogits",
    num_classes: int = 8,
    batch_size: int = 32,
    max_samples_per_dataset: int | None = None,
    rgb_parent_dirs: str | list[str] | tuple[str, ...] | None = None,
    label_parent_dir: str | None = None,
    seed: int = 42,
) -> Path:
    selected_df = load_selected_models(selected_models_csv)

    sim_image_root = Path(sim_image_root)
    sim_label_root = Path(sim_label_root)
    real_image_root = Path(real_image_root)
    real_label_root = Path(real_label_root)
    cache_dir = Path(cache_dir)
    deeplab_ckpt_path = Path(deeplab_ckpt_path)
    rgb_parent_dir_set = as_parent_dir_set(rgb_parent_dirs)

    for path, label in [
        (sim_image_root, "sim_image_root"),
        (sim_label_root, "sim_label_root"),
        (real_image_root, "real_image_root"),
        (real_label_root, "real_label_root"),
        (deeplab_ckpt_path, "deeplab_ckpt_path"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    reference_metadata = {
        "selected_models_csv": str(Path(selected_models_csv).resolve()),
        "sim_image_root": str(sim_image_root.resolve()),
        "sim_label_root": str(sim_label_root.resolve()),
        "real_image_root": str(real_image_root.resolve()),
        "real_label_root": str(real_label_root.resolve()),
        "rgb_parent_dirs": sorted(rgb_parent_dir_set) if rgb_parent_dir_set is not None else None,
        "label_parent_dir": label_parent_dir,
        "deeplab_ckpt_path": str(deeplab_ckpt_path.resolve()),
        "feature_layer": feature_layer,
        "num_classes": int(num_classes),
        "batch_size": int(batch_size),
        "max_samples_per_dataset": max_samples_per_dataset,
        "seed": int(seed),
    }
    reference_compare_keys = [
        "sim_image_root",
        "sim_label_root",
        "real_image_root",
        "real_label_root",
        "rgb_parent_dirs",
        "label_parent_dir",
        "deeplab_ckpt_path",
        "feature_layer",
        "num_classes",
        "batch_size",
        "max_samples_per_dataset",
        "seed",
    ]
    validate_or_refresh_metadata(
        reference_metadata,
        cache_dir / "reference_metadata.json",
        compare_keys=reference_compare_keys,
    )

    embedder = DeepLabEmbedder(
        ckpt_path=deeplab_ckpt_path,
        num_classes=int(num_classes),
        feature_layer=feature_layer,
    )

    cache_dataset(
        dataset_name="sim",
        image_root=sim_image_root,
        label_root=sim_label_root,
        cache_root=cache_dir,
        rgb_parent_dirs=rgb_parent_dir_set,
        label_parent_dir=label_parent_dir,
        max_samples_per_dataset=max_samples_per_dataset,
        seed=int(seed),
        embedder=embedder,
        batch_size=int(batch_size),
    )
    cache_dataset(
        dataset_name="real",
        image_root=real_image_root,
        label_root=real_label_root,
        cache_root=cache_dir,
        rgb_parent_dirs=rgb_parent_dir_set,
        label_parent_dir=label_parent_dir,
        max_samples_per_dataset=max_samples_per_dataset,
        seed=int(seed),
        embedder=embedder,
        batch_size=int(batch_size),
    )

    for row_idx, row in selected_df.iterrows():
        model_name = str(row["model_name"])
        translated_image_root = Path(str(row["image_dir"]))
        translated_label_root = Path(str(row["label_dir"]))

        if not translated_image_root.exists():
            raise FileNotFoundError(
                f"Translated image dir does not exist for model '{model_name}': {translated_image_root}"
            )
        if not translated_label_root.exists():
            raise FileNotFoundError(
                f"Translated label dir does not exist for model '{model_name}': {translated_label_root}"
            )

        model_cache_dir = cache_dir / model_name
        model_cache_dir.mkdir(parents=True, exist_ok=True)

        row_metadata = {
            **reference_metadata,
            "row_index": int(row_idx),
            "model_name": model_name,
            "translated_image_root": str(translated_image_root.resolve()),
            "translated_label_root": str(translated_label_root.resolve()),
            "selected_model_row": {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in row.to_dict().items()
            },
        }
        row_compare_keys = reference_compare_keys + [
            "model_name",
            "translated_image_root",
            "translated_label_root",
        ]
        validate_or_refresh_metadata(
            row_metadata,
            model_cache_dir / "metadata.json",
            compare_keys=row_compare_keys,
        )
        Path(model_cache_dir / "selected_model_row.json").write_text(
            json.dumps(row_metadata["selected_model_row"], indent=2, sort_keys=True) + "\n"
        )

        cache_dataset(
            dataset_name="translated",
            image_root=translated_image_root,
            label_root=translated_label_root,
            cache_root=model_cache_dir,
            rgb_parent_dirs=rgb_parent_dir_set,
            label_parent_dir=label_parent_dir,
            max_samples_per_dataset=max_samples_per_dataset,
            seed=int(seed) + row_idx,
            embedder=embedder,
            batch_size=int(batch_size),
        )
        print(f"Saved translated class feature cache for {model_name} to {model_cache_dir}")

    print(f"\nFinished caching class feature vectors for selected models under {cache_dir}")
    return cache_dir


def main() -> None:
    # CSV listing the selected models and their translated image / label directories.
    selected_models_csv = "/develop/code/eval/thesis/selected_models.csv"
    # Root directory for simulated RGB imagery.
    sim_image_root = "/develop/data/remote_sensing/tiled/projection/sim_proj/opt"
    # Root directory for simulated label masks.
    sim_label_root = "/develop/data/remote_sensing/tiled/projection/sim_proj/gt_ss_mask"
    # Root directory for real RGB imagery.
    real_image_root = "/develop/data/remote_sensing/tiled/projection/oem_proj/opt"
    # Root directory for real label masks.
    real_label_root = "/develop/data/remote_sensing/tiled/projection/oem_proj/gt_ss_mask"
    # Root cache directory. Shared `sim/` and `real/` caches live here and each
    # model gets `cache_dir/<model_name>/translated/`.
    cache_dir = "/develop/code/eval/thesis/class_feature_cache"
    # DeepLab checkpoint used for class-conditional feature extraction.
    deeplab_ckpt_path = (
        "/cgi/data/nvesd/workspaces/logan/code/land_mapping/runs/deeplab/oem_subset/real-sim/"
        "training/checkpoints/step-50000.ckpt"
    )
    # DeepLab feature layer to pool over for class-wise vectors.
    feature_layer = "prelogits"
    # Number of semantic classes excluding ignore.
    num_classes = 8
    # Batch size for embedding extraction.
    batch_size = 32
    # Optional max number of image/mask pairs to cache per dataset. Use `None` for all.
    max_samples_per_dataset = None
    # Optional RGB parent folder filter such as `opt`. Use `None` to scan the full tree.
    rgb_parent_dirs = None
    # Optional label parent directory replacement such as `gt_ss_mask`. Use `None`
    # when image and label trees already align by relative path.
    label_parent_dir = None
    # Random seed used for optional pair subsampling and translated-run sampling offsets.
    seed = 42

    saved_root = cache_selected_model_class_features(
        selected_models_csv=selected_models_csv,
        sim_image_root=sim_image_root,
        sim_label_root=sim_label_root,
        real_image_root=real_image_root,
        real_label_root=real_label_root,
        cache_dir=cache_dir,
        deeplab_ckpt_path=deeplab_ckpt_path,
        feature_layer=feature_layer,
        num_classes=num_classes,
        batch_size=batch_size,
        max_samples_per_dataset=max_samples_per_dataset,
        rgb_parent_dirs=rgb_parent_dirs,
        label_parent_dir=label_parent_dir,
        seed=seed,
    )
    print(f"Saved selected-model class feature caches to {saved_root}")


if __name__ == "__main__":
    main()
