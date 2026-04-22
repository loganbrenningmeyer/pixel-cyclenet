#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from cyclenet.eval.embed import DeepLabEmbedder


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_config(config_path: str | Path) -> DictConfig:
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str | Path) -> None:
    OmegaConf.save(config, save_path)


def cfg_select(config: DictConfig, key: str, default=None):
    value = OmegaConf.select(config, key)
    return default if value is None else value


def as_parent_dir_set(value) -> set[str] | None:
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
    pairs = []
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
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        if existing != metadata:
            raise ValueError(
                "Cache metadata does not match the current script configuration. "
                f"Either use a different cache directory or remove the stale cache at {metadata_path}."
            )
        return

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    sim_image_root = Path(config.data.sim_image_root)
    sim_label_root = Path(config.data.sim_label_root)
    real_image_root = Path(config.data.real_image_root)
    real_label_root = Path(config.data.real_label_root)

    translated_image_root_value = cfg_select(config, "data.translated_image_root", None)
    translated_image_root = (
        Path(translated_image_root_value) if translated_image_root_value is not None else None
    )
    translated_label_root = Path(config.data.translated_label_root)

    rgb_parent_dirs = as_parent_dir_set(config.data.rgb_parent_dirs)
    label_parent_dir_value = cfg_select(config, "data.label_parent_dir", None)
    label_parent_dir = str(label_parent_dir_value) if label_parent_dir_value is not None else None

    deeplab_ckpt_path = Path(config.embedding.deeplab_ckpt_path)
    feature_layer = str(cfg_select(config, "embedding.feature_layer", "prelogits"))
    num_classes = int(cfg_select(config, "embedding.num_classes", 8))
    batch_size = int(cfg_select(config, "embedding.batch_size", 32))

    max_samples_per_dataset = cfg_select(config, "embedding.max_samples_per_dataset", None)
    if max_samples_per_dataset is not None:
        max_samples_per_dataset = int(max_samples_per_dataset)

    seed = int(cfg_select(config, "run.seed", 42))
    cache_dir = Path(config.data.cache_dir)

    if not sim_image_root.exists():
        raise FileNotFoundError(f"sim_image_root does not exist: {sim_image_root}")
    if not sim_label_root.exists():
        raise FileNotFoundError(f"sim_label_root does not exist: {sim_label_root}")
    if not real_image_root.exists():
        raise FileNotFoundError(f"real_image_root does not exist: {real_image_root}")
    if not real_label_root.exists():
        raise FileNotFoundError(f"real_label_root does not exist: {real_label_root}")
    if translated_image_root is not None and not translated_image_root.exists():
        raise FileNotFoundError(f"translated_image_root does not exist: {translated_image_root}")
    if translated_image_root is not None and not translated_label_root.exists():
        raise FileNotFoundError(f"translated_label_root does not exist: {translated_label_root}")
    if not deeplab_ckpt_path.exists():
        raise FileNotFoundError(f"deeplab_ckpt_path does not exist: {deeplab_ckpt_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, cache_dir / "config.yaml")
    metadata = {
        "sim_image_root": str(sim_image_root.resolve()),
        "sim_label_root": str(sim_label_root.resolve()),
        "real_image_root": str(real_image_root.resolve()),
        "real_label_root": str(real_label_root.resolve()),
        "translated_image_root": (
            str(translated_image_root.resolve()) if translated_image_root is not None else None
        ),
        "translated_label_root": str(translated_label_root.resolve()),
        "rgb_parent_dirs": sorted(rgb_parent_dirs) if rgb_parent_dirs is not None else None,
        "label_parent_dir": label_parent_dir,
        "deeplab_ckpt_path": str(deeplab_ckpt_path.resolve()),
        "feature_layer": feature_layer,
        "num_classes": num_classes,
        "batch_size": batch_size,
        "max_samples_per_dataset": max_samples_per_dataset,
        "seed": seed,
    }
    write_metadata(metadata, cache_dir / "metadata.json")

    embedder = DeepLabEmbedder(
        ckpt_path=deeplab_ckpt_path,
        num_classes=num_classes,
        feature_layer=feature_layer,
    )

    dataset_specs = [
        ("sim", sim_image_root, sim_label_root),
        ("real", real_image_root, real_label_root),
    ]
    if translated_image_root is not None:
        dataset_specs.append(("translated", translated_image_root, translated_label_root))

    for dataset_name, image_root, label_root in dataset_specs:
        dataset_cache_dir = cache_dir / dataset_name
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

    print(f"\nSaved reusable class feature caches to {cache_dir}")


if __name__ == "__main__":
    main()
