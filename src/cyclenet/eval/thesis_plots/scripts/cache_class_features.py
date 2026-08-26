from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor

from cyclenet.data.dataset import load_label_mask
from cyclenet.eval.embed import DeepLabEmbedder


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
REQUIRED_SELECTED_MODEL_COLUMNS = {"model_name", "image_dir", "label_dir"}
CLASS_FEATURE_CACHE_FILENAMES = {
    "deeplab": "deeplab_class_features.npz",
    "fid": "fid_class_features.npz",
}
T = TypeVar("T")


def _batched(items: list[T], batch_size: int) -> list[list[T]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


class ClassFeatureEmbedder(Protocol):
    def embed_by_class(
        self,
        img_paths: list[str],
        label_paths: list[str],
        batch_size: int = 64,
        save_path: str | Path | None = None,
    ) -> dict[int, np.ndarray]: ...


def normalize_feature_extractor(feature_extractor: str) -> str:
    normalized = feature_extractor.lower().strip()
    if normalized not in CLASS_FEATURE_CACHE_FILENAMES:
        raise ValueError(
            f"Unsupported feature_extractor '{feature_extractor}'. "
            f"Expected one of: {', '.join(sorted(CLASS_FEATURE_CACHE_FILENAMES))}."
        )
    return normalized


def class_feature_cache_filename(feature_extractor: str) -> str:
    return CLASS_FEATURE_CACHE_FILENAMES[normalize_feature_extractor(feature_extractor)]


def class_feature_metadata_name(feature_extractor: str, base_name: str) -> str:
    feature_extractor = normalize_feature_extractor(feature_extractor)
    if feature_extractor == "deeplab":
        return base_name
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    return f"{stem}_{feature_extractor}{suffix}"


def save_class_embeddings(
    feats_by_class: dict[int, np.ndarray],
    save_path: str | Path | None,
) -> None:
    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        **{
            f"class_{class_id}": feats
            for class_id, feats in sorted(feats_by_class.items())
        },
    )


class FIDClassEmbedder:
    def __init__(
        self,
        num_classes: int = 8,
        feature_layer: str = "Mixed_7c",
        device: str | torch.device | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = int(num_classes)
        self.feature_layer = feature_layer

        weights = Inception_V3_Weights.IMAGENET1K_V1
        base_model = inception_v3(weights=weights).to(self.device)
        base_model.eval()
        self.model = create_feature_extractor(
            base_model,
            return_nodes={feature_layer: "feat"},
        ).to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()

    def embed_by_class(
        self,
        img_paths: list[str],
        label_paths: list[str],
        batch_size: int = 64,
        save_path: str | Path | None = None,
    ) -> dict[int, np.ndarray]:
        if len(img_paths) != len(label_paths):
            raise ValueError(
                f"img_paths and label_paths must have the same length, got "
                f"{len(img_paths)} and {len(label_paths)}"
            )

        feats_by_class: dict[int, list[np.ndarray]] = {
            class_id: [] for class_id in range(1, self.num_classes + 1)
        }
        feature_dim: int | None = None
        paired_paths = list(zip(img_paths, label_paths, strict=True))

        with torch.inference_mode():
            for batch_pairs in _batched(paired_paths, batch_size):
                images = []
                masks = []

                for img_path, label_path in batch_pairs:
                    with Image.open(img_path) as img:
                        images.append(self.transforms(img.convert("RGB")))

                    mask_np = np.asarray(load_label_mask(Path(label_path)), dtype=np.int64)
                    masks.append(torch.from_numpy(mask_np))

                x = torch.stack(images, dim=0).to(self.device)
                feats = self.model(x)["feat"]
                if feats.ndim == 2:
                    feats = feats[:, :, None, None]
                elif feats.ndim != 4:
                    raise RuntimeError(
                        f"Expected FID feature layer '{self.feature_layer}' to return a "
                        f"2D or 4D tensor, got shape {tuple(feats.shape)}."
                    )

                _, channels, feat_height, feat_width = feats.shape
                feature_dim = channels

                mask_tensor = torch.stack(masks, dim=0).unsqueeze(1).float().to(self.device)
                mask_tensor = F.interpolate(
                    mask_tensor,
                    size=(feat_height, feat_width),
                    mode="nearest",
                ).squeeze(1).long()

                for class_id in range(1, self.num_classes + 1):
                    class_mask = (mask_tensor == class_id).unsqueeze(1)
                    class_counts = class_mask.sum(dim=(2, 3)).squeeze(1)
                    valid = class_counts > 0
                    if not valid.any():
                        continue

                    masked_sum = (feats * class_mask).sum(dim=(2, 3))
                    pooled = masked_sum[valid] / class_counts[valid].unsqueeze(1).clamp_min(1)
                    feats_by_class[class_id].append(
                        pooled.cpu().numpy().reshape(-1, channels).astype(np.float32, copy=False)
                    )

        feats_by_class_np = {
            class_id: (
                np.concatenate(class_feats, axis=0)
                if class_feats
                else np.empty((0, feature_dim or 0), dtype=np.float32)
            )
            for class_id, class_feats in feats_by_class.items()
        }
        save_class_embeddings(feats_by_class_np, save_path)
        return feats_by_class_np


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
    embedder: ClassFeatureEmbedder,
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
    feature_extractor: str,
    embedder: ClassFeatureEmbedder,
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
        cache_path=dataset_cache_dir / class_feature_cache_filename(feature_extractor),
    )
    summarize_embeddings(f"{feature_extractor}:{dataset_name}", feats_by_class)
    return feats_by_class


def cache_selected_model_class_features(
    selected_models_csv: str | Path,
    sim_image_root: str | Path,
    sim_label_root: str | Path,
    real_image_root: str | Path,
    real_label_root: str | Path,
    cache_dir: str | Path,
    deeplab_ckpt_path: str | Path | None = None,
    feature_layer: str | None = None,
    feature_extractor: str = "deeplab",
    num_classes: int = 8,
    batch_size: int = 32,
    max_samples_per_dataset: int | None = None,
    rgb_parent_dirs: str | list[str] | tuple[str, ...] | None = None,
    label_parent_dir: str | None = None,
    seed: int = 42,
) -> Path:
    selected_df = load_selected_models(selected_models_csv)
    feature_extractor = normalize_feature_extractor(feature_extractor)
    resolved_feature_layer = feature_layer or (
        "prelogits" if feature_extractor == "deeplab" else "Mixed_7c"
    )

    sim_image_root = Path(sim_image_root)
    sim_label_root = Path(sim_label_root)
    real_image_root = Path(real_image_root)
    real_label_root = Path(real_label_root)
    cache_dir = Path(cache_dir)
    resolved_deeplab_ckpt_path = (
        Path(deeplab_ckpt_path) if deeplab_ckpt_path is not None else None
    )
    rgb_parent_dir_set = as_parent_dir_set(rgb_parent_dirs)

    for path, label in [
        (sim_image_root, "sim_image_root"),
        (sim_label_root, "sim_label_root"),
        (real_image_root, "real_image_root"),
        (real_label_root, "real_label_root"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    if feature_extractor == "deeplab":
        if resolved_deeplab_ckpt_path is None:
            raise ValueError("deeplab_ckpt_path is required when feature_extractor='deeplab'")
        if not resolved_deeplab_ckpt_path.exists():
            raise FileNotFoundError(
                f"deeplab_ckpt_path does not exist: {resolved_deeplab_ckpt_path}"
            )

    cache_dir.mkdir(parents=True, exist_ok=True)

    reference_metadata = {
        "selected_models_csv": str(Path(selected_models_csv).resolve()),
        "sim_image_root": str(sim_image_root.resolve()),
        "sim_label_root": str(sim_label_root.resolve()),
        "real_image_root": str(real_image_root.resolve()),
        "real_label_root": str(real_label_root.resolve()),
        "rgb_parent_dirs": sorted(rgb_parent_dir_set) if rgb_parent_dir_set is not None else None,
        "label_parent_dir": label_parent_dir,
        "deeplab_ckpt_path": (
            str(resolved_deeplab_ckpt_path.resolve())
            if feature_extractor == "deeplab" and resolved_deeplab_ckpt_path is not None
            else None
        ),
        "feature_extractor": feature_extractor,
        "feature_layer": resolved_feature_layer,
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
        "feature_layer",
        "num_classes",
        "batch_size",
        "max_samples_per_dataset",
        "seed",
    ]
    if feature_extractor != "deeplab":
        reference_compare_keys.append("feature_extractor")
    if feature_extractor == "deeplab":
        reference_compare_keys.append("deeplab_ckpt_path")

    validate_or_refresh_metadata(
        reference_metadata,
        cache_dir / class_feature_metadata_name(feature_extractor, "reference_metadata.json"),
        compare_keys=reference_compare_keys,
    )

    if feature_extractor == "deeplab":
        if resolved_deeplab_ckpt_path is None:
            raise AssertionError("resolved_deeplab_ckpt_path must be set for DeepLab features")
        embedder: ClassFeatureEmbedder = DeepLabEmbedder(
            ckpt_path=resolved_deeplab_ckpt_path,
            num_classes=int(num_classes),
            feature_layer=resolved_feature_layer,
        )
    else:
        embedder = FIDClassEmbedder(
            num_classes=int(num_classes),
            feature_layer=resolved_feature_layer,
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
        feature_extractor=feature_extractor,
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
        feature_extractor=feature_extractor,
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
            model_cache_dir / class_feature_metadata_name(feature_extractor, "metadata.json"),
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
            feature_extractor=feature_extractor,
            embedder=embedder,
            batch_size=int(batch_size),
        )
        print(
            f"Saved {feature_extractor} translated class feature cache for "
            f"{model_name} to {model_cache_dir}"
        )

    print(
        f"\nFinished caching {feature_extractor} class feature vectors for selected "
        f"models under {cache_dir}"
    )
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
    # Feature extractor for class-wise vectors. Supported values: `deeplab` and `fid`.
    feature_extractor = "deeplab"
    # Feature layer to pool over for class-wise vectors. Use `None` for the
    # extractor default: `prelogits` for DeepLab, `Mixed_7c` for FID/Inception.
    feature_layer = None
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
        feature_extractor=feature_extractor,
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
