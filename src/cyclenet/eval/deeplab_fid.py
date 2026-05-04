import csv
from pathlib import Path

import numpy as np
import torch

from cyclenet.eval.embed import DeepLabEmbedder
from cyclenet.eval.frechet_dist import frechet_distance


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_prefixed_float(name: str, prefix: str) -> float:
    if not name.startswith(prefix):
        raise ValueError(f"Expected '{name}' to start with '{prefix}'.")
    return float(name[len(prefix) :])


def parse_step_index(name: str) -> int:
    if not name.startswith("step-"):
        raise ValueError(f"Expected '{name}' to start with 'step-'.")
    return int(name.removeprefix("step-"))


def iter_candidate_dirs(step_dir: str | Path) -> list[tuple[int, float, float, Path]]:
    root = Path(step_dir)
    if not root.exists():
        raise FileNotFoundError(f"step_dir does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"step_dir must be a directory, got: {root}")
    if not root.name.startswith("step-"):
        raise ValueError(f"Expected step_dir name like 'step-*', got '{root.name}'")

    candidates: list[tuple[int, float, float, Path]] = []
    step = parse_step_index(root.name)

    strength_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("strength-")],
        key=lambda path: parse_prefixed_float(path.name, "strength-"),
    )
    for strength_dir in strength_dirs:
        noise_strength = parse_prefixed_float(strength_dir.name, "strength-")
        cfg_dirs = sorted(
            [path for path in strength_dir.iterdir() if path.is_dir() and path.name.startswith("cfg-")],
            key=lambda path: parse_prefixed_float(path.name, "cfg-"),
        )
        for cfg_dir in cfg_dirs:
            cfg_weight = parse_prefixed_float(cfg_dir.name, "cfg-")
            candidates.append((step, noise_strength, cfg_weight, cfg_dir))

    if not candidates:
        raise ValueError(
            f"No strength/cfg directories were found under {root}. "
            "Expected a layout like step-*/strength-*/cfg-*."
        )

    return candidates


def collect_images(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Image root does not exist: {root_dir}")

    img_paths: list[Path] = []
    for path in sorted(root_dir.rglob("*")):
        if (
            not path.is_file()
            or path.parent.name == "gt_ss_mask"
            or path.suffix.lower() not in IMAGE_EXTS
        ):
            continue
        img_paths.append(path)

    if not img_paths:
        raise ValueError(f"No image files found under {root_dir}")

    return img_paths


def load_embedding_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache does not exist: {path}")

    feats = np.load(path)
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D embedding array at {path}, got shape {feats.shape}")
    if len(feats) == 0:
        raise ValueError(f"Embedding array is empty at {path}")
    return feats


def load_or_compute_embeddings(
    embedder: DeepLabEmbedder,
    img_root: str | Path,
    batch_size: int,
    cache_path: str | Path,
) -> np.ndarray:
    cache_path = Path(cache_path)
    if cache_path.exists():
        return load_embedding_array(cache_path)

    img_paths = collect_images(img_root)
    return embedder.embed(
        [str(path) for path in img_paths],
        batch_size=batch_size,
        save_path=cache_path,
    )


def deeplab_fid_sweep(
    reference_dir: Path | str,
    cyclenet_sim_dir: Path | str,
    steps: list[int],
    deeplab_ckpt_path: Path | str = "/cgi/data/nvesd/workspaces/logan/code/land_mapping/runs/deeplab/oem_subset/real-sim/training/checkpoints/step-50000.ckpt",
    feature_layer: str = "prelogits",
    reference_cache_dir: Path | str = "/cgi/data/nvesd/workspaces/logan/data/eval/cyclenet/remote_sensing/project_translated/reference_cache/deeplab",
    translated_embed_filename: str = "deeplab_translated_embed.npy",
):
    # Number of images embedded together on each forward pass.
    batch_size = 32

    for step in steps:

        step_dir = Path(cyclenet_sim_dir) / f"step-{step}"

        # CSV path where the aggregated DeepLab Fréchet stats for this step will be saved.
        csv_out_path = step_dir / "deeplab_fd_stats.csv"

        if not deeplab_ckpt_path.exists():
            raise FileNotFoundError(f"DeepLab checkpoint does not exist: {deeplab_ckpt_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = DeepLabEmbedder(
            ckpt_path=deeplab_ckpt_path,
            feature_layer=feature_layer,
            device=device,
        )

        reference_cache_dir.mkdir(parents=True, exist_ok=True)
        real_embed_path = reference_cache_dir / "real_embed.npy"
        real_feats = load_or_compute_embeddings(
            embedder=embedder,
            img_root=reference_dir,
            batch_size=batch_size,
            cache_path=real_embed_path,
        )

        summary_rows: list[dict[str, object]] = []
        for step, noise_strength, cfg_weight, translated_dir in iter_candidate_dirs(step_dir):
            translated_embed_path = translated_dir / translated_embed_filename
            translated_feats = load_or_compute_embeddings(
                embedder=embedder,
                img_root=translated_dir,
                batch_size=batch_size,
                cache_path=translated_embed_path,
            )
            deeplab_fd = frechet_distance(translated_feats, real_feats)

            print(
                f"step-{step} / strength-{noise_strength:.1f} / cfg-{cfg_weight:.1f}".center(50, "=")
            )
            print(f"[ DeepLab FD ]: {deeplab_fd:.6f}")

            summary_rows.append(
                {
                    "step": step,
                    "noise_strength": noise_strength,
                    "cfg_weight": cfg_weight,
                    "translated_dir": str(translated_dir),
                    "translated_embed_path": str(translated_embed_path),
                    "reference_embed_path": str(real_embed_path),
                    "feature_layer": feature_layer,
                    "n_reference": int(real_feats.shape[0]),
                    "n_translated": int(translated_feats.shape[0]),
                    "feature_dim": int(real_feats.shape[1]),
                    "deeplab_fd": deeplab_fd,
                }
            )

        if not summary_rows:
            raise ValueError("No DeepLab Fréchet stats were computed.")

        csv_out_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "noise_strength",
                    "cfg_weight",
                    "translated_dir",
                    "translated_embed_path",
                    "reference_embed_path",
                    "feature_layer",
                    "n_reference",
                    "n_translated",
                    "feature_dim",
                    "deeplab_fd",
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\nSaved DeepLab Fréchet stats CSV to {csv_out_path}")
