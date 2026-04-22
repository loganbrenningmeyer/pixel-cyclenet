import csv
from pathlib import Path
import numpy as np

from cyclenet.eval.frechet_dist import frechet_distance


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


def main() -> None:
    # Reference cache directory produced by `project_translated.py` for CLIP embeddings.
    # This directory should contain `real_embed.npy` and optionally `sim_embed.npy`.
    reference_cache_dir = Path("/develop/code/eval/cyclenet/remote_sensing/project_translated/reference_cache/clip")

    # Single `step-{step}` directory whose `strength-{strength}/cfg-{cfg}` subdirectories
    # contain cached `clip_translated_embed.npy` arrays from `project_translated.py`.
    step_dir = Path("/develop/code/eval/cyclenet/remote_sensing/project_translated/oem_only/ema/step-2500")

    # Filename of the cached translated CLIP embeddings within each candidate directory.
    translated_embed_filename = "clip_translated_embed.npy"

    # CSV path where the aggregated CLIP-Fréchet stats for this step will be saved.
    csv_out_path = step_dir / "clip_fid_stats.csv"

    real_feats = load_embedding_array(reference_cache_dir / "real_embed.npy")
    summary_rows: list[dict[str, object]] = []

    for step, noise_strength, cfg_weight, translated_dir in iter_candidate_dirs(step_dir):
        translated_feats = load_embedding_array(translated_dir / translated_embed_filename)
        clip_fid = frechet_distance(translated_feats, real_feats)

        print(
            f"step-{step} / strength-{noise_strength:.1f} / cfg-{cfg_weight:.1f}".center(50, "=")
        )
        print(f"[ CLIP-FID ]: {clip_fid:.6f}")

        summary_rows.append(
            {
                "step": step,
                "noise_strength": noise_strength,
                "cfg_weight": cfg_weight,
                "translated_dir": str(translated_dir),
                "translated_embed_path": str(translated_dir / translated_embed_filename),
                "reference_embed_path": str(reference_cache_dir / "real_embed.npy"),
                "n_reference": int(real_feats.shape[0]),
                "n_translated": int(translated_feats.shape[0]),
                "feature_dim": int(real_feats.shape[1]),
                "clip_fid": clip_fid,
            }
        )

    if not summary_rows:
        raise ValueError("No CLIP-FID stats were computed.")

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
                "n_reference",
                "n_translated",
                "feature_dim",
                "clip_fid",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved CLIP-FID stats CSV to {csv_out_path}")


if __name__ == "__main__":
    main()
