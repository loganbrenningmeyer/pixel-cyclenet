import argparse
import csv
import html
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DEFAULT_MODELS = ["all_real", "oem_only", "oem_only_seg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create CLIP PCA and t-SNE projection plots for every translate_sweep "
            "candidate directory under the requested model folders."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root translate_sweep directory containing model subdirectories.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model subdirectories to process under --root.",
    )
    parser.add_argument(
        "--clip-model-path",
        type=str,
        default=None,
        help=(
            "Override CLIP model path. If omitted, the script tries to read "
            "eval.clip_model_path from each model's config.yaml and falls back "
            "to openai/clip-vit-base-patch32."
        ),
    )
    parser.add_argument(
        "--clip-local-files-only",
        action="store_true",
        help="Force CLIP loading with local_files_only=True.",
    )
    parser.add_argument(
        "--clip-feature-layer",
        type=str,
        default=None,
        help="Override CLIP feature layer: projected or pooler.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=None,
        help="Override CLIP image batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for t-SNE.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Requested t-SNE perplexity. It will be clipped to a valid value.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing projection CSV/SVG files.",
    )
    return parser.parse_args()


def cfg_select(config: DictConfig | None, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    value = OmegaConf.select(config, key)
    return default if value is None else value


def load_model_config(model_dir: Path) -> DictConfig | None:
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        return None
    return OmegaConf.load(config_path)


def projection_config(model_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = load_model_config(model_dir)
    clip_model_path = (
        args.clip_model_path
        or cfg_select(config, "eval.clip_model_path", None)
        or "openai/clip-vit-base-patch32"
    )
    local_files_only = bool(
        args.clip_local_files_only
        or cfg_select(config, "eval.clip_local_files_only", False)
    )
    feature_layer = (
        args.clip_feature_layer
        or cfg_select(config, "eval.clip_feature_layer", "projected")
    )
    batch_size = int(
        args.clip_batch_size
        or cfg_select(config, "eval.clip_batch_size", 64)
    )
    return {
        "clip_model_path": str(clip_model_path),
        "clip_local_files_only": local_files_only,
        "clip_feature_layer": str(feature_layer),
        "clip_batch_size": batch_size,
    }


def gather_image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    return paths


def normalize_paths(paths: list[Path]) -> list[str]:
    return [str(p) for p in paths]


def build_clip_embedder(
    model_path: str,
    device: torch.device,
    batch_size: int,
    local_files_only: bool,
    feature_layer: str,
):
    model = CLIPModel.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        model_path,
        local_files_only=local_files_only,
    )
    model.eval()

    reported_feature_path = False

    def get_clip_embeddings(img_paths: list[str]) -> np.ndarray:
        nonlocal reported_feature_path
        feats_all = []

        for start in tqdm(
            range(0, len(img_paths), batch_size),
            desc="CLIP",
            unit="batch",
            disable=len(img_paths) <= batch_size,
        ):
            batch_paths = img_paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))

            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                if feature_layer.lower() == "pooler":
                    if not hasattr(model, "vision_model"):
                        raise ValueError("clip_feature_layer='pooler' requires model.vision_model.")
                    vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
                    feats = vision_out.pooler_output
                    feature_path = "vision_model.pooler_output"
                elif feature_layer.lower() == "projected":
                    feats = model.get_image_features(**inputs)
                    feature_path = "tensor"
                else:
                    raise ValueError("clip_feature_layer must be 'projected' or 'pooler'.")

            if not isinstance(feats, torch.Tensor):
                if hasattr(feats, "image_embeds"):
                    feats = feats.image_embeds
                    feature_path = "image_embeds"
                elif hasattr(feats, "pooler_output"):
                    feats = feats.pooler_output
                    if hasattr(model, "visual_projection"):
                        in_features = getattr(model.visual_projection, "in_features", None)
                        if in_features is None or feats.shape[-1] == in_features:
                            feats = model.visual_projection(feats)
                            feature_path = "pooler_output + visual_projection"
                        else:
                            feature_path = (
                                "pooler_output "
                                f"(projection skipped: {feats.shape[-1]} != {in_features})"
                            )
                    else:
                        feature_path = "pooler_output"
                else:
                    raise TypeError(
                        "CLIP image feature call returned unsupported output type: "
                        f"{type(feats).__name__}"
                    )

            if not reported_feature_path:
                print(f"CLIP feature path: {feature_path}, dim={feats.shape[-1]}")
                reported_feature_path = True

            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_all.append(feats.cpu().numpy())

        return np.concatenate(feats_all, axis=0)

    return get_clip_embeddings


def pca_project(feats: np.ndarray) -> np.ndarray:
    if feats.shape[0] < 2:
        return np.zeros((feats.shape[0], 2), dtype=np.float64)

    centered = feats.astype(np.float64) - feats.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:2].T
    coords = centered @ comps

    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1))], axis=1)
    return coords[:, :2]


def tsne_project(feats: np.ndarray, random_state: int, requested_perplexity: float) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except Exception as exc:
        raise RuntimeError(
            "t-SNE requires scikit-learn. Install it in the environment with "
            "`pip install scikit-learn` or use the updated project dependency list."
        ) from exc

    n = feats.shape[0]
    if n < 3:
        return np.zeros((n, 2), dtype=np.float64)

    max_perplexity = max(1.0, float(n - 1) / 3.0)
    perplexity = min(float(requested_perplexity), max_perplexity)

    reducer = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return reducer.fit_transform(feats)


def write_projection_csv(coords: np.ndarray, labels: list[str], paths: list[Path], out_path: Path):
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "x", "y", "path"])
        writer.writeheader()
        for label, xy, path in zip(labels, coords, paths):
            writer.writerow(
                {
                    "label": label,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "path": str(path),
                }
            )


def write_projection_svg(
    coords: np.ndarray,
    labels: list[str],
    paths: list[Path],
    out_path: Path,
    title: str,
):
    width = 900
    height = 700
    pad = 50
    colors = {
        "source_sim": "#6b7280",
        "translated_sim": "#2563eb",
        "real": "#dc2626",
    }

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    xy = (coords - mins) / span
    px = pad + xy[:, 0] * (width - 2 * pad)
    py = height - pad - xy[:, 1] * (height - 2 * pad)

    legend = [
        ("source_sim", "Source sim"),
        ("translated_sim", "Translated sim"),
        ("real", "Real"),
    ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="50" y="34" font-family="Arial, sans-serif" font-size="20" fill="#111827">{html.escape(title)}</text>',
    ]

    lx = width - 210
    ly = 30
    for i, (label, text) in enumerate(legend):
        y = ly + i * 24
        parts.append(f'<circle cx="{lx}" cy="{y}" r="6" fill="{colors[label]}" opacity="0.85"/>')
        parts.append(
            f'<text x="{lx + 14}" y="{y + 5}" font-family="Arial, sans-serif" font-size="14" fill="#111827">{text}</text>'
        )

    for x, y, label, path in zip(px, py, labels, paths):
        title_text = html.escape(f"{label}: {path}")
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{colors[label]}" opacity="0.62">'
            f"<title>{title_text}</title>"
            "</circle>"
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def projection_outputs_exist(candidate_dir: Path) -> bool:
    required = [
        candidate_dir / "clip_pca.csv",
        candidate_dir / "clip_pca.svg",
        candidate_dir / "clip_tsne.csv",
        candidate_dir / "clip_tsne.svg",
    ]
    return all(path.exists() for path in required)


def write_projection_error(candidate_dir: Path, name: str, exc: Exception):
    out_path = candidate_dir / f"clip_{name}_error.txt"
    out_path.write_text(f"{type(exc).__name__}: {exc}\n")


def run_candidate_projections(
    candidate_dir: Path,
    source_paths: list[Path],
    source_feats: np.ndarray,
    real_paths: list[Path],
    real_feats: np.ndarray,
    get_clip_embeddings,
    random_state: int,
    tsne_perplexity: float,
):
    fake_paths = gather_image_paths(candidate_dir)
    if not fake_paths:
        raise RuntimeError(f"No candidate images found under {candidate_dir}")

    fake_feats = get_clip_embeddings(normalize_paths(fake_paths))

    feats = np.concatenate([source_feats, fake_feats, real_feats], axis=0)
    labels = (
        ["source_sim"] * len(source_feats)
        + ["translated_sim"] * len(fake_feats)
        + ["real"] * len(real_feats)
    )
    paths = source_paths + fake_paths + real_paths

    pca_coords = pca_project(feats)
    write_projection_csv(pca_coords, labels, paths, candidate_dir / "clip_pca.csv")
    write_projection_svg(pca_coords, labels, paths, candidate_dir / "clip_pca.svg", "CLIP PCA")

    tsne_coords = tsne_project(feats, random_state=random_state, requested_perplexity=tsne_perplexity)
    write_projection_csv(tsne_coords, labels, paths, candidate_dir / "clip_tsne.csv")
    write_projection_svg(tsne_coords, labels, paths, candidate_dir / "clip_tsne.svg", "CLIP t-SNE")


def resolve_device(arg: str | None) -> torch.device:
    if arg is not None:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Translate sweep root does not exist: {root}")

    device = resolve_device(args.device)

    for model_name in args.models:
        model_dir = root / model_name
        if not model_dir.exists():
            raise RuntimeError(f"Model directory does not exist: {model_dir}")

        config = projection_config(model_dir, args)
        print(f"[{model_name}] using CLIP model: {config['clip_model_path']}")

        source_dir = model_dir / "reference" / "source"
        real_dir = model_dir / "reference" / "real"
        candidates_root = model_dir / "candidates"

        source_paths = gather_image_paths(source_dir)
        real_paths = gather_image_paths(real_dir)
        candidate_dirs = sorted(path for path in candidates_root.iterdir() if path.is_dir())

        if not source_paths:
            raise RuntimeError(f"No source reference images found under {source_dir}")
        if not real_paths:
            raise RuntimeError(f"No real reference images found under {real_dir}")
        if not candidate_dirs:
            raise RuntimeError(f"No candidate directories found under {candidates_root}")

        get_clip_embeddings = build_clip_embedder(
            model_path=config["clip_model_path"],
            device=device,
            batch_size=config["clip_batch_size"],
            local_files_only=config["clip_local_files_only"],
            feature_layer=config["clip_feature_layer"],
        )

        print(f"[{model_name}] embedding source references ({len(source_paths)} images)")
        source_feats = get_clip_embeddings(normalize_paths(source_paths))
        print(f"[{model_name}] embedding real references ({len(real_paths)} images)")
        real_feats = get_clip_embeddings(normalize_paths(real_paths))

        for candidate_dir in tqdm(candidate_dirs, desc=f"{model_name} candidates", unit="cand"):
            if not args.force and projection_outputs_exist(candidate_dir):
                continue

            for error_name in ["pca", "tsne"]:
                error_path = candidate_dir / f"clip_{error_name}_error.txt"
                if error_path.exists() and args.force:
                    error_path.unlink()

            try:
                run_candidate_projections(
                    candidate_dir=candidate_dir,
                    source_paths=source_paths,
                    source_feats=source_feats,
                    real_paths=real_paths,
                    real_feats=real_feats,
                    get_clip_embeddings=get_clip_embeddings,
                    random_state=args.random_state,
                    tsne_perplexity=args.tsne_perplexity,
                )
            except Exception as exc:
                print(f"[warn] {candidate_dir.name}: {type(exc).__name__}: {exc}")
                if not (candidate_dir / "clip_pca.csv").exists():
                    write_projection_error(candidate_dir, "pca", exc)
                write_projection_error(candidate_dir, "tsne", exc)


if __name__ == "__main__":
    main()
