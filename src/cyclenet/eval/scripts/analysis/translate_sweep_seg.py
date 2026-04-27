import argparse
import csv
import html
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from cyclenet.data import TranslateSegDataset
from cyclenet.diffusion import DiffusionSchedule, build_seg_condition, cyclenet_ddim_loop, cyclenet_ddpm_loop
from cyclenet.models import ControlNet, CycleNet, UNet
from cyclenet.models.conditioning import DomainEmbedding


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TORCH_FIDELITY_INCEPTION = "weights-inception-2015-12-05-6726825d.pth"


def ddp_setup() -> tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size,
        )

        return True, rank, local_rank, world_size

    return False, 0, 0, 1


def ddp_cleanup(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def barrier(is_ddp: bool):
    if is_ddp:
        dist.barrier()


def load_config(config_path: str | Path) -> DictConfig:
    return OmegaConf.load(config_path)


def cfg_select(config: DictConfig, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, key)
    return default if value is None else value


def as_parent_dir_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(v) for v in value}


def checkpoint_name(value: int | str) -> str:
    if isinstance(value, int):
        return f"step-{value}.ckpt"

    value = str(value)
    if value.endswith(".ckpt"):
        return value
    if value.startswith("step-"):
        return f"{value}.ckpt"
    if value.isdigit():
        return f"step-{value}.ckpt"
    return value


def required_sweep_values(config: DictConfig, key: str) -> list[Any]:
    values = OmegaConf.select(config, key)
    if values is None:
        raise ValueError(f"Missing required sweep config value: {key}")
    if isinstance(values, (str, int, float)):
        return [values]

    values = list(values)
    if not values:
        raise ValueError(f"Required sweep config value is empty: {key}")
    return values


def parse_ignore_pairs(config: DictConfig, key: str = "sweep.ignore_pairs") -> set[tuple[float, float]]:
    value = OmegaConf.select(config, key)
    if value is None:
        return set()

    pairs = []
    for item in list(value):
        if isinstance(item, str):
            parts = [p.strip() for p in item.split(",")]
            if len(parts) != 2:
                raise ValueError(f"Ignore pair string must have two comma-separated values: {item}")
            noise_strength, cfg_weight = float(parts[0]), float(parts[1])
        elif OmegaConf.is_list(item) or isinstance(item, (list, tuple)):
            if len(item) != 2:
                raise ValueError(f"Ignore pair sequence must have length 2: {item}")
            noise_strength, cfg_weight = float(item[0]), float(item[1])
        else:
            noise_strength = OmegaConf.select(item, "noise_strength")
            cfg_weight = OmegaConf.select(item, "cfg_weight")
            if noise_strength is None or cfg_weight is None:
                raise ValueError(
                    "Ignore pair entries must be [noise_strength, cfg_weight], "
                    "'noise_strength,cfg_weight', or {noise_strength: ..., cfg_weight: ...}."
                )
            noise_strength, cfg_weight = float(noise_strength), float(cfg_weight)

        pairs.append((noise_strength, cfg_weight))

    return set(pairs)


def gather_image_paths(
    root: str | Path,
    rgb_parent_dirs: set[str] | None = None,
    exts: set[str] = IMAGE_EXTS,
) -> list[Path]:
    root = Path(root)
    paths = []

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in exts:
            continue
        if rgb_parent_dirs is not None and path.parent.name not in rgb_parent_dirs:
            continue
        paths.append(path)

    return paths


def select_indices(n: int, k: int | None, seed: int) -> list[int]:
    if n <= 0:
        return []
    if k is None or k <= 0 or k >= n:
        return list(range(n))

    rng = random.Random(seed)
    return sorted(rng.sample(range(n), k))


def select_paths(paths: list[Path], k: int | None, seed: int) -> list[Path]:
    indices = select_indices(len(paths), k, seed)
    return [paths[i] for i in indices]


def resize_filter() -> Any:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_resized_reference_images(
    paths: list[Path],
    out_dir: Path,
    image_size: int,
) -> list[Path]:
    reset_dir(out_dir)
    saved_paths = []

    for i, path in enumerate(tqdm(paths, desc=f"Writing {out_dir.name}", unit="img")):
        out_path = out_dir / f"{i:06d}.png"
        with Image.open(path) as img:
            img = img.convert("RGB").resize((image_size, image_size), resize_filter())
            img.save(out_path)
        saved_paths.append(out_path)

    return saved_paths


def save_source_references(
    dataset: TranslateSegDataset,
    indices: list[int],
    src_root: Path,
    out_dir: Path,
) -> list[Path]:
    reset_dir(out_dir)
    saved_paths = []

    for idx in tqdm(indices, desc=f"Writing {out_dir.name}", unit="img"):
        x_src, _seg_src, filepath = dataset[idx]
        rel_path = Path(filepath).resolve().relative_to(src_root.resolve())
        out_path = (out_dir / rel_path).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(((x_src.clamp(-1, 1) + 1.0) / 2.0).float(), out_path)
        saved_paths.append(out_path)

    return saved_paths


def expected_source_reference_paths(
    dataset: TranslateSegDataset,
    indices: list[int],
    src_root: Path,
    out_dir: Path,
) -> list[Path]:
    paths = []
    for idx in indices:
        rel_path = Path(dataset.samples[idx][0]).resolve().relative_to(src_root.resolve())
        paths.append((out_dir / rel_path).with_suffix(".png"))
    return paths


def write_manifest(rows: list[dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def reference_paths_exist(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    return all(Path(row["reference_path"]).exists() for row in rows)


def load_reference_manifests(
    source_manifest_path: Path,
    real_manifest_path: Path,
) -> tuple[list[int], list[Path], list[Path]]:
    source_rows = read_manifest(source_manifest_path)
    real_rows = read_manifest(real_manifest_path)

    source_indices = [int(row["sample_index"]) for row in source_rows]
    source_ref_paths = [Path(row["reference_path"]) for row in source_rows]
    real_ref_paths = [Path(row["reference_path"]) for row in real_rows]

    return source_indices, source_ref_paths, real_ref_paths


def seg_sample_paths(dataset: TranslateSegDataset, idx: int) -> tuple[Path, Path]:
    rgb_path, label_path = dataset.samples[idx]
    return Path(rgb_path), Path(label_path)


def build_model(cyclenet_config: DictConfig, unet_config: DictConfig, device: torch.device) -> CycleNet:
    backbone = UNet(
        in_ch=3,
        base_ch=unet_config.model.base_ch,
        t_dim=unet_config.model.t_dim,
        d_dim=unet_config.model.d_dim,
        ch_mults=unet_config.model.ch_mults,
        num_res_blocks=unet_config.model.num_res_blocks,
        enc_heads=unet_config.model.enc_heads,
        mid_heads=unet_config.model.mid_heads,
        res_dropout=unet_config.model.res_dropout,
        attn_dropout=unet_config.model.attn_dropout,
        ffn_dropout=unet_config.model.ffn_dropout,
    ).to(device)

    domain_emb = DomainEmbedding(d_dim=unet_config.model.d_dim).to(device)

    num_seg_classes = int(cyclenet_config.model.num_seg_classes)
    # -- RGB + seg or seg only
    use_rgb_condition = cyclenet_config.model.use_rgb_condition
    control_in_ch = num_seg_classes + 3 if use_rgb_condition else num_seg_classes

    control = ControlNet(backbone, in_ch=control_in_ch).to(device)

    return CycleNet(
        backbone=backbone,
        control=control,
        domain_emb=domain_emb,
        t_dim=unet_config.model.t_dim,
        d_dim=unet_config.model.d_dim,
    ).to(device)


def load_checkpoint(model: CycleNet, ckpt_path: Path, model_key: str):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if model_key not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain key '{model_key}'.")
    model.load_state_dict(ckpt[model_key], strict=True)


def build_schedule(unet_config: DictConfig, device: torch.device) -> DiffusionSchedule:
    return DiffusionSchedule(
        schedule=unet_config.diffusion.schedule,
        T=unet_config.diffusion.T,
        beta_start=unet_config.diffusion.beta_start,
        beta_end=unet_config.diffusion.beta_end,
        device=device,
        s=unet_config.diffusion.s,
    )


def normalize_paths(paths: list[Path]) -> list[str]:
    return [str(p) for p in paths]


def default_torch_home() -> Path:
    return Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))


def ensure_torch_fidelity_weights(
    weights_path: str | None,
    torch_home: str | None = None,
) -> Path | None:
    if torch_home is not None:
        os.environ["TORCH_HOME"] = str(torch_home)

    if weights_path is None:
        return None

    source = Path(weights_path)
    if not source.exists():
        raise FileNotFoundError(f"Configured Inception weights do not exist: {source}")

    cache_dir = default_torch_home() / "hub" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / TORCH_FIDELITY_INCEPTION

    if target.exists():
        return target

    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)

    return target


def build_clip_embedder(
    model_path: str,
    device: torch.device,
    batch_size: int = 64,
    local_files_only: bool = False,
    feature_layer: str = "projected",
) -> Any:
    from transformers import CLIPModel, CLIPProcessor

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


def compute_clip_metrics_from_feats(fake_feats: np.ndarray, real_feats: np.ndarray) -> dict[str, float]:
    real_centroid = real_feats.mean(axis=0)
    fake_centroid = fake_feats.mean(axis=0)
    real_centroid = real_centroid / np.linalg.norm(real_centroid)
    fake_centroid = fake_centroid / np.linalg.norm(fake_centroid)

    sims = fake_feats @ real_feats.T

    return {
        "real_clip_centroid_cosine": float(fake_centroid @ real_centroid),
        "real_clip_centroid_l2": float(np.linalg.norm(fake_centroid - real_centroid)),
        "real_clip_nearest_cosine_mean": float(sims.max(axis=1).mean()),
        "real_clip_frechet": clip_frechet_distance(fake_feats, real_feats),
        "real_clip_mmd_rbf": clip_mmd_rbf(fake_feats, real_feats),
    }


def compute_clip_metrics(
    fake_paths: list[Path],
    real_feats: np.ndarray,
    get_clip_embeddings_fn: Any,
) -> dict[str, float]:
    fake_feats = get_clip_embeddings_fn(normalize_paths(fake_paths))
    return compute_clip_metrics_from_feats(fake_feats, real_feats)


def covariance_matrix(feats: np.ndarray) -> np.ndarray:
    if feats.shape[0] < 2:
        return np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float64)
    return np.cov(feats, rowvar=False).astype(np.float64)


def symmetric_matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) * 0.5)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def clip_frechet_distance(fake_feats: np.ndarray, real_feats: np.ndarray) -> float:
    """
    Fréchet distance in normalized CLIP embedding space. This is FID-style, but
    uses CLIP features instead of Inception features.
    """
    fake = fake_feats.astype(np.float64)
    real = real_feats.astype(np.float64)
    mu_fake = fake.mean(axis=0)
    mu_real = real.mean(axis=0)
    cov_fake = covariance_matrix(fake)
    cov_real = covariance_matrix(real)

    cov_real_sqrt = symmetric_matrix_sqrt(cov_real)
    middle = cov_real_sqrt @ cov_fake @ cov_real_sqrt
    cov_mean = symmetric_matrix_sqrt(middle)

    diff = mu_fake - mu_real
    dist = diff @ diff + np.trace(cov_fake + cov_real - 2.0 * cov_mean)
    return float(max(dist, 0.0))


def median_pairwise_sqdist(feats: np.ndarray, max_samples: int = 1024) -> float:
    if feats.shape[0] > max_samples:
        feats = feats[:max_samples]
    sq = np.sum((feats[:, None, :] - feats[None, :, :]) ** 2, axis=-1)
    vals = sq[np.triu_indices_from(sq, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    return float(np.median(vals))


def rbf_kernel_mean(x: np.ndarray, y: np.ndarray, gamma: float, block: int = 512) -> float:
    total = 0.0
    count = 0
    for i in range(0, x.shape[0], block):
        xb = x[i : i + block]
        for j in range(0, y.shape[0], block):
            yb = y[j : j + block]
            sq = np.sum((xb[:, None, :] - yb[None, :, :]) ** 2, axis=-1)
            total += float(np.exp(-gamma * sq).sum())
            count += sq.size
    return total / max(count, 1)


def clip_mmd_rbf(fake_feats: np.ndarray, real_feats: np.ndarray) -> float:
    """
    Biased RBF-kernel MMD in CLIP feature space. Lower means the translated
    feature distribution is closer to the real feature distribution.
    """
    pooled = np.concatenate([fake_feats, real_feats], axis=0).astype(np.float64)
    sigma_sq = median_pairwise_sqdist(pooled)
    gamma = 1.0 / max(2.0 * sigma_sq, 1e-12)

    xx = rbf_kernel_mean(fake_feats, fake_feats, gamma)
    yy = rbf_kernel_mean(real_feats, real_feats, gamma)
    xy = rbf_kernel_mean(fake_feats, real_feats, gamma)
    return float(max(xx + yy - 2.0 * xy, 0.0))


def write_clip_umap(
    source_feats: np.ndarray,
    fake_feats: np.ndarray,
    real_feats: np.ndarray,
    source_paths: list[Path],
    fake_paths: list[Path],
    real_paths: list[Path],
    out_dir: Path,
    random_state: int,
) -> dict[str, str]:
    try:
        import umap
    except Exception as exc:
        return {"clip_umap_error": f"{type(exc).__name__}: {exc}"}

    feats = np.concatenate([source_feats, fake_feats, real_feats], axis=0)
    labels = (
        ["source_sim"] * len(source_feats)
        + ["translated_sim"] * len(fake_feats)
        + ["real"] * len(real_feats)
    )
    paths = source_paths + fake_paths + real_paths

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    coords = reducer.fit_transform(feats)

    csv_path = out_dir / "clip_umap.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "x", "y", "path"])
        writer.writeheader()
        for label, xy, path in zip(labels, coords, paths):
            writer.writerow({
                "label": label,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "path": str(path),
            })

    svg_path = out_dir / "clip_umap.svg"
    write_umap_svg(coords, labels, paths, svg_path)

    return {
        "clip_umap_csv": str(csv_path),
        "clip_umap_svg": str(svg_path),
    }


def write_umap_svg(coords: np.ndarray, labels: list[str], paths: list[Path], out_path: Path):
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
        '<text x="50" y="34" font-family="Arial, sans-serif" font-size="20" fill="#111827">CLIP UMAP</text>',
    ]

    lx = width - 210
    ly = 30
    for i, (label, text) in enumerate(legend):
        y = ly + i * 24
        parts.append(f'<circle cx="{lx}" cy="{y}" r="6" fill="{colors[label]}" opacity="0.85"/>')
        parts.append(f'<text x="{lx + 14}" y="{y + 5}" font-family="Arial, sans-serif" font-size="14" fill="#111827">{text}</text>')

    for x, y, label, path in zip(px, py, labels, paths):
        title = html.escape(f"{label}: {path}")
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{colors[label]}" opacity="0.62">'
            f"<title>{title}</title>"
            "</circle>"
        )

    parts.append("</svg>")
    out_path.write_text("\n".join(parts))


def write_metrics(metrics_rows: list[dict[str, Any]], out_dir: Path):
    csv_path = out_dir / "metrics.csv"
    json_path = out_dir / "metrics.json"

    fieldnames = sorted({key for row in metrics_rows for key in row.keys()})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    with json_path.open("w") as f:
        json.dump(metrics_rows, f, indent=2)


def append_metric_error(row: dict[str, Any], name: str, exc: Exception):
    row[f"{name}_error"] = f"{type(exc).__name__}: {exc}"


def reduce_lpips_stats(stats: dict[str, float], device: torch.device, is_ddp: bool) -> dict[str, float]:
    tensor = torch.tensor(
        [
            stats.get("source_lpips_count", 0.0),
            stats.get("source_lpips_sum", 0.0),
            stats.get("source_lpips_sumsq", 0.0),
        ],
        device=device,
        dtype=torch.float64,
    )

    if is_ddp:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return {
        "source_lpips_count": float(tensor[0].item()),
        "source_lpips_sum": float(tensor[1].item()),
        "source_lpips_sumsq": float(tensor[2].item()),
    }


def lpips_metrics_from_stats(stats: dict[str, float]) -> dict[str, float]:
    count = stats.get("source_lpips_count", 0.0)
    if count <= 0:
        return {}

    mean = stats["source_lpips_sum"] / count
    var = max((stats["source_lpips_sumsq"] / count) - (mean * mean), 0.0)
    return {
        "source_lpips_mean": float(mean),
        "source_lpips_std": float(math.sqrt(var)),
    }


def maybe_compute_baseline(
    metrics: set[str],
    real_ref_dir: Path,
    source_ref_dir: Path,
    source_ref_paths: list[Path],
    num_real: int,
    real_feats: np.ndarray | None,
    get_clip_embeddings_fn: Any | None,
    fid_computer: Any | None,
    fid_init_error: Exception | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": "source_baseline",
        "comparison": "source_sim_vs_real",
        "checkpoint": "",
        "cfg_weight": "",
        "noise_strength": "",
        "num_fake": len(source_ref_paths),
        "num_real": num_real,
        "out_dir": str(source_ref_dir),
    }

    if "fid" in metrics and fid_computer is not None:
        try:
            row["real_fid"] = fid_computer.compute(real_ref_dir, source_ref_dir)
        except Exception as exc:
            row["real_fid"] = math.nan
            append_metric_error(row, "fid", exc)
    elif "fid" in metrics and fid_init_error is not None:
        row["real_fid"] = math.nan
        append_metric_error(row, "fid", fid_init_error)

    if "clip" in metrics and real_feats is not None and get_clip_embeddings_fn is not None:
        try:
            row.update(compute_clip_metrics(source_ref_paths, real_feats, get_clip_embeddings_fn))
        except Exception as exc:
            append_metric_error(row, "clip", exc)

    return row


def translate_candidate(
    model: CycleNet,
    dataloader: DataLoader,
    src_root: Path,
    out_dir: Path,
    device: torch.device,
    sched: DiffusionSchedule,
    src_domain_idx: int,
    use_rgb_condition: bool,
    sampler: str,
    cfg_weight: float,
    noise_strength: float,
    ddim_steps: int,
    eta: float,
    lpips_batch_fn: Any | None,
    show_progress: bool = True,
) -> tuple[list[Path], dict[str, float]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fake_paths = []
    lpips_values = []

    tgt_domain_idx = 1 - src_domain_idx
    torch.set_grad_enabled(False)

    with torch.inference_mode():
        for x_src, seg_src, filepaths in tqdm(
            dataloader,
            desc="Translating",
            unit="batch",
            disable=not show_progress,
        ):
            bsz = x_src.shape[0]
            x_src = x_src.to(device, non_blocking=True)
            seg_src = seg_src.to(device, non_blocking=True)
            c_img = build_seg_condition(x_src, seg_src, use_rgb_condition)

            src_idx = torch.full((bsz,), src_domain_idx, device=device, dtype=torch.long)
            tgt_idx = torch.full((bsz,), tgt_domain_idx, device=device, dtype=torch.long)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                if sampler.lower() == "ddpm":
                    samples, _ = cyclenet_ddpm_loop(
                        model=model,
                        x_src=x_src,
                        src_idx=src_idx,
                        tgt_idx=tgt_idx,
                        c_img=c_img,
                        sched=sched,
                        w=cfg_weight,
                        strength=noise_strength,
                    )
                elif sampler.lower() == "ddim":
                    samples, _ = cyclenet_ddim_loop(
                        model=model,
                        x_src=x_src,
                        src_idx=src_idx,
                        tgt_idx=tgt_idx,
                        c_img=c_img,
                        sched=sched,
                        w=cfg_weight,
                        strength=noise_strength,
                        num_steps=ddim_steps,
                        eta=eta,
                    )
                else:
                    raise ValueError("Sampler must be 'ddpm' or 'ddim'.")

            if lpips_batch_fn is not None:
                values = lpips_batch_fn(x_src.float(), samples.float())
                lpips_values.extend(values.tolist())

            for img, filepath in zip(samples, filepaths):
                rel_path = Path(filepath).resolve().relative_to(src_root.resolve())
                out_path = (out_dir / rel_path).with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(((img.clamp(-1, 1) + 1.0) / 2.0).float().cpu(), out_path)
                fake_paths.append(out_path)

    lpips_stats = {"source_lpips_count": 0.0, "source_lpips_sum": 0.0, "source_lpips_sumsq": 0.0}
    if lpips_values:
        arr = np.asarray(lpips_values, dtype=np.float64)
        lpips_stats = {
            "source_lpips_count": float(arr.size),
            "source_lpips_sum": float(arr.sum()),
            "source_lpips_sumsq": float(np.square(arr).sum()),
        }

    return fake_paths, lpips_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    is_ddp, rank, local_rank, world_size = ddp_setup()
    is_main = rank == 0

    run_dir = Path(config.run.run_dir)
    cyclenet_train_dir = run_dir / "training"
    cyclenet_config = load_config(cyclenet_train_dir / "config.yaml")

    unet_train_dir = Path(cyclenet_config.run.unet_ckpt).parent.parent
    unet_config = load_config(unet_train_dir / "config.yaml")

    seed = int(cfg_select(config, "eval.seed", cfg_select(config, "run.seed", 42)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    eval_out_dir = Path(config.eval.out_dir)
    if is_main:
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, eval_out_dir / "config.yaml")
    barrier(is_ddp)

    src_root = Path(config.data.src_dir)
    src_parent_dirs = as_parent_dir_set(cfg_select(config, "data.rgb_parent_dirs", None))
    if src_parent_dirs is None:
        raise ValueError("data.rgb_parent_dirs must be set for the segmentation sweep so RGB files can be matched to label masks.")
    real_root = Path(cfg_select(config, "eval.real_dir", cyclenet_config.data.tgt_dir))
    real_parent_dirs = as_parent_dir_set(cfg_select(config, "eval.real_rgb_parent_dirs", cfg_select(cyclenet_config, "data.rgb_parent_dirs", None)))
    image_size = int(config.data.image_size)
    num_seg_classes = int(cfg_select(config, "model.num_seg_classes", cyclenet_config.model.num_seg_classes))
    use_rgb_condition = cyclenet_config.model.use_rgb_condition
    label_parent_dir = str(cfg_select(config, "data.label_parent_dir", cfg_select(cyclenet_config, "data.label_parent_dir", "gt_ss_mask")))

    source_dataset = TranslateSegDataset(
        src_dir=str(src_root),
        image_size=image_size,
        num_classes=num_seg_classes,
        rgb_parent_dirs=src_parent_dirs,
        label_parent_dir=label_parent_dir,
    )
    if len(source_dataset) == 0:
        raise RuntimeError(f"No source images found under {src_root}.")

    ref_dir = eval_out_dir / "reference"
    source_ref_dir = ref_dir / "source"
    real_ref_dir = ref_dir / "real"
    source_manifest_path = eval_out_dir / "source_manifest.csv"
    real_manifest_path = eval_out_dir / "real_manifest.csv"

    if is_main:
        reuse_references = bool(cfg_select(config, "eval.reuse_references", True))
        refresh_references = bool(cfg_select(config, "eval.refresh_references", False))

        can_reuse = (
            reuse_references
            and not refresh_references
            and source_manifest_path.exists()
            and real_manifest_path.exists()
        )

        if can_reuse:
            source_rows = read_manifest(source_manifest_path)
            real_rows = read_manifest(real_manifest_path)
            can_reuse = reference_paths_exist(source_rows) and reference_paths_exist(real_rows)

        if can_reuse:
            print(f"Reusing reference manifests and images from {eval_out_dir}")
        else:
            source_size = cfg_select(config, "eval.source_sample_size", 256)
            source_indices = select_indices(len(source_dataset), int(source_size), seed)

            real_paths_all = gather_image_paths(real_root, real_parent_dirs)
            if not real_paths_all:
                raise RuntimeError(f"No real reference images found under {real_root}.")

            real_size = cfg_select(config, "eval.real_sample_size", source_size)
            real_paths = select_paths(real_paths_all, int(real_size), seed + 1)

            source_ref_paths = save_source_references(source_dataset, source_indices, src_root, source_ref_dir)
            real_ref_paths = save_resized_reference_images(real_paths, real_ref_dir, image_size)

            source_manifest_rows = []
            for idx, ref_path in zip(source_indices, source_ref_paths):
                src_path, label_path = seg_sample_paths(source_dataset, idx)
                source_manifest_rows.append({
                    "sample_index": idx,
                    "source_path": str(src_path),
                    "label_path": str(label_path),
                    "reference_path": str(ref_path),
                })
            write_manifest(source_manifest_rows, source_manifest_path)

            real_manifest_rows = [
                {"sample_index": i, "real_path": str(path), "reference_path": str(ref_path)}
                for i, (path, ref_path) in enumerate(zip(real_paths, real_ref_paths))
            ]
            write_manifest(real_manifest_rows, real_manifest_path)

    barrier(is_ddp)

    source_indices, source_ref_paths, real_ref_paths = load_reference_manifests(
        source_manifest_path,
        real_manifest_path,
    )

    metric_values = cfg_select(config, "eval.metrics", ["fid", "clip", "lpips"])
    if isinstance(metric_values, str):
        metric_values = [metric_values]
    metrics = {str(m).lower() for m in metric_values}
    metrics_rows: list[dict[str, Any]] = []

    fid_computer = None
    fid_init_error = None
    if is_main and "fid" in metrics:
        try:
            torch_home = cfg_select(config, "eval.torch_home", None)
            inception_weights_path = cfg_select(config, "eval.inception_weights_path", None)
            cached_path = ensure_torch_fidelity_weights(inception_weights_path, torch_home)
            if cached_path is not None:
                print(f"Using torch-fidelity Inception weights: {cached_path}")

            from cyclenet.eval.fid import FIDComputer

            fid_computer = FIDComputer(device=device)
        except Exception as exc:
            fid_init_error = exc
            print(f"[warn] disabling FID metrics: {type(exc).__name__}: {exc}")

    get_clip_embeddings_fn = None
    real_feats = None
    source_feats = None
    clip_init_error = None
    if is_main and "clip" in metrics:
        try:
            clip_model_path = str(cfg_select(config, "eval.clip_model_path", "openai/clip-vit-base-patch32"))
            clip_batch_size = int(cfg_select(config, "eval.clip_batch_size", 64))
            clip_local_files_only = bool(cfg_select(config, "eval.clip_local_files_only", False))
            clip_feature_layer = str(cfg_select(config, "eval.clip_feature_layer", "projected"))
            get_clip_embeddings_fn = build_clip_embedder(
                model_path=clip_model_path,
                device=device,
                batch_size=clip_batch_size,
                local_files_only=clip_local_files_only,
                feature_layer=clip_feature_layer,
            )
            real_feats = get_clip_embeddings_fn(normalize_paths(real_ref_paths))
            source_feats = get_clip_embeddings_fn(normalize_paths(source_ref_paths))
        except Exception as exc:
            clip_init_error = exc
            print(f"[warn] disabling CLIP metrics: {type(exc).__name__}: {exc}")

    lpips_batch_fn = None
    lpips_import_error = None
    if "lpips" in metrics:
        try:
            import lpips

            lpips_model = lpips.LPIPS(net="alex").to(device)
            lpips_model.eval()

            def lpips_batch_local(b1: torch.Tensor, b2: torch.Tensor) -> np.ndarray:
                with torch.no_grad():
                    d: torch.Tensor = lpips_model(b1.to(device), b2.to(device))
                return d.view(-1).cpu().numpy()

            lpips_batch_fn = lpips_batch_local
        except Exception as exc:
            lpips_import_error = exc
            if is_main:
                print(f"[warn] disabling LPIPS metrics: {type(exc).__name__}: {exc}")

    if is_main and bool(cfg_select(config, "eval.compute_source_baseline", True)):
        baseline = maybe_compute_baseline(
            metrics,
            real_ref_dir,
            source_ref_dir,
            source_ref_paths,
            len(real_ref_paths),
            real_feats,
            get_clip_embeddings_fn,
            fid_computer,
            fid_init_error,
        )
        if "clip" in metrics and clip_init_error is not None:
            append_metric_error(baseline, "clip", clip_init_error)
        metrics_rows.append(baseline)
        write_metrics(metrics_rows, eval_out_dir)

    rank_source_indices = source_indices[rank::world_size]
    source_subset = Subset(source_dataset, rank_source_indices)

    batch_size = int(config.sampling.batch_size)
    per_rank_batch_size = max(1, batch_size // world_size)
    dataloader = DataLoader(
        source_subset,
        batch_size=per_rank_batch_size,
        shuffle=False,
        num_workers=int(cfg_select(config, "eval.num_workers", 2)),
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(cyclenet_config, unet_config, device)
    model.eval()
    sched = build_schedule(unet_config, device)

    checkpoints = [checkpoint_name(v) for v in required_sweep_values(config, "sweep.checkpoints")]
    cfg_weights = [float(v) for v in required_sweep_values(config, "sweep.cfg_weights")]
    noise_strengths = [float(v) for v in required_sweep_values(config, "sweep.noise_strengths")]
    ignore_pairs = parse_ignore_pairs(config)
    model_key = str(cfg_select(config, "eval.model_key", "ema_model"))

    if is_main and ignore_pairs:
        ignored = ", ".join(f"(strength={s:g}, cfg={w:g})" for s, w in sorted(ignore_pairs))
        print(f"Skipping configured sweep pairs: {ignored}")

    sampler = str(config.sampling.sampler)
    ddim_steps = int(config.sampling.ddim_steps)
    eta = float(config.sampling.eta)
    src_domain_idx = int(config.data.src_idx)
    translation_seed = int(cfg_select(config, "eval.translation_seed", seed))

    for ckpt_name in checkpoints:
        ckpt_path = run_dir / "training" / "checkpoints" / ckpt_name
        if is_main:
            print(f"Loading {ckpt_path}")
        load_checkpoint(model, ckpt_path, model_key)
        model.eval()

        for noise_strength in noise_strengths:
            for cfg_weight in cfg_weights:
                if (noise_strength, cfg_weight) in ignore_pairs:
                    continue

                torch.manual_seed(translation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(translation_seed)

                combo_name = (
                    f"{Path(ckpt_name).stem}"
                    f"_strength-{noise_strength:.2f}"
                    f"_cfg-{cfg_weight:.1f}"
                )
                combo_dir = eval_out_dir / "candidates" / combo_name
                fake_dir = combo_dir / "fake"

                if is_main:
                    reset_dir(fake_dir)
                barrier(is_ddp)

                row: dict[str, Any] = {
                    "kind": "translated_candidate",
                    "comparison": "translated_sim_vs_real",
                    "checkpoint": ckpt_name,
                    "cfg_weight": cfg_weight,
                    "noise_strength": noise_strength,
                    "num_real": len(real_ref_paths),
                    "out_dir": str(combo_dir),
                }

                fake_paths, lpips_metrics = translate_candidate(
                    model=model,
                    dataloader=dataloader,
                    src_root=src_root,
                    out_dir=fake_dir,
                    device=device,
                    sched=sched,
                    src_domain_idx=src_domain_idx,
                    use_rgb_condition=use_rgb_condition,
                    sampler=sampler,
                    cfg_weight=cfg_weight,
                    noise_strength=noise_strength,
                    ddim_steps=ddim_steps,
                    eta=eta,
                    lpips_batch_fn=lpips_batch_fn,
                    show_progress=is_main,
                )
                lpips_stats = reduce_lpips_stats(lpips_metrics, device, is_ddp)
                barrier(is_ddp)

                if is_main:
                    fake_paths = gather_image_paths(fake_dir)
                    row["num_fake"] = len(fake_paths)
                    row.update(lpips_metrics_from_stats(lpips_stats))
                    if "lpips" in metrics and lpips_import_error is not None:
                        row["source_lpips_mean"] = math.nan
                        append_metric_error(row, "lpips", lpips_import_error)

                    if "fid" in metrics and fid_computer is not None:
                        try:
                            row["real_fid"] = fid_computer.compute(real_ref_dir, fake_dir)
                        except Exception as exc:
                            row["real_fid"] = math.nan
                            append_metric_error(row, "fid", exc)
                    elif "fid" in metrics and fid_init_error is not None:
                        row["real_fid"] = math.nan
                        append_metric_error(row, "fid", fid_init_error)

                    if "clip" in metrics and real_feats is not None and source_feats is not None and get_clip_embeddings_fn is not None:
                        try:
                            fake_feats = get_clip_embeddings_fn(normalize_paths(fake_paths))
                            row.update(compute_clip_metrics_from_feats(fake_feats, real_feats))

                            if bool(cfg_select(config, "eval.plot_clip_umap", True)):
                                umap_state = int(cfg_select(config, "eval.umap_random_state", seed))
                                row.update(write_clip_umap(
                                    source_feats=source_feats,
                                    fake_feats=fake_feats,
                                    real_feats=real_feats,
                                    source_paths=source_ref_paths,
                                    fake_paths=fake_paths,
                                    real_paths=real_ref_paths,
                                    out_dir=combo_dir,
                                    random_state=umap_state,
                                ))
                        except Exception as exc:
                            append_metric_error(row, "clip", exc)
                    elif "clip" in metrics and clip_init_error is not None:
                        append_metric_error(row, "clip", clip_init_error)

                    metrics_rows.append(row)
                    write_metrics(metrics_rows, eval_out_dir)

    barrier(is_ddp)
    if is_main:
        print(f"Wrote metrics to {eval_out_dir / 'metrics.csv'}")
    ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
