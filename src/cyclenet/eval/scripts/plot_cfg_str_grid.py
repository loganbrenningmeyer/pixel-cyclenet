import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from cyclenet.data import TranslateDataset
from cyclenet.diffusion import DiffusionSchedule, cyclenet_ddim_loop, cyclenet_ddpm_loop
from cyclenet.eval.plotting.image_grid import plot_image_grid
from cyclenet.models import ControlNet, CycleNet, UNet
from cyclenet.models.conditioning import DomainEmbedding


def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str):
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


def parse_prefixed_float(name: str, prefix: str) -> float:
    if not name.startswith(prefix):
        raise ValueError(f"Expected '{name}' to start with '{prefix}'.")
    return float(name[len(prefix) :])


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
    control = ControlNet(backbone, in_ch=3).to(device)

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_sample_indices(n_total: int, n_select: int | None, seed: int) -> list[int]:
    if n_total <= 0:
        return []
    if n_select is None or n_select <= 0 or n_select >= n_total:
        return list(range(n_total))

    rng = random.Random(seed)
    return sorted(rng.sample(range(n_total), k=n_select))


def tensor_to_numpy_image(img: torch.Tensor) -> np.ndarray:
    img = ((img.clamp(-1.0, 1.0) + 1.0) / 2.0).float().cpu()
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0)


def save_numpy_image(img: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)).save(out_path)


def load_numpy_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def sample_output_name(src_root: Path, filepath: str, sample_idx: int) -> str:
    rel_path = Path(filepath).resolve().relative_to(src_root.resolve())
    stem = "__".join(rel_path.with_suffix("").parts)
    safe_stem = stem.replace(" ", "_")
    return f"{sample_idx:03d}_{safe_stem}"


def discover_saved_grid_layout(
    translated_samples_dir: Path,
) -> tuple[list[str], list[float], list[float], list[list[np.ndarray]]]:
    if not translated_samples_dir.exists():
        raise FileNotFoundError(f"translated_samples_dir does not exist: {translated_samples_dir}")

    sample_dirs = sorted(path for path in translated_samples_dir.iterdir() if path.is_dir())
    if not sample_dirs:
        raise ValueError(f"No sample directories found under {translated_samples_dir}")

    sample_names = [path.name for path in sample_dirs]
    all_strengths: set[float] = set()
    all_cfgs: set[float] = set()

    for sample_dir in sample_dirs:
        strength_dirs = [path for path in sample_dir.iterdir() if path.is_dir() and path.name.startswith("strength-")]
        if not strength_dirs:
            raise ValueError(f"No strength directories found under {sample_dir}")
        for strength_dir in strength_dirs:
            all_strengths.add(parse_prefixed_float(strength_dir.name, "strength-"))
            cfg_dirs = [path for path in strength_dir.iterdir() if path.is_dir() and path.name.startswith("cfg-")]
            if not cfg_dirs:
                raise ValueError(f"No cfg directories found under {strength_dir}")
            for cfg_dir in cfg_dirs:
                all_cfgs.add(parse_prefixed_float(cfg_dir.name, "cfg-"))

    noise_strengths = sorted(all_strengths)
    cfg_weights = sorted(all_cfgs)

    sample_grids: list[list[np.ndarray]] = []
    for sample_dir in sample_dirs:
        images: list[np.ndarray] = []
        for noise_strength in noise_strengths:
            strength_dir = sample_dir / f"strength-{noise_strength:.2f}"
            if not strength_dir.exists():
                raise FileNotFoundError(f"Missing strength directory for sample {sample_dir.name}: {strength_dir}")
            for cfg_weight in cfg_weights:
                cfg_dir = strength_dir / f"cfg-{cfg_weight:.1f}"
                img_path = cfg_dir / "img.png"
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing saved translated image: {img_path}")
                images.append(load_numpy_image(img_path))
        sample_grids.append(images)

    return sample_names, noise_strengths, cfg_weights, sample_grids


def load_saved_source_images(
    sample_names: list[str],
    source_samples_dir: Path | None,
) -> list[np.ndarray | None]:
    if source_samples_dir is None or not source_samples_dir.exists():
        return [None for _ in sample_names]

    source_images: list[np.ndarray | None] = []
    for sample_name in sample_names:
        source_path = source_samples_dir / f"{sample_name}.png"
        source_images.append(load_numpy_image(source_path) if source_path.exists() else None)
    return source_images


def translate_batch(
    model: CycleNet,
    x_src: torch.Tensor,
    device: torch.device,
    sched: DiffusionSchedule,
    src_domain_idx: int,
    sampler: str,
    cfg_weight: float,
    noise_strength: float,
    ddim_steps: int,
    eta: float,
) -> torch.Tensor:
    bsz = x_src.shape[0]
    x_src = x_src.to(device, non_blocking=True)
    x_src_ctrl = ((x_src + 1.0) / 2.0).clamp(0.0, 1.0)

    src_idx = torch.full((bsz,), src_domain_idx, device=device, dtype=torch.long)
    tgt_idx = torch.full((bsz,), 1 - src_domain_idx, device=device, dtype=torch.long)

    with autocast(device_type=device.type, enabled=device.type == "cuda"):
        if sampler.lower() == "ddpm":
            samples, _ = cyclenet_ddpm_loop(
                model=model,
                x_src=x_src,
                src_idx=src_idx,
                tgt_idx=tgt_idx,
                c_img=x_src_ctrl,
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
                c_img=x_src_ctrl,
                sched=sched,
                w=cfg_weight,
                strength=noise_strength,
                num_steps=ddim_steps,
                eta=eta,
            )
        else:
            raise ValueError("sampling.sampler must be 'ddpm' or 'ddim'.")

    return samples


def main():
    # -------------------------
    # Parse args / load + save config 
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    out_dir = Path(config.data.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    # -------------------------
    # [Saved models]: Load CycleNet config / UNet backbone config
    # -------------------------
    cyclenet_train_dir = Path(config.run.run_dir) / "training"
    cyclenet_config = load_config(cyclenet_train_dir / "config.yaml")

    unet_train_dir = Path(cyclenet_config.run.unet_ckpt).parent.parent
    unet_config = load_config(unet_train_dir / "config.yaml")

    seed = int(cfg_select(config, "run.seed", 42))
    translation_seed = int(cfg_select(config, "plotting.translation_seed", seed))
    set_seed(seed)

    grids_dir = out_dir / "grids"
    sources_dir = out_dir / "source_samples"
    translated_dir = out_dir / "translated_samples"
    grids_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    translated_dir.mkdir(parents=True, exist_ok=True)
    rerun_from_saved = bool(cfg_select(config, "plotting.rerun_from_saved", False))

    if rerun_from_saved:
        translated_samples_dir_value = cfg_select(config, "plotting.translated_samples_dir", None)
        translated_samples_dir = (
            Path(translated_samples_dir_value) if translated_samples_dir_value is not None else translated_dir
        )
        source_samples_dir_value = cfg_select(config, "plotting.source_samples_dir", None)
        source_samples_dir = (
            Path(source_samples_dir_value)
            if source_samples_dir_value is not None
            else translated_samples_dir.parent / "source_samples"
        )

        sample_names, noise_strengths, cfg_weights, sample_grids = discover_saved_grid_layout(
            translated_samples_dir
        )
        source_images = load_saved_source_images(sample_names, source_samples_dir)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        src_root = Path(config.data.src_dir)
        rgb_parent_dirs = as_parent_dir_set(cfg_select(config, "data.rgb_parent_dirs", None))

        dataset = TranslateDataset(
            src_dir=str(src_root),
            rgb_parent_dirs=rgb_parent_dirs,
            image_size=int(config.data.image_size),
        )
        if len(dataset) == 0:
            raise ValueError(f"No source images found under {src_root}.")

        sample_indices = select_sample_indices(
            n_total=len(dataset),
            n_select=cfg_select(config, "plotting.num_samples", None),
            seed=seed,
        )
        if not sample_indices:
            raise ValueError("No samples selected for plotting.")

        selected_tensors: list[torch.Tensor] = []
        selected_paths: list[str] = []
        for idx in sample_indices:
            x_src, filepath = dataset[idx]
            selected_tensors.append(x_src)
            selected_paths.append(filepath)
        sample_names = [
            sample_output_name(src_root, filepath, sample_idx)
            for sample_idx, filepath in enumerate(selected_paths)
        ]

        for sample_idx, x_src in enumerate(selected_tensors):
            save_numpy_image(
                tensor_to_numpy_image(x_src),
                sources_dir / f"{sample_names[sample_idx]}.png",
            )

        model = build_model(cyclenet_config, unet_config, device)
        model_key = str(cfg_select(config, "run.model_key", "ema_model"))
        ckpt_path = cyclenet_train_dir / "checkpoints" / str(config.run.ckpt_name)
        load_checkpoint(model, ckpt_path, model_key)
        model.eval()

        sched = build_schedule(unet_config, device)

        sampler = str(cfg_select(config, "sampling.sampler", "ddim"))
        batch_size = int(cfg_select(config, "sampling.batch_size", len(selected_tensors)))
        ddim_steps = int(cfg_select(config, "sampling.ddim_steps", 100))
        eta = float(cfg_select(config, "sampling.eta", 0.0))
        cfg_weights = [float(v) for v in list(config.sampling.cfg_weights)]
        noise_strengths = [float(v) for v in list(config.sampling.noise_strengths)]
        src_domain_idx = int(config.data.src_idx)

        sample_grids = [[] for _ in selected_tensors]
        source_images = [tensor_to_numpy_image(x_src) for x_src in selected_tensors]
        setting_pairs = [
            (noise_strength, cfg_weight)
            for noise_strength in noise_strengths
            for cfg_weight in cfg_weights
        ]

        torch.set_grad_enabled(False)
        with torch.inference_mode():
            for noise_strength, cfg_weight in tqdm(setting_pairs, desc="Sampling grids", unit="setting"):
                set_seed(translation_seed)
                cfg_str = f"cfg-{cfg_weight:.1f}"
                strength_str = f"strength-{noise_strength:.2f}"

                translated_images: list[np.ndarray] = []
                for start in range(0, len(selected_tensors), batch_size):
                    batch = torch.stack(selected_tensors[start : start + batch_size], dim=0)
                    samples = translate_batch(
                        model=model,
                        x_src=batch,
                        device=device,
                        sched=sched,
                        src_domain_idx=src_domain_idx,
                        sampler=sampler,
                        cfg_weight=cfg_weight,
                        noise_strength=noise_strength,
                        ddim_steps=ddim_steps,
                        eta=eta,
                    )
                    translated_images.extend(tensor_to_numpy_image(img) for img in samples)

                for sample_idx, img in enumerate(translated_images):
                    sample_grids[sample_idx].append(img)
                    save_numpy_image(
                        img,
                        translated_dir / sample_names[sample_idx] / strength_str / cfg_str / "img.png",
                    )

    row_labels = [f"{strength:g}" for strength in noise_strengths]
    col_labels = [f"{cfg:g}" for cfg in cfg_weights]
    title_prefix = cfg_select(config, "plotting.title", None)
    plot_scale = float(cfg_select(config, "plotting.scale", 1.0))
    save_dpi = int(cfg_select(config, "plotting.dpi", 300))
    source_mode = str(cfg_select(config, "plotting.source_mode", "grid_column"))
    source_width_ratio = float(cfg_select(config, "plotting.source_width_ratio", 1.15))
    source_gap_ratio = float(cfg_select(config, "plotting.source_gap_ratio", 0.10))
    grid_wspace = float(cfg_select(config, "plotting.grid_wspace", 0.045))
    grid_hspace = float(cfg_select(config, "plotting.grid_hspace", 0.045))
    show_source_divider = bool(cfg_select(config, "plotting.show_source_divider", True))
    source_divider_color = str(cfg_select(config, "plotting.source_divider_color", "black"))
    source_divider_linewidth = float(cfg_select(config, "plotting.source_divider_linewidth", 0.8))
    source_divider_gap_fraction = float(cfg_select(config, "plotting.source_divider_gap_fraction", 0.5))
    source_label_position = str(cfg_select(config, "plotting.source_label_position", "top"))
    title_y = float(cfg_select(config, "plotting.title_y", 0.955))
    title_fontsize = float(cfg_select(config, "plotting.title_fontsize", 9.0))
    xlabel_y = float(cfg_select(config, "plotting.xlabel_y", 0.035))
    xlabel_fontsize = float(cfg_select(config, "plotting.xlabel_fontsize", 11.0))
    row_label_side = str(cfg_select(config, "plotting.row_label_side", "right"))
    row_label_right_pad = float(cfg_select(config, "plotting.row_label_right_pad", 0.01))
    row_label_fontsize = float(cfg_select(config, "plotting.row_label_fontsize", 11.0))
    col_label_fontsize = float(cfg_select(config, "plotting.col_label_fontsize", 11.0))
    ylabel_fontsize = float(cfg_select(config, "plotting.ylabel_fontsize", 11.0))
    ylabel_pad = float(cfg_select(config, "plotting.ylabel_pad", 0.08))
    ylabel_right_pad = float(cfg_select(config, "plotting.ylabel_right_pad", 0.07))
    ylabel_side = str(cfg_select(config, "plotting.ylabel_side", "right"))
    default_ylabel_rotation = 270.0 if ylabel_side.lower() == "right" else 90.0
    ylabel_rotation = float(cfg_select(config, "plotting.ylabel_rotation", default_ylabel_rotation))
    subplot_top_with_title = float(cfg_select(config, "plotting.subplot_top_with_title", 0.93))
    subplot_bottom_with_xlabels = float(cfg_select(config, "plotting.subplot_bottom_with_xlabels", 0.10))

    for sample_idx, images in enumerate(sample_grids):
        sample_name = sample_names[sample_idx]

        plot_image_grid(
            images=images,
            n_cols=len(cfg_weights),
            row_labels=row_labels,
            col_labels=col_labels,
            title=title_prefix,
            source_image=source_images[sample_idx],
            source_label="Source (Sim)",
            xlabel="CFG weight ($w$)",
            ylabel="Noise strength ($\\gamma$)",
            save_path=grids_dir / f"{sample_name}.pdf",
            dpi=save_dpi,
            scale=plot_scale,
            source_mode=source_mode,
            source_width_ratio=source_width_ratio,
            source_gap_ratio=source_gap_ratio,
            grid_wspace=grid_wspace,
            grid_hspace=grid_hspace,
            show_source_divider=show_source_divider,
            source_divider_color=source_divider_color,
            source_divider_linewidth=source_divider_linewidth,
            source_divider_gap_fraction=source_divider_gap_fraction,
            source_label_position=source_label_position,
            row_label_side=row_label_side,
            row_label_fontsize=row_label_fontsize,
            col_label_fontsize=col_label_fontsize,
            title_span="full",
            title_fontsize=title_fontsize,
            title_y=title_y,
            xlabel_fontsize=xlabel_fontsize,
            xlabel_y=xlabel_y,
            row_label_right_pad=row_label_right_pad,
            ylabel_fontsize=ylabel_fontsize,
            ylabel_pad=ylabel_pad,
            ylabel_right_pad=ylabel_right_pad,
            ylabel_side=ylabel_side,
            ylabel_rotation=ylabel_rotation,
            ylabel_multiline=False,
            subplot_top_with_title=subplot_top_with_title,
            subplot_bottom_with_xlabels=subplot_bottom_with_xlabels,
        )

    print(f"Saved {len(sample_grids)} grids to {grids_dir}")


if __name__ == "__main__":
    main()
