import os
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from cyclenet.data import TranslateDataset
from cyclenet.diffusion import DiffusionSchedule, cyclenet_ddim_loop, cyclenet_ddpm_loop
from cyclenet.models import CycleNet, UNet, ControlNet
from cyclenet.models.conditioning import DomainEmbedding


def ddp_setup():
    """
    Initializes torch.distributed if launched with torchrun.

    Returns: (is_ddp, rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=local_rank,
        )

        return True, rank, local_rank, world_size

    return False, 0, 0, 1


def ddp_cleanup(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)


def cfg_select(config: DictConfig, key: str, default=None):
    value = OmegaConf.select(config, key)
    return default if value is None else value


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


def required_sweep_values(config: DictConfig, key: str) -> list:
    values = OmegaConf.select(config, key)
    if values is None:
        raise ValueError(f"Missing required sweep config value: {key}")
    if isinstance(values, (str, int, float)):
        return [values]

    values = list(values)
    if not values:
        raise ValueError(f"Required sweep config value is empty: {key}")
    return values


def resolve_template_path(template: str, **kwargs) -> Path:
    return Path(str(template).format(**kwargs))


def combo_name(step: int | str, strength: float, cfg_weight: float) -> str:
    return f"step-{step}_strength-{strength}_cfg-{cfg_weight}"


def main():
    # -------------------------
    # [Translation Sweep]: Load config
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    translate_config = load_config(args.config)

    num_shards = int(cfg_select(translate_config, "data.num_shards", 1))
    shard_index = int(cfg_select(translate_config, "data.shard_index", 0))
    if num_shards < 1:
        raise ValueError(f"data.num_shards must be >= 1, got {num_shards}.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"data.shard_index must be in [0, {num_shards - 1}], got {shard_index}."
        )

    sweep_steps = required_sweep_values(translate_config, "sweep.steps")
    sweep_cfgs = [float(v) for v in required_sweep_values(translate_config, "sweep.cfg_weights")]
    sweep_strengths = [float(v) for v in required_sweep_values(translate_config, "sweep.noise_strengths")]

    out_dir_template = cfg_select(translate_config, "data.out_dir_template", None)
    base_out_dir_value = cfg_select(translate_config, "data.out_dir", None)
    if out_dir_template is None and base_out_dir_value is None:
        raise ValueError("Sweep mode requires either data.out_dir_template or data.out_dir.")

    # -------------------------
    # [Saved models]: Load CycleNet config / UNet backbone config
    # -------------------------
    cyclenet_train_dir = Path(translate_config.run.run_dir) / "training"
    cyclenet_config = load_config(cyclenet_train_dir / "config.yaml")

    unet_train_dir = Path(cyclenet_config.run.unet_ckpt).parent.parent
    unet_config = load_config(unet_train_dir / "config.yaml")

    # -------------------------
    # Initialize DDP
    # -------------------------
    is_ddp, rank, local_rank, world_size = ddp_setup()
    is_main = (rank == 0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    base_out_dir = Path(base_out_dir_value) if base_out_dir_value is not None else None
    if is_main and base_out_dir is not None:
        base_out_dir.mkdir(parents=True, exist_ok=True)
        config_name = "config.yaml"
        if num_shards > 1:
            config_name = f"config.shard-{shard_index}-of-{num_shards}.yaml"
        save_config(translate_config, base_out_dir / config_name)

    if is_ddp:
        dist.barrier()

    # -------------------------
    # Source Samples Dataset / DataLoader
    # -------------------------
    rgb_parent_dirs = (
        set(translate_config.data.rgb_parent_dirs)
        if translate_config.data.get("rgb_parent_dirs") is not None
        else None
    )

    dataset = TranslateDataset(
        src_dir=translate_config.data.src_dir,
        rgb_parent_dirs=rgb_parent_dirs,
        image_size=translate_config.data.image_size,
    )

    shard_indices = list(range(shard_index, len(dataset), num_shards))
    indices = shard_indices[rank::world_size]
    subset = Subset(dataset, indices)

    if is_main:
        print(
            f"Shard {shard_index + 1}/{num_shards}: "
            f"{len(shard_indices)} of {len(dataset)} images selected."
        )

    dataloader = DataLoader(
        subset,
        batch_size=max(1, translate_config.sampling.batch_size // world_size),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------
    # Initialize UNet / DomainEmbedding / ControlNet
    # -------------------------
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

    model = CycleNet(
        backbone=backbone,
        control=control,
        domain_emb=domain_emb,
        t_dim=unet_config.model.t_dim,
        d_dim=unet_config.model.d_dim,
    ).to(device)

    sched = DiffusionSchedule(
        schedule=unet_config.diffusion.schedule,
        T=unet_config.diffusion.T,
        beta_start=unet_config.diffusion.beta_start,
        beta_end=unet_config.diffusion.beta_end,
        device=device,
        s=unet_config.diffusion.s,
    )

    model.eval()
    torch.set_grad_enabled(False)

    src_domain_idx = translate_config.data.src_idx
    tgt_domain_idx = 1 - src_domain_idx

    sampler = translate_config.sampling.sampler
    num_steps = translate_config.sampling.ddim_steps
    eta = translate_config.sampling.eta

    for step_value in sweep_steps:
        ckpt_name = checkpoint_name(step_value)
        ckpt_path = cyclenet_train_dir / "checkpoints" / ckpt_name
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model_key = str(cfg_select(translate_config, "run.model_key", "ema_model"))
        if model_key not in ckpt:
            raise KeyError(f"Checkpoint {ckpt_path} does not contain key '{model_key}'.")
        model.load_state_dict(ckpt[model_key], strict=True)

        step_stem = Path(ckpt_name).stem

        for noise_strength in sweep_strengths:
            for cfg_weight in sweep_cfgs:
                combo = combo_name(step=step_stem.replace("step-", ""), strength=noise_strength, cfg_weight=cfg_weight)
                template_kwargs = {
                    "step": step_stem.replace("step-", ""),
                    "ckpt_name": ckpt_name,
                    "ckpt_stem": step_stem,
                    "strength": noise_strength,
                    "cfg": cfg_weight,
                }
                out_dir = (
                    resolve_template_path(str(out_dir_template), **template_kwargs)
                    if out_dir_template is not None
                    else base_out_dir / combo
                )

                if is_main:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    run_config = OmegaConf.create(OmegaConf.to_container(translate_config, resolve=False))
                    run_config.run.ckpt_name = ckpt_name
                    run_config.data.out_dir = str(out_dir)
                    run_config.sampling.noise_strength = noise_strength
                    run_config.sampling.cfg_weight = cfg_weight
                    config_name = "config.yaml"
                    if num_shards > 1:
                        config_name = f"config.shard-{shard_index}-of-{num_shards}.yaml"
                    save_config(run_config, out_dir / config_name)
                    print(f"[{combo}] -> {out_dir}")

                if is_ddp:
                    dist.barrier()

                with torch.inference_mode():
                    loader_iter = dataloader
                    if is_main:
                        loader_iter = tqdm(
                            dataloader,
                            desc=f"Translating {combo}",
                            unit="batch",
                        )

                    for x_src, filepaths in loader_iter:
                        B = x_src.shape[0]

                        src_idx = torch.full((B,), fill_value=src_domain_idx, device=device, dtype=torch.long)
                        tgt_idx = torch.full((B,), fill_value=tgt_domain_idx, device=device, dtype=torch.long)

                        x_src = x_src.to(device, non_blocking=True)
                        x_src_ctrl = ((x_src + 1.0) / 2.0).clamp(0.0, 1.0)

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
                                    num_steps=num_steps,
                                    eta=eta,
                                )
                            else:
                                raise ValueError("Sampler must be 'ddpm' or 'ddim'.")

                        for img, filepath in zip(samples, filepaths):
                            rel_path = Path(filepath).relative_to(translate_config.data.src_dir)
                            out_path = out_dir / rel_path
                            out_path.parent.mkdir(parents=True, exist_ok=True)

                            img = img.clamp(-1, 1)
                            img = ((img + 1.0) / 2.0).float()
                            save_image(img.cpu(), out_path)

                if is_ddp:
                    dist.barrier()
                if is_main:
                    print(f"[{combo}] done")

    if is_ddp:
        dist.barrier()
    if is_main:
        print("All ranks done!")

    ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
