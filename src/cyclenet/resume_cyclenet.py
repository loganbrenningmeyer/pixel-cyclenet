import os
import copy
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from cyclenet.training import CycleNetTrainer
from cyclenet.data import CycleDomainDataset, SourceDataset, DomainSampler, load_cyclenet_transforms
from cyclenet.diffusion import DiffusionSchedule
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


def make_adamw_param_groups(model: CycleNet, weight_decay: float):
    """
    Sets all CycleNet normalization / bias parameters to have no weight decay
    """
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        n = name.lower()
        is_bias = n.endswith("bias")
        is_norm = (
            "norm" in n
            or "groupnorm" in n
            or n.endswith("gn.weight")
            or n.endswith("gn.bias")
        )

        if is_bias or is_norm:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _resolve_optional_bool(override_value, saved_value: bool) -> bool:
    """
    Resolves an optional bool override from OmegaConf / YAML.
    """
    if override_value is None:
        return saved_value

    if isinstance(override_value, bool):
        return override_value

    if isinstance(override_value, str):
        value = override_value.strip().lower()
        if value in {"true", "1", "yes", "on"}:
            return True
        if value in {"false", "0", "no", "off"}:
            return False

    raise ValueError(f"Invalid boolean override: {override_value!r}")


def resolve_model_config(
    source_config: DictConfig,
    resume_config: DictConfig,
) -> tuple[DictConfig, bool]:
    """
    Resolves the active model config from the saved config plus optional
    resume-time overrides.
    """
    resolved_model_config = copy.deepcopy(source_config.model)
    changed = False

    weight_keys = ("recon_weight", "cycle_weight", "consis_weight", "invar_weight")

    for key in weight_keys:
        saved_value = float(OmegaConf.select(source_config, f"model.{key}"))
        override_value = OmegaConf.select(resume_config, f"model.{key}")
        active_value = saved_value if override_value is None else float(override_value)
        resolved_model_config[key] = active_value

        if active_value != saved_value:
            changed = True

    saved_invar_unet_grad = _resolve_optional_bool(
        OmegaConf.select(source_config, "model.invar_unet_grad"),
        False,
    )
    override_invar_unet_grad = OmegaConf.select(resume_config, "model.invar_unet_grad")
    active_invar_unet_grad = _resolve_optional_bool(
        override_invar_unet_grad,
        saved_invar_unet_grad,
    )
    resolved_model_config.invar_unet_grad = active_invar_unet_grad

    if active_invar_unet_grad != saved_invar_unet_grad:
        changed = True

    return resolved_model_config, changed


def make_output_config(
    source_config: DictConfig,
    resolved_model_config: DictConfig,
    out_run_dir: Path,
) -> DictConfig:
    """
    Creates the config to save for a fine-tune branch with updated model config.
    """
    output_config = copy.deepcopy(source_config)
    output_config.model = copy.deepcopy(resolved_model_config)

    output_config.run.runs_dir = str(out_run_dir.parent)
    output_config.run.name = out_run_dir.name

    return output_config


def main():
    # -------------------------
    # [Resume]: Load config
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    resume_config = load_config(args.config)

    # -------------------------
    # [Saved model]: Load config
    # -------------------------
    source_run_dir = Path(resume_config.run.run_dir)
    source_train_dir = source_run_dir / "training"
    source_config = load_config(source_train_dir / "config.yaml")

    # -------------------------
    # [Saved UNet]: Load config
    # -------------------------
    unet_ckpt_path = Path(source_config.run.unet_ckpt)
    unet_train_dir = unet_ckpt_path.parent.parent
    unet_config = load_config(unet_train_dir / "config.yaml")

    # -------------------------
    # Initialize DDP
    # -------------------------
    is_ddp, rank, local_rank, world_size = ddp_setup()
    is_main = (rank == 0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Set seeds
    # -------------------------
    seed = int(source_config.run.seed) if source_config.run.seed is not None else 0

    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)

    # -------------------------
    # Resolve model config / output run dir
    # -------------------------
    model_config, model_config_changed = resolve_model_config(source_config, resume_config)

    out_run_dir_value = OmegaConf.select(resume_config, "run.out_run_dir")

    if model_config_changed:
        if out_run_dir_value is None:
            raise ValueError(
                "If resuming with changed model settings, resume config must set run.out_run_dir."
            )

        run_dir = Path(out_run_dir_value)
        train_dir = run_dir / "training"
        output_config = make_output_config(source_config, model_config, run_dir)
    else:
        run_dir = source_run_dir
        train_dir = source_train_dir
        output_config = source_config

    # -------------------------
    # Create output dir / save config
    # -------------------------
    if is_main:
        train_dir.mkdir(parents=True, exist_ok=True)

        if model_config_changed:
            save_config(output_config, train_dir / "config.yaml")

        save_config(resume_config, train_dir / "resume_config.yaml")

        if model_config_changed:
            print(f"Starting fine-tune branch in {run_dir}")
            print(
                "Model config: "
                f"recon={model_config.recon_weight}, "
                f"cycle={model_config.cycle_weight}, "
                f"consis={model_config.consis_weight}, "
                f"invar={model_config.invar_weight}, "
                f"invar_unet_grad={model_config.invar_unet_grad}"
            )
        else:
            print(f"Continuing run in {run_dir}")

    if is_ddp:
        dist.barrier()

    # -------------------------
    # Balanced DomainDatasets
    # -------------------------
    rank_batch_size = source_config.train.batch_size // world_size

    transforms = load_cyclenet_transforms(source_config.data.transform_id, source_config.data.image_size)

    # -- Create real / sim datasets + concatenate [real, sim]
    rgb_parent_dirs = set(source_config.data.rgb_parent_dirs) if source_config.data.get("rgb_parent_dirs") is not None else None

    real_ds = CycleDomainDataset(
        data_dir=source_config.data.tgt_dir,
        rgb_parent_dirs=rgb_parent_dirs,
        domain_idx=1,
        transforms=transforms,
    )
    sim_ds = CycleDomainDataset(
        data_dir=source_config.data.src_dir,
        rgb_parent_dirs=rgb_parent_dirs,
        domain_idx=0,
        transforms=transforms,
    )

    if is_main:
        print(f"( Real Dataset ): {len(real_ds)} images")
        print(f"( Sim Dataset ): {len(sim_ds)} images")

    dataset = ConcatDataset([real_ds, sim_ds])

    # -- Create DomainSampler to balance real / sim samples
    batch_sampler = DomainSampler(
        n_real=len(real_ds),
        n_sim=len(sim_ds),
        batch_size=rank_batch_size,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # -------------------------
    # Source Samples Dataset / DataLoader (only main rank)
    # -------------------------
    sample_dataset = SourceDataset(
        src_dir=source_config.data.src_dir,
        image_size=source_config.data.image_size,
        rgb_parent_dirs=rgb_parent_dirs,
    )

    sample_loader = None
    if is_main:
        sample_loader = DataLoader(
            sample_dataset,
            batch_size=source_config.sampling.num_samples,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

    # -------------------------
    # Load UNet Backbone / DomainEmbedding
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
        ffn_dropout=unet_config.model.ffn_dropout
    ).to(device)

    domain_emb = DomainEmbedding(d_dim=unet_config.model.d_dim).to(device)

    unet_ckpt = torch.load(str(unet_ckpt_path), map_location="cpu")
    backbone.load_state_dict(unet_ckpt["ema_model"])
    domain_emb.load_state_dict(unet_ckpt["domain_emb"])

    # -------------------------
    # Initialize ControlNet
    # -------------------------
    control = ControlNet(backbone, in_ch=3).to(device)

    # -------------------------
    # Initialize CycleNet / EMA model
    # -------------------------
    model = CycleNet(
        backbone=backbone,
        control=control,
        domain_emb=domain_emb,
        t_dim=unet_config.model.t_dim,
        d_dim=unet_config.model.d_dim
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # -------------------------
    # Create Optimizer
    # -------------------------
    param_groups = make_adamw_param_groups(model, source_config.train.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=source_config.train.lr)

    # -------------------------
    # Create GradScaler
    # -------------------------
    scaler = GradScaler(device="cuda")

    # -------------------------
    # Load saved model
    # -------------------------
    ckpt_path = source_train_dir / "checkpoints" / resume_config.run.ckpt_name

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    ema_model.load_state_dict(ckpt["ema_model"])

    if not model_config_changed:
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"]) if "scaler" in ckpt else None

    start_step = int(ckpt["step"]) + 1
    start_epoch = int(ckpt["epoch"]) + 1

    # -------------------------
    # Create DiffusionSchedule
    # -------------------------
    sched = DiffusionSchedule(
        schedule=unet_config.diffusion.schedule,
        T=unet_config.diffusion.T,
        beta_start=unet_config.diffusion.beta_start,
        beta_end=unet_config.diffusion.beta_end,
        device=device,
        s=unet_config.diffusion.s,
    )

    # -------------------------
    # Wrap CycleNet in DDP
    # -------------------------
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # -------------------------
    # Create CycleNetTrainer / Run training
    # -------------------------
    trainer = CycleNetTrainer(
        model=model,
        ema_model=ema_model,
        sched=sched,
        optimizer=optimizer,
        scaler=scaler,
        dataloader=dataloader,
        sample_loader=sample_loader,
        device=device,
        train_dir=train_dir,
        model_config=model_config,
        log_config=output_config.logging,
        sample_config=output_config.sampling,
        ema_decay=output_config.train.ema_decay,
        is_main=is_main,
        start_step=start_step,
        start_epoch=start_epoch,
    )

    try:
        trainer.train(resume_config.train.total_steps)
    finally:
        ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
