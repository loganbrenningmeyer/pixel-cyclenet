import os
import copy
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import OmegaConf, DictConfig

from cyclenet.training import UNetTrainer
from cyclenet.data import DomainDataset, DomainSampler, load_unet_transforms
from cyclenet.diffusion import DiffusionSchedule
from cyclenet.models import UNet
from cyclenet.models.conditioning import DomainEmbedding
from cyclenet.models.utils import unwrap


def ddp_setup():
    """
    Initializes torch.distributed if launched with torchrun.

    Returns: (is_ddp, rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def ddp_cleanup(is_ddp: bool):
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)


def make_adamw_param_groups(model: UNet, domain_emb: DomainEmbedding, weight_decay: float):
    """
    Sets all UNet normalization / bias parameters + DomainEmbedding to have no weight decay
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

    emb_params = [p for p in domain_emb.parameters() if p.requires_grad]

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
        {"params": emb_params, "weight_decay": 0.0},
    ]


def main():
    # -------------------------
    # Parse args / load + save config
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # -------------------------
    # Initialize DDP
    # -------------------------
    is_ddp, rank, local_rank, world_size = ddp_setup()
    is_main = (rank == 0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Set seeds
    # -------------------------
    seed = int(config.run.seed) if config.run.seed is not None else 0

    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)

    # -------------------------
    # Create training dir / save config
    # -------------------------
    train_dir = Path(config.run.runs_dir, config.run.name, "training")
    if is_main:
        train_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, train_dir / 'config.yaml')

    if is_ddp:
        dist.barrier()

    # -------------------------
    # Balanced DomainDatasets
    # -------------------------
    rank_batch_size = config.train.batch_size // world_size

    transforms = load_unet_transforms(config.data.transform_id, config.data.image_size)

    # -- Create real / sim datasets + concatenate [real, sim]
    real_ds = DomainDataset(config.data.tgt_dir, domain_idx=1, transforms=transforms)
    sim_ds  = DomainDataset(config.data.src_dir, domain_idx=0, transforms=transforms)

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
    # Initialize DomainEmbedding
    # -------------------------
    domain_emb = DomainEmbedding(d_dim=config.model.d_dim).to(device)

    # -------------------------
    # Initialize UNet model / EMA model
    # -------------------------
    model = UNet(
        in_ch=3,
        base_ch=config.model.base_ch,
        t_dim=config.model.t_dim,
        d_dim=config.model.d_dim,
        ch_mults=config.model.ch_mults,
        num_res_blocks=config.model.num_res_blocks,
        enc_heads=config.model.enc_heads,
        mid_heads=config.model.mid_heads,
        res_dropout=config.model.res_dropout,
        attn_dropout=config.model.attn_dropout,
        ffn_dropout=config.model.ffn_dropout
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # -------------------------
    # Create Optimizer (UNet + DomainEmbedding)
    # -------------------------
    # -- Separate params by weight decay / no weight decay
    param_groups = make_adamw_param_groups(model, domain_emb, config.train.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr)

    # -------------------------
    # Resume Training
    # -------------------------
    start_step = 1
    start_epoch = 1

    if config.run.resume.enable:
        ckpt_path = train_dir / "checkpoints" / config.run.resume.ckpt_name

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        domain_emb.load_state_dict(ckpt["domain_emb"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"]) + 1
        start_epoch = int(ckpt["epoch"]) + 1

        if is_main:
            print(f"\n==== Resuming {config.run.resume.ckpt_name} from step {ckpt['step']} ====")

    # -------------------------
    # Create DiffusionSchedule
    # -------------------------
    sched = DiffusionSchedule(
        schedule=config.diffusion.schedule,
        T=config.diffusion.T,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        device=device,
        s=config.diffusion.s,
    )

    # -------------------------
    # Wrap UNet & DomainEmbedding in DDP / sync EMA model
    # -------------------------
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        domain_emb = DDP(domain_emb, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # -- Only sync EMA on new training runs
        if not config.run.resume.enable:
            for p_ema, p_model in zip(ema_model.parameters(), unwrap(model).parameters()):
                p_ema.data.copy_(p_model.data)

    # -------------------------
    # Create UNetTrainer / run training
    # -------------------------
    trainer = UNetTrainer(
        model=model,
        ema_model=ema_model,
        domain_emb=domain_emb,
        sched=sched,
        optimizer=optimizer,
        dataloader=dataloader,
        device=device,
        train_dir=train_dir,
        log_config=config.logging,
        sample_config=config.sampling,
        ema_decay=config.train.ema_decay,
        is_main=is_main,
        start_step=start_step,
        start_epoch=start_epoch,
    )

    try:
        trainer.train(config.train.steps)
    finally:
        ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
