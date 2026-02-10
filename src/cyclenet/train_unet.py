import os
import copy
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf, DictConfig

from cyclenet.data import UNetDataset
from cyclenet.diffusion import DiffusionSchedule
from cyclenet.models import UNet
from cyclenet.models.conditioning import DomainEmbedding
from cyclenet.training import UNetTrainer


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


def main():
    # -------------------------
    # Parse args / load + save config
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # ----------
    # Initialize DDP
    # ----------
    is_ddp, rank, local_rank, world_size = ddp_setup()
    is_main = (rank == 0)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Create training dirs / save config
    # -------------------------
    train_dir = Path(config.run.runs_dir, config.run.name, "training")
    if is_main:
        train_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (train_dir / "figs").mkdir(parents=True, exist_ok=True)

        save_config(config, train_dir / 'config.yaml')

    if is_ddp:
        dist.barrier()

    # -------------------------
    # Dataset = DistributedSampler
    # -------------------------
    dataset = UNetDataset(config.data.src_dir, config.data.tgt_dir, image_size=config.data.image_size)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if is_ddp else None

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size // world_size,
        sampler=sampler,
        shuffle=(sampler is None),
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
    # Resume Training
    # -------------------------
    start_step = 1
    if config.run.resume.enable:
        ckpt_path = train_dir / "checkpoints" / config.run.resume.ckpt_name
        if is_main:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(ckpt["model"])
            ema_model.load_state_dict(ckpt["ema_model"])
            domain_emb.load_state_dict(ckpt["domain_emb"])
            start_step = int(ckpt["step"]) + 1
            print(f"\n==== Resuming {config.run.resume.ckpt_name} from step {ckpt['step']} ====")

        if is_ddp:
            # broadcast parameters so every rank has same weights
            dist.barrier()
            for p in model.parameters():
                dist.broadcast(p.data, src=0)
            for p in ema_model.parameters():
                dist.broadcast(p.data, src=0)
            for p in domain_emb.parameters():
                dist.broadcast(p.data, src=0)
            # broadcast start_step
            t = torch.tensor([start_step], device=device, dtype=torch.long)
            dist.broadcast(t, src=0)
            start_step = int(t.item())

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
    # Create Optimizer (UNet + DomainEmbedding)
    # -------------------------
    optimizer = torch.optim.AdamW(
        params=list(model.parameters()) + list(domain_emb.parameters()),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay
    )

    # ----------
    # Wrap UNet / DomainEmbedding in DDP
    # ----------
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        domain_emb = DDP(domain_emb, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

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
        ema_decay=config.train.ema_decay,
        is_main=is_main,
        start_step=start_step
    )

    try:
        trainer.train(config.train.steps)
    finally:
        ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
