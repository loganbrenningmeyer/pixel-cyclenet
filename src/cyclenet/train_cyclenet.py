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

from cyclenet.data import CycleNetDataset, SourceDataset
from cyclenet.diffusion import DiffusionSchedule
from cyclenet.models import CycleNet, UNet, ControlNet
from cyclenet.models.conditioning import DomainEmbedding
from cyclenet.training import CycleNetTrainer


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
    # Parse args / load training config
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    # -------------------------
    # Load UNet backbone config
    # -------------------------
    unet_ckpt_path = Path(config.run.unet_ckpt)
    unet_train_dir = unet_ckpt_path.parent.parent
    unet_config = load_config(unet_train_dir / "config.yaml")

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
    if is_ddp:
        dist.barrier()

    # -------------------------
    # Training Dataset / DistributedSampler
    # -------------------------
    dataset = CycleNetDataset(config.data.src_dir, config.data.tgt_dir, image_size=config.data.image_size)

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
    # Source Samples Dataset / DataLoader (only main rank)
    # -------------------------
    sample_loader = None
    if is_main:
        sample_dataset = SourceDataset(config.data.src_dir, image_size=config.data.image_size)
        sample_loader = DataLoader(
            sample_dataset,
            batch_size=config.sampling.num_samples,
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

    # -- DDP broadcast loaded models
    if is_main:
        ckpt = torch.load(str(unet_ckpt_path), map_location="cpu")
        backbone.load_state_dict(ckpt["ema_model"])
        domain_emb.load_state_dict(ckpt["domain_emb"])

    if is_ddp:
        dist.barrier()
        for p in backbone.parameters():
            dist.broadcast(p.data, src=0)
        for p in domain_emb.parameters():
            dist.broadcast(p.data, src=0)

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
    # Resume Training
    # -------------------------
    start_step = 1
    if config.run.resume.enable:
        ckpt_path = train_dir / "checkpoints" / config.run.resume.ckpt_name
        if is_main:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(ckpt["model"])
            ema_model.load_state_dict(ckpt["ema_model"])
            start_step = int(ckpt["step"]) + 1

        if is_ddp:
            dist.barrier()
            for p in model.parameters():
                dist.broadcast(p.data, src=0)
            for p in ema_model.parameters():
                dist.broadcast(p.data, src=0)
            # -- Broadcast start_step
            t = torch.tensor([start_step], device=device, dtype=torch.long)
            dist.broadcast(t, src=0)
            start_step = int(t.item())
    
    # -------------------------
    # Create DiffusionSchedule
    # -------------------------
    sched = DiffusionSchedule(
        schedule=unet_config.diffusion.schedule,
        T=unet_config.diffusion.T,
        beta_start=unet_config.diffusion.beta_start,
        beta_end=unet_config.diffusion.beta_end,
        device=device,
        s=unet_config.diffusion.s
    )

    # -------------------------
    # Create Optimizer 
    # -------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params=trainable_params, 
        lr=config.train.lr, 
        weight_decay=config.train.weight_decay
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
        dataloader=dataloader,
        sample_loader=sample_loader,
        device=device,
        train_dir=train_dir,
        log_config=config.logging,
        sample_config=config.sampling,
        ema_decay=config.train.ema_decay,
        is_main=is_main,
        recon_weight=config.model.recon_weight,
        cycle_weight=config.model.cycle_weight,
        consis_weight=config.model.consis_weight,
        invar_weight=config.model.invar_weight,
        start_step=start_step
    )

    try:
        trainer.train(config.train.steps)
    finally:
        ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()