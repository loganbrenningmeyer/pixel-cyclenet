import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast
from torchvision.utils import save_image
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
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


def main():
    # -------------------------
    # [Translation]: Load config
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    translate_config = load_config(args.config)

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

    # -------------------------
    # Create out dir / save config
    # -------------------------
    out_dir = Path(translate_config.data.out_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_config(translate_config, out_dir / "config.yaml")

    if is_ddp:
        dist.barrier()

    # -------------------------
    # Source Samples Dataset / DataLoader
    # -------------------------
    dataset = TranslateDataset(translate_config.data.src_dir, image_size=translate_config.data.image_size)

    # -- Create subset per-rank
    indices = list(range(rank, len(dataset), world_size))
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=translate_config.sampling.batch_size // world_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------
    # Initialize UNet / DomainEmbedding
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

    # -------------------------
    # Initialize ControlNet with segmentation map channels
    # -------------------------
    num_seg_classes = cyclenet_config.model.num_seg_classes
    control_in_ch = 3 + num_seg_classes

    control = ControlNet(backbone, in_ch=control_in_ch).to(device)

    # -------------------------
    # Load CycleNet (EMA model)
    # -------------------------
    model = CycleNet(
        backbone=backbone,
        control=control,
        domain_emb=domain_emb,
        t_dim=unet_config.model.t_dim,
        d_dim=unet_config.model.d_dim,
    ).to(device)

    ckpt_path = Path(translate_config.run.run_dir) / "training" / "checkpoints" / translate_config.run.ckpt_name
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    model.load_state_dict(ckpt["ema_model"], strict=True)

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
    # Tranlsate Dataset
    # -------------------------
    model.eval()
    torch.set_grad_enabled(False)

    # -- Source / target data domain index
    src_domain_idx = translate_config.data.src_idx
    tgt_domain_idx = 1 - src_domain_idx

    with torch.inference_mode():

        # -- tqdm on main only
        loader_iter = dataloader
        if is_main:
            loader_iter = tqdm(dataloader, desc="Translating", unit="batch")

        for x_src, filepaths in loader_iter:
            # -------------------------
            # Define source / target indices
            # -------------------------
            B = x_src.shape[0]

            src_idx = torch.full((B,), fill_value=src_domain_idx, device=device, dtype=torch.long)
            tgt_idx = torch.full((B,), fill_value=tgt_domain_idx, device=device, dtype=torch.long)

            # -------------------------
            # Generate samples
            # -------------------------
            sampler = translate_config.sampling.sampler
            cfg_weight = translate_config.sampling.cfg_weight
            noise_strength = translate_config.sampling.noise_strength
            num_steps = translate_config.sampling.ddim_steps
            eta = translate_config.sampling.eta

            # -- Move to GPU / create control image
            x_src = x_src.to(device, non_blocking=True)
            x_src_ctrl = ((x_src + 1.0) / 2.0).clamp(0.0, 1.0)

            with autocast(device_type="cuda"):
                # -------------------------
                # DDPM
                # -------------------------
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
                # -------------------------
                # DDIM
                # -------------------------
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
                
            # -------------------------
            # Save samples
            # -------------------------
            for img, filepath in zip(samples, filepaths):
                rel_path = Path(filepath).relative_to(translate_config.data.src_dir)
                out_path = Path(out_dir) / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # -- Convert to [0, 1]
                img = img.clamp(-1, 1)
                img = ((img + 1.0) / 2.0).float()

                # -- Save image
                save_image(img.cpu(), out_path)

    if is_ddp:
        dist.barrier()
    if is_main:
        print("All ranks done!")


if __name__ == "__main__":
    main()