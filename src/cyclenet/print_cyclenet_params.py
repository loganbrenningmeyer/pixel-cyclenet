import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cyclenet.models import UNet, ControlNet, CycleNet
from cyclenet.models.conditioning import DomainEmbedding


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a CycleNet checkpoint and print parameter counts.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to training/checkpoints/*.ckpt")
    parser.add_argument(
        "--state",
        type=str,
        default="model",
        choices=["model", "ema_model"],
        help="Checkpoint state dict to load",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    train_dir = ckpt_path.parent.parent
    config_path = train_dir / "config.yaml"

    config = OmegaConf.load(config_path)

    unet_ckpt_path = Path(config.run.unet_ckpt)
    unet_train_dir = unet_ckpt_path.parent.parent
    unet_config_path = unet_train_dir / "config.yaml"
    unet_config = OmegaConf.load(unet_config_path)

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
    )
    domain_emb = DomainEmbedding(d_dim=unet_config.model.d_dim)
    control = ControlNet(backbone, in_ch=3)

    model = CycleNet(
        backbone=backbone,
        control=control,
        domain_emb=domain_emb,
        t_dim=unet_config.model.t_dim,
        d_dim=unet_config.model.d_dim,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt[args.state], strict=True)

    total, trainable = count_params(model)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Config: {config_path}")
    print(f"UNet config: {unet_config_path}")
    print(f"Loaded state: {args.state}")
    print(f"CycleNet total params: {total:,}")
    print(f"CycleNet trainable params: {trainable:,}")


if __name__ == "__main__":
    main()
