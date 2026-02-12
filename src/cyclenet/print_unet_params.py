import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cyclenet.models import UNet


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a UNet checkpoint and print parameter counts.")
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
        ffn_dropout=config.model.ffn_dropout,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt[args.state], strict=True)

    total, trainable = count_params(model)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Config: {config_path}")
    print(f"Loaded state: {args.state}")
    print(f"UNet total params: {total:,}")
    print(f"UNet trainable params: {trainable:,}")


if __name__ == "__main__":
    main()
