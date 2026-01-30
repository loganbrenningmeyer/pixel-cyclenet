import copy
import torch
import torch.nn as nn

from cyclenet.models import UNet
from cyclenet.utils import sinusoidal_encoding


class ZeroConvBlock(nn.Module):
    def __init__(self, in_ch: int, num_skips: int):
        super().__init__()

        self.zero_convs = nn.ModuleList()
        for _ in range(num_skips):
            zero_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1)
            nn.init.zeros_(zero_conv.weight)
            nn.init.zeros_(zero_conv.bias)
            self.zero_convs.append(zero_conv)

    def forward(self, skips: list[torch.Tensor]) -> list[torch.Tensor]:
        outs = []

        assert len(self.zero_convs) == len(skips)
        for skip, zero_conv in zip(skips, self.zero_convs):
            outs.append(zero_conv(skip))
        return outs


class ControlNet(nn.Module):
    def __init__(self, backbone: UNet):
        super().__init__()
        self.backbone = backbone

        # -------------------------
        # Copy backbone UNet encoder/mid blocks
        # -------------------------
        self.t_mlp = copy.deepcopy(backbone.t_mlp)
        self.stem = copy.deepcopy(backbone.stem)
        self.encoder = copy.deepcopy(backbone.encoder)
        self.mid = copy.deepcopy(backbone.mid)

        # -------------------------
        # Initialize zero-convolutions
        # -------------------------
        self.base_ch = backbone.base_ch
        self.ch_mults = backbone.ch_mults

        self.encoder_zero_convs = nn.ModuleList()

        # -- EncoderBlocks
        assert len(self.ch_mults) == len(self.encoder)

        enc_out_ch = self.base_ch

        for ch_mult, enc_block in zip(self.ch_mults, self.encoder):
            enc_out_ch = self.base_ch * ch_mult
            # -- Initialize 1x1 zero conv for each skip
            num_skips = enc_block.num_skips
            self.encoder_zero_convs.append(ZeroConvBlock(enc_out_ch, num_skips))

        # -- Bottleneck
        self.mid_zero_conv = nn.Conv2d(enc_out_ch, enc_out_ch, kernel_size=1)
        nn.init.zeros_(self.mid_zero_conv.weight)
        nn.init.zeros_(self.mid_zero_conv.bias)


    def forward(self, x: torch.Tensor, t: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns list of ControlNet skips to be consumed by the backbone
        Bottleneck and Decoder blocks
        """
        # -------------------------
        # Time Embedding
        # -------------------------
        t_emb = sinusoidal_encoding(t, self.base_ch)
        t_emb = self.t_mlp(t_emb)

        # -------------------------
        # Stem
        # -------------------------
        x = self.stem(x)

        # -------------------------
        # Store zero-conv ControlNet skips
        # -------------------------
        ctrl_skips = []

        for enc_block, enc_zero_conv in zip(self.encoder, self.encoder_zero_convs):
            x, skips = enc_block(x, t_emb)
            # -- Apply zero-convs
            outs = enc_zero_conv(skips)
            ctrl_skips.extend(outs)

        # -------------------------
        # Store bottleneck skip
        # -------------------------
        x = self.mid(x, t_emb)
        out = self.mid_zero_conv(x)
        ctrl_skips.append(out)

        return ctrl_skips