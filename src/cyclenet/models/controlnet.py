import copy
import torch
import torch.nn as nn

from cyclenet.models import UNet
from cyclenet.models.blocks import ZeroConvBlock
from cyclenet.models.conditioning import sinusoidal_embedding
from cyclenet.models.utils import zero_module


class ControlNet(nn.Module):
    def __init__(self, backbone: UNet, in_ch: int):
        super().__init__()
        # -------------------------
        # Copy backbone UNet encoder/mid blocks
        # -------------------------
        self.t_mlp = copy.deepcopy(backbone.t_mlp)
        self.stem = copy.deepcopy(backbone.stem)
        self.encoder = copy.deepcopy(backbone.encoder)
        self.mid = copy.deepcopy(backbone.mid)

        self.base_ch = backbone.base_ch
        self.ch_mults = backbone.ch_mults

        # -------------------------
        # Conditioning stem
        # -------------------------
        self.c_stem = nn.Sequential(
            nn.Conv2d(in_ch, self.base_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, self.base_ch),
            nn.SiLU(),
        )

        # -------------------------
        # Initialize zero-convolutions
        # -------------------------
        self.input_zero_conv = nn.Conv2d(self.base_ch, self.base_ch, kernel_size=1)
        self.input_zero_conv = zero_module(self.input_zero_conv)

        self.encoder_zero_convs = nn.ModuleList()

        # -- EncoderBlocks
        assert len(self.ch_mults) == len(self.encoder)

        enc_out_ch = self.base_ch

        for ch_mult, enc_block in zip(self.ch_mults, self.encoder):
            enc_out_ch = self.base_ch * ch_mult
            # -- Initialize 1x1 zero conv for each skip
            num_skips = enc_block.num_skips
            self.encoder_zero_convs.append(ZeroConvBlock(enc_out_ch, num_skips))

        # -- Bottleneck (init to zeros)
        self.mid_zero_conv = nn.Conv2d(enc_out_ch, enc_out_ch, kernel_size=1)
        self.mid_zero_conv = zero_module(self.mid_zero_conv)

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, d_emb: torch.Tensor | None) -> list[torch.Tensor]:
        """
        Returns list of ControlNet skips to be consumed by the backbone
        Bottleneck and Decoder blocks
        """
        # -------------------------
        # Time Embedding
        # -------------------------
        t_emb = sinusoidal_embedding(t, self.base_ch)
        t_emb = self.t_mlp(t_emb)

        # -------------------------
        # Domain Embeddings -> Context Tokens
        # -------------------------
        d_ctx = None if d_emb is None else d_emb.unsqueeze(1)

        # -------------------------
        # Input stem / Conditioning stem
        # -------------------------
        h = self.stem(x)
        hc = self.c_stem(c)

        # -------------------------
        # Zero-conv conditioning / add to input
        # -------------------------
        h = h + self.input_zero_conv(hc)

        # -------------------------
        # Store zero-conv ControlNet skips
        # -------------------------
        ctrl_skips = []

        for enc_block, enc_zero_conv in zip(self.encoder, self.encoder_zero_convs):
            h, skips = enc_block(h, t_emb, d_emb, d_ctx)
            # -- Apply zero-convs
            outs = enc_zero_conv(skips)
            ctrl_skips.extend(outs)

        # -------------------------
        # Store bottleneck skip
        # -------------------------
        h = self.mid(h, t_emb, d_emb, d_ctx)
        out = self.mid_zero_conv(h)
        ctrl_skips.append(out)

        return ctrl_skips
