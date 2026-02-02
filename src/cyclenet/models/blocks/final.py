import torch
import torch.nn as nn

from cyclenet.models.utils import zero_module


class FinalLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # -- Initialize conv to zeros
        self.conv = zero_module(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x
