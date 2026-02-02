import torch
import torch.nn as nn

from cyclenet.models.utils import zero_module


class ZeroConvBlock(nn.Module):
    def __init__(self, in_ch: int, num_skips: int):
        super().__init__()

        self.zero_convs = nn.ModuleList()
        for _ in range(num_skips):
            zero_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1)
            zero_conv = zero_module(zero_conv)
            self.zero_convs.append(zero_conv)

    def forward(self, skips: list[torch.Tensor]) -> list[torch.Tensor]:
        outs = []

        assert len(self.zero_convs) == len(skips)
        for skip, zero_conv in zip(skips, self.zero_convs):
            outs.append(zero_conv(skip))
        return outs
