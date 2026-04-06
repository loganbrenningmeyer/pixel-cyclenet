import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        # -------------------------
        # Changed downsample stride 2 conv to stride 1 conv
        # -- Downsampling done by average pooling in forward()
        # -------------------------
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -------------------------
        # Changed stride 2 conv downsample to avgpool -> stride 1 conv
        # -------------------------
        # return self.conv(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = F.interpolate(x, scale_factor=2, mode="nearest")
        # -------------------------
        # Changed upsample nearest to bilinear
        # -------------------------
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)
