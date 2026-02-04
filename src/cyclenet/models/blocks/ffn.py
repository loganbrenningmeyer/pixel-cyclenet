import torch
import torch.nn as nn

from ..utils import zero_module


class FFNBlock(nn.Module):
    """
    Transformer Feed-Forward Network used as the final layer in
    a TransformerBlock, applies position-wise MLP on (B, C, H, W) input

    Parameters:
        in_ch: Input channel dimensions
        mult: Multiplier on in_ch for hidden dimensionality
        dropout: Dropout probability
    """

    def __init__(self, in_ch: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # -- FFN hidden dimensionality
        hidden = in_ch * mult

        self.norm = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # -------------------------
        # 1x1 Conv on (B, C, H, W) equivalent to Linear on (B, H*W, C)
        # -------------------------
        self.fc1 = nn.Conv2d(in_ch, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, in_ch, kernel_size=1)

        # -- Zero-init last layer so starts at 0
        self.fc2 = zero_module(self.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        # -- Fully connected layer 1
        x = self.act(self.fc1(self.norm(x)))
        # -- Dropout
        x = self.drop(x)
        # -- Fully connected layer 2
        x = self.fc2(x)

        return x_in + x
