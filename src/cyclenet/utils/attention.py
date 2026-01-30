import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """
    PyTorch-backed multi-head self-attention for (B, C, H, W) inputs.

    Args:
        in_ch (int): Input channel width / embedding dimensionality
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability

    Returns:
        x (torch.Tensor): Self-attended output of shape (B, C, H, W)
    """
    def __init__(
            self,
            in_ch: int,
            num_heads: int,
            dropout: float = 0.0
    ):
        super().__init__()

        if in_ch % num_heads != 0:
            raise ValueError(f"in_ch ({in_ch}) must be divisible by num_heads ({num_heads})")

        self.in_ch = in_ch
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=in_ch,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.out_proj = nn.Linear(in_ch, in_ch)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, T, C)
        B, C, H, W = x.shape
        T = H * W
        x = x.flatten(2).transpose(1, 2)

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attn_out = self.drop(attn_out)
        x = x + self.out_proj(attn_out)

        # (B, T, C) -> (B, C, H, W)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        return x
