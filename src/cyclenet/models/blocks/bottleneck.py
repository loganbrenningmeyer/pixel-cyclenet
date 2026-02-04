import torch
import torch.nn as nn

from .attention import TransformerBlock
from .resblock import ResBlock


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_ch: int,
        t_dim: int,
        d_dim: int,
        num_heads: int,
        res_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.res1 = ResBlock(in_ch, in_ch, t_dim, d_dim, res_dropout)
        self.transformer_block = (
            TransformerBlock(in_ch, d_dim, num_heads, attn_dropout, ffn_dropout)
            if num_heads != 0
            else nn.Identity()
        )
        self.res2 = ResBlock(in_ch, in_ch, t_dim, d_dim, res_dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, d_emb: torch.Tensor, d_ctx: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb, d_emb)
        x = self.transformer_block(x, d_ctx)
        x = self.res2(x, t_emb, d_emb)
        return x
