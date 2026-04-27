import torch
import torch.nn as nn

from .attention import TransformerBlock
from .resblock import ResBlock, SPADEResBlock
from ..utils import ContextIdentity


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
        self.res1 = ResBlock(
            in_ch=in_ch,
            out_ch=in_ch,
            t_dim=t_dim,
            d_dim=d_dim,
            dropout=res_dropout,
        )
        self.transformer_block = (
            TransformerBlock(
                in_ch=in_ch,
                d_dim=d_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                ffn_drouput=ffn_dropout,
            )
            if num_heads != 0
            else ContextIdentity()
        )
        self.res2 = ResBlock(
            in_ch=in_ch,
            out_ch=in_ch,
            t_dim=t_dim,
            d_dim=d_dim,
            dropout=res_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        d_emb: torch.Tensor,
        d_ctx: torch.Tensor,
    ) -> torch.Tensor:
        x = self.res1(x, t_emb, d_emb)
        x = self.transformer_block(x, d_ctx)
        x = self.res2(x, t_emb, d_emb)
        return x


class SPADEBottleneck(nn.Module):
    """
    
    """
    def __init__(
        self,
        in_ch: int,
        seg_ch: int,
        t_dim: int,
        d_dim: int,
        s_dim: int,
        num_heads: int,
        res_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        self.res1 = SPADEResBlock(
            in_ch=in_ch,
            out_ch=in_ch,
            seg_ch=seg_ch,
            t_dim=t_dim,
            d_dim=d_dim,
            s_dim=s_dim,
            dropout=res_dropout,
        )
        self.transformer_block = (
            TransformerBlock(
                in_ch=in_ch,
                d_dim=d_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                ffn_drouput=ffn_dropout,
            )
            if num_heads != 0
            else ContextIdentity()
        )
        self.res2 = SPADEResBlock(
            in_ch=in_ch,
            out_ch=in_ch,
            seg_ch=seg_ch,
            t_dim=t_dim,
            d_dim=d_dim,
            s_dim=s_dim,
            dropout=res_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        seg: torch.Tensor,
        t_emb: torch.Tensor,
        d_emb: torch.Tensor,
        d_ctx: torch.Tensor,
    ) -> torch.Tensor:
        x = self.res1(x, seg, t_emb, d_emb)
        x = self.transformer_block(x, d_ctx)
        x = self.res2(x, seg, t_emb, d_emb)
        return x