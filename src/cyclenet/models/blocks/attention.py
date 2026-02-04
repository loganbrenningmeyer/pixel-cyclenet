import torch
import torch.nn as nn

from .ffn import FFNBlock
from ..utils import zero_module


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention for (B, C, H, W) inputs.

    Parameters:
        in_ch (int): Input channel width / embedding dimensionality
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, in_ch: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if in_ch % num_heads != 0:
            raise ValueError(
                f"in_ch ({in_ch}) must be divisible by num_heads ({num_heads})"
            )

        self.norm = nn.GroupNorm(32, in_ch)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_ch, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # -- Zero-init attention out_proj
        self.attn.out_proj = zero_module(self.attn.out_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        x_in = x

        # -- Pre-norm before attention
        x = self.norm(x)

        # -- (B, C, H, W) -> (B, HW, C)
        x = x.flatten(2).transpose(1, 2)

        # -- Apply Multi-head self-attention
        attn_out, _ = self.attn(x, x, x, need_weights=False)

        # -- (B, HW, C) -> (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, C, H, W)

        # -- Return attn + residual
        return x_in + attn_out


class CrossAttentionBlock(nn.Module):
    """
    Multi-head cross-attention for (B, C, H, W) queries attending to context tokens.

    Parameters:
        in_ch (int): Input channel width / embedding dimensionality
        d_dim: Dimensionality of token embeddings
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, in_ch: int, d_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if in_ch % num_heads != 0:
            raise ValueError(
                f"in_ch ({in_ch}) must be divisible by num_heads ({num_heads})"
            )

        # -- Project tokens to in_ch if different dimensions
        self.ctx_proj = nn.Linear(d_dim, in_ch) if d_dim != in_ch else nn.Identity()

        self.norm = nn.GroupNorm(32, in_ch)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_ch, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # -- Zero-init attention out_proj
        self.attn.out_proj = zero_module(self.attn.out_proj)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Performs multi-head cross-attention using the input x for queries and
        context for keys and values

        Args:
            x: (B, C, H, W)
            ctx: (B, T, d_dim)
        """
        # -- Handle null context
        if ctx is None:
            return x

        B, C, H, W = x.shape
        x_in = x

        # -- Pre-norm before attention
        x = self.norm(x)

        # -- (B, C, H, W) -> (B, HW, C)
        x_seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)

        # -- Project context to C channels
        ctx = self.ctx_proj(ctx)  # (B, T, C)
        # -- Cast to x_seq dtype/device (avoid fp16 issues)
        ctx = ctx.to(dtype=x_seq.dtype, device=x_seq.device)

        # -- Apply Multi-head cross-attention (x: Q), (ctx: KV)
        attn_out, _ = self.attn(x_seq, ctx, ctx, need_weights=False)

        # -- (B, HW, C) -> (B, C, H, W)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, C, H, W)

        # -- Return attn + residual
        return x_in + attn_out


class TransformerBlock(nn.Module):
    """
    Performs self-attention followed by cross-attention using the input x for
    QKV in self-attention, while using x for Q and context for KV in cross-attention

    Parameters:
        in_ch (int): Input channel width / embedding dimensionality
        d_dim: Dimensionality of token embeddings
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        in_ch: int,
        d_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        ffn_drouput: float = 0.0,
        ffn_mult: int = 4,
    ):
        super().__init__()
        if in_ch % num_heads != 0:
            raise ValueError(
                f"in_ch ({in_ch}) must be divisible by num_heads ({num_heads})"
            )

        # -- Self-attention / Cross-attention
        self.self_attn = SelfAttentionBlock(in_ch, num_heads, attn_dropout)
        self.cross_attn = CrossAttentionBlock(in_ch, d_dim, num_heads, attn_dropout)
        # -- Feed-forward network
        self.ffn = FFNBlock(in_ch, ffn_mult, ffn_drouput)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x)
        if ctx is not None:
            x = self.cross_attn(x, ctx)
        x = self.ffn(x)
        return x
