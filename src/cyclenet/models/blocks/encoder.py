import torch
import torch.nn as nn

from .attention import TransformerBlock
from .resblock import ResBlock
from .updown import DownsampleBlock
from ..utils import ContextIdentity


class EncoderBlock(nn.Module):
    """


    Args:
        in_ch (int):
        out_ch (int):
        skip_ch (int):
        t_dim (int):
        d_dim (int):
        num_res_blocks (int):
        num_heads (int):
        is_down (bool):
        res_dropout (float):
        attn_dropout (float):
        ffn_dropout (float):

    Returns:
        x (torch.Tensor):
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_dim: int,
        d_dim: int,
        num_res_blocks: int,
        num_heads: int,
        is_down: bool,
        res_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()

        self.is_down = is_down
        self.num_skips = num_res_blocks + 1 if is_down else num_res_blocks

        # -------------------------
        # Residual Blocks
        # -------------------------
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    t_dim=t_dim,
                    d_dim=d_dim,
                    dropout=res_dropout,
                )
            ]
            + [
                ResBlock(
                    in_ch=out_ch,
                    out_ch=out_ch,
                    t_dim=t_dim,
                    d_dim=d_dim,
                    dropout=res_dropout,
                )
                for _ in range(num_res_blocks - 1)
            ]
        )

        # -------------------------
        # Self-Attention / Cross-Attention / FFN
        # -------------------------
        if num_heads != 0:
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        in_ch=out_ch,
                        d_dim=d_dim,
                        num_heads=num_heads,
                        attn_dropout=attn_dropout,
                        ffn_drouput=ffn_dropout,
                    )
                    for _ in range(num_res_blocks)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [ContextIdentity() for _ in range(num_res_blocks)]
            )

        # -------------------------
        # Downsample
        # -------------------------
        self.down = DownsampleBlock(out_ch, out_ch) if is_down else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        d_emb: torch.Tensor,
        d_ctx: torch.Tensor,
    ) -> torch.Tensor:
        skips = []
        # -------------------------
        # Residual Blocks / Self-Attention
        # -------------------------
        for res_block, transformer_block in zip(
            self.res_blocks, self.transformer_blocks
        ):
            x = res_block(x, t_emb, d_emb)
            x = transformer_block(x, d_ctx)
            skips.append(x)

        # -------------------------
        # Downsample
        # -------------------------
        x = self.down(x)
        if self.is_down:
            skips.append(x)

        return x, skips
