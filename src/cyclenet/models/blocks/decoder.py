import torch
import torch.nn as nn

from cyclenet.models.blocks import ResBlock, UpsampleBlock, TransformerBlock
from cyclenet.models.utils import ContextIdentity


class DecoderBlock(nn.Module):
    """


    Args:
        in_ch (int):
        out_ch (int):
        skip_ch (int):
        t_dim (int):
        d_dim (int):
        num_heads (int):
        is_up (bool):
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
        skip_chs: list[int],
        t_dim: int,
        d_dim: int,
        num_heads: int,
        is_up: bool,
        res_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        super().__init__()
        # -- Track number of skip connections
        self.n_skips = len(skip_chs)
        # -------------------------
        # Residual Blocks (one for each skip connection)
        # -------------------------
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(
                    in_ch=in_ch + skip_chs[0],
                    out_ch=out_ch,
                    t_dim=t_dim,
                    d_dim=d_dim,
                    dropout=res_dropout,
                )
            ]
            + [
                ResBlock(
                    in_ch=out_ch + skip_chs[i],
                    out_ch=out_ch,
                    t_dim=t_dim,
                    d_dim=d_dim,
                    dropout=res_dropout,
                )
                for i in range(1, self.n_skips)
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
                    for _ in range(self.n_skips)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [ContextIdentity() for _ in range(self.n_skips)]
            )

        # -------------------------
        # Upsampling
        # -------------------------
        self.up = UpsampleBlock(out_ch, out_ch) if is_up else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        d_emb: torch.Tensor,
        d_ctx: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> torch.Tensor:
        # -------------------------
        # Residual Blocks / Self-Attention
        # -------------------------
        for res_block, transformer_block, skip in zip(
            self.res_blocks, self.transformer_blocks, skips
        ):
            x = torch.cat([x, skip], dim=1)
            x = res_block(x, t_emb, d_emb)
            x = transformer_block(x, d_ctx)

        # -------------------------
        # Upsample
        # -------------------------
        x = self.up(x)

        return x
