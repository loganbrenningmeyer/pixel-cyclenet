import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import DomainEmbedding, sinusoidal_embedding
from .blocks import EncoderBlock, DecoderBlock, Bottleneck, FinalLayer


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 128,
        t_dim: int = 512,
        d_dim: int = 128,
        ch_mults: list[int] = [1, 2, 2, 2],
        num_res_blocks: int = 2,
        enc_heads: list[int] = [0, 1, 0, 0],
        mid_heads: int = 1,
        res_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
    ):
        """
        Diffusion UNet model based on the original DDPM tensorflow
        implementation.

        Args:
            in_ch (int): Number of channels of input samples
            base_ch (int): Base channel width of the model
            num_res_blocks (int): Number of ResBlocks in each EncoderBlock
            ch_mults (list[int]): Base channel multipliers for each encoder layer
            enc_heads (list[int]): Number of attention heads in each encoder layer
            mid_heads (int): Number of attention heads in the Bottleneck
        """
        super().__init__()

        self.base_ch = base_ch
        self.num_res_blocks = num_res_blocks
        self.ch_mults = ch_mults

        # -------------------------
        # Time Embedding MLP
        # -------------------------
        self.t_mlp = nn.Sequential(
            nn.Linear(base_ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )

        # -------------------------
        # Stem
        # -------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, base_ch),
            nn.SiLU(),
        )

        # -- Update post-stem in_ch
        in_ch = base_ch

        # -------------------------
        # Encoder
        # -------------------------
        skip_chs = []

        self.encoder = nn.ModuleList()
        for i, (ch_mult, num_heads) in enumerate(zip(ch_mults, enc_heads)):
            # -- No downsampling at final block
            is_down = i != len(ch_mults) - 1
            # -- Compute out_ch based on block's ch_mult
            out_ch = base_ch * ch_mult
            # -- Store EncoderBlock skip channels
            enc_skip_chs = (
                [out_ch] * (num_res_blocks + 1)
                if is_down
                else [out_ch] * num_res_blocks
            )
            skip_chs.extend(enc_skip_chs)
            # -- Initialize EncoderBlock
            self.encoder.append(
                EncoderBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    t_dim=t_dim,
                    d_dim=d_dim,
                    num_res_blocks=num_res_blocks,
                    num_heads=num_heads,
                    is_down=is_down,
                    res_dropout=res_dropout,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                )
            )
            # -- Update in_ch for next EncoderBlock
            in_ch = out_ch

        # -------------------------
        # Bottleneck
        # -------------------------
        self.mid = Bottleneck(
            in_ch=in_ch,
            t_dim=t_dim,
            d_dim=d_dim,
            num_heads=mid_heads,
            res_dropout=res_dropout,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
        )

        # -------------------------
        # Decoder
        # -------------------------
        self.decoder = nn.ModuleList()
        for i, (ch_mult, num_heads) in enumerate(zip(ch_mults[::-1], enc_heads[::-1])):
            # -- No upsampling at final highest-resolution block
            is_up = i != len(ch_mults) - 1
            # -- Define out_ch
            out_ch = base_ch * ch_mult
            # -- Pop DecoderBlock skip channels
            #    Keep post-downsample skips; they should be consumed at the same (lower) resolution.
            #    Therefore all but the highest-resolution block use n_res + 1.
            num_skips = num_res_blocks if i == len(ch_mults) - 1 else num_res_blocks + 1
            dec_skip_chs = [skip_chs.pop() for _ in range(num_skips)]
            # -- Initialize DecoderBlock
            self.decoder.append(
                DecoderBlock(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    skip_chs=dec_skip_chs,
                    t_dim=t_dim,
                    d_dim=d_dim,
                    num_heads=num_heads,
                    is_up=is_up,
                    res_dropout=res_dropout,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                )
            )
            # -- Update in_ch for next DecoderBlock
            in_ch = out_ch

        # -------------------------
        # Final Layer
        # -------------------------
        self.final = FinalLayer(in_ch, 3)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, d_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        
        
        Args:
        
        
        Returns:
        
        """
        # -------------------------
        # Time Embedding
        # -------------------------
        t_emb = sinusoidal_embedding(t, self.base_ch)
        t_emb = self.t_mlp(t_emb)

        # -------------------------
        # Domain Embeddings -> Context Tokens
        # -------------------------
        d_ctx = d_emb.unsqueeze(1)

        # -------------------------
        # Encoder + Bottleneck
        # -------------------------
        x, skips = self.encode(x, t_emb, d_emb, d_ctx)

        # -------------------------
        # Decoder
        # -------------------------
        for dec_block in self.decoder:
            skips_i = [skips.pop() for _ in range(dec_block.n_skips)]
            x = dec_block(x, t_emb, d_emb, d_ctx, skips_i)

        # -------------------------
        # Final Layer
        # -------------------------
        x = self.final(x)

        return x

    def encode(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        d_emb: torch.Tensor,
        d_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        
        
        Args:
        
        
        Returns:
        
        """
        # -------------------------
        # Stem
        # -------------------------
        x = self.stem(x)

        # -------------------------
        # Encoder
        # -------------------------
        skips = []

        for enc_block in self.encoder:
            x, skips_i = enc_block(x, t_emb, d_emb, d_ctx)
            skips.extend(skips_i)

        # -------------------------
        # Bottleneck
        # -------------------------
        x = self.mid(x, t_emb, d_emb, d_ctx)

        return x, skips
