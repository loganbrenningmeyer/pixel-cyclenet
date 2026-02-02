import torch
import torch.nn as nn

from cyclenet.models import UNet, ControlNet
from cyclenet.models.conditioning import DomainEmbedding, sinusoidal_embedding


class CycleNet(nn.Module):
    def __init__(
        self,
        backbone: UNet,
        control: ControlNet,
        domain_emb: DomainEmbedding,
        t_dim: int = 512,
        d_dim: int = 128,
    ):
        super().__init__()

        self.backbone = backbone
        self.control = control
        self.domain_emb = domain_emb
        self.t_dim = t_dim
        self.d_dim = d_dim

        # -------------------------
        # Freeze backbone UNet Encoder / Bottleneck
        # -------------------------
        frozen_params = [
            self.backbone.stem.parameters(),
            self.backbone.t_mlp.parameters(),
            self.backbone.encoder.parameters(),
            self.backbone.mid.parameters(),
        ]
        for frozen_param in frozen_params:
            for p in frozen_param:
                p.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_idx: torch.Tensor,
        uncond_idx: torch.Tensor,
        p_dropout: float = 0.0,
    ):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        # -------------------------
        # CFG conditional embedding dropout (swap with uncond_idx)
        # -------------------------
        cond_idx_drop = self.domain_emb.drop_cond_idx(cond_idx, uncond_idx, p_dropout)

        # -------------------------
        # Get samples' domain embeddings / -> Context tokens
        # -------------------------
        cond_emb = self.domain_emb(cond_idx_drop)
        uncond_emb = self.domain_emb(uncond_idx)

        # -------------------------
        # ControlNet: Unconditional (source)
        # -------------------------
        # -- Use input as ControlNet conditioning
        c = x
        ctrl_skips = self.control(x, c, t, uncond_emb)

        # -------------------------
        # UNet Backbone Encode (Frozen)
        # -------------------------
        with torch.no_grad():
            t_emb = sinusoidal_embedding(t, self.backbone.base_ch)
            t_emb = self.backbone.t_mlp(t_emb)

            cond_ctx = cond_emb.unsqueeze(1)

            x, skips = self.backbone.encode(x, t_emb, cond_emb, cond_ctx)

        # -------------------------
        # UNet Backbone Decode: Add UNet skips + ControlNet skips
        # -------------------------
        x = x + ctrl_skips.pop()

        for dec_block in self.backbone.decoder:
            skips_i = [skips.pop() for _ in range(dec_block.n_skips)]
            skips_i = [s + ctrl_skips.pop() for s in skips_i]
            x = dec_block(x, t_emb, cond_emb, cond_ctx, skips_i)

        # -------------------------
        # UNet Backbone FinalLayer
        # -------------------------
        x = self.backbone.final(x)

        return x
        

