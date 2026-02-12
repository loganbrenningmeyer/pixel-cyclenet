import torch
import torch.nn as nn

from .controlnet import ControlNet
from .unet import UNet
from .conditioning import DomainEmbedding, sinusoidal_embedding


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
        # Freeze backbone UNet Encoder, Bottleneck, and DomainEmbedding
        # -------------------------
        frozen_params = [
            self.backbone.stem.parameters(),
            self.backbone.t_mlp.parameters(),
            self.backbone.encoder.parameters(),
            self.backbone.mid.parameters(),
            self.domain_emb.parameters(),
        ]
        for frozen_param in frozen_params:
            for p in frozen_param:
                p.requires_grad_(False)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        from_idx: torch.Tensor,
        to_idx: torch.Tensor,
        c_img: torch.Tensor,
        no_unet_grad: bool = False,
    ) -> torch.Tensor:
        """
        
        
        Args:
            x_t (torch.Tensor): 
            t (torch.Tensor): 
            from_idx (torch.Tensor): Array of shape (B,) defining each sample's unconditional domain
            to_idx (torch.Tensor): Array of shape (B,) defining each sample's conditional domain
            c_img (torch.Tensor): Conditioning image in the range [0, 1] of shape (B, C, H, W)
            no_unet_grad (bool): Enables / disables UNet Decoder / FinalLayer gradients in forward pass
        
        Returns:
        
        """
        # -------------------------
        # Get samples' domain embeddings / -> Context tokens
        # -------------------------
        from_emb = self.domain_emb(from_idx)
        to_emb = self.domain_emb(to_idx)

        # -------------------------
        # ControlNet: (origin domain)
        # -------------------------
        ctrl_skips = self.control(x_t, t, c_img, from_emb)

        # -------------------------
        # UNet Backbone Encode: (destination domain)
        # -------------------------
        with torch.no_grad():
            t_emb = sinusoidal_embedding(t, self.backbone.base_ch)
            t_emb = self.backbone.t_mlp(t_emb)

            to_ctx = to_emb.unsqueeze(1)

            h, skips = self.backbone.encode(x_t, t_emb, to_emb, to_ctx)

        if no_unet_grad:
            # -------------------------
            # Disable UNet backbone Decoder / FinalLayer gradients
            # -------------------------
            for p in self.backbone.decoder.parameters():
                p.requires_grad_(False)
            for p in self.backbone.final.parameters():
                p.requires_grad_(False)

        # -------------------------
        # UNet Backbone Decode: Add UNet skips + ControlNet skips
        # -------------------------
        h = h + ctrl_skips.pop()

        for dec_block in self.backbone.decoder:
            skips_i = [skips.pop() for _ in range(dec_block.n_skips)]
            skips_i = [s + ctrl_skips.pop() for s in skips_i]
            h = dec_block(h, t_emb, to_emb, to_ctx, skips_i)

        # -------------------------
        # UNet Backbone FinalLayer
        # -------------------------
        h = self.backbone.final(h)

        if no_unet_grad:
            # -------------------------
            # Re-enable UNet backbone Decoder / FinalLayer gradients
            # -------------------------
            for p in self.backbone.decoder.parameters():
                p.requires_grad_(True)
            for p in self.backbone.final.parameters():
                p.requires_grad_(True)

        return h
        