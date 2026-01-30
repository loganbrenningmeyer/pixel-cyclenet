import torch
import torch.nn as nn

from cyclenet.models import UNet, ControlNet


class CycleNet(nn.Module):
    def __init__(self, backbone: UNet, controlnet: ControlNet):
        super().__init__()

        self.backbone = backbone
        self.controlnet = controlnet

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

    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond_id: str, uncond_id: str):
        """
        
        """
        pass
