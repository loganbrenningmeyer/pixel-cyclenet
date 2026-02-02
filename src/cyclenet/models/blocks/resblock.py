import torch
import torch.nn as nn

from cyclenet.models.utils import zero_module


class ResBlock(nn.Module):
    """


    Parameters:
        in_ch (int):
        out_ch (int):
        t_dim (int):
        d_dim (int | None):
        dropout (float):
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_dim: int,
        d_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        # -------------------------
        # Skip Projection
        # -------------------------
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

        # -------------------------
        # Time Embedding Projection
        # -------------------------
        self.t_proj = nn.Linear(t_dim, out_ch)

        # -------------------------
        # Activation / Dropout
        # -------------------------
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # -------------------------
        # Normalization
        # -------------------------
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)

        # -------------------------
        # Convolutions
        # -------------------------
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # -- Initialize conv2 to zeros
        self.conv2 = zero_module(self.conv2)

        # -------------------------
        # Domain AdaGN (GroupNorm + FiLM)
        # -------------------------
        self.use_adagn = d_dim is not None
        if self.use_adagn:
            self.d1 = nn.Linear(d_dim, 2 * in_ch)  # scale/shift for norm1
            self.d2 = nn.Linear(d_dim, 2 * out_ch)  # scale/shift for norm2

            # -- Initialize to zeros to start identity
            self.d1 = zero_module(self.d1)
            self.d2 = zero_module(self.d2)

    def _apply_adagn(self, x: torch.Tensor, norm: nn.GroupNorm, d_proj: nn.Linear, d_emb: torch.Tensor):
        """
        Applys AdaGN (GroupNorm + FiLM) using the domain embedding d_emb.
        
        Args:
            x (torch.Tensor): Input of shape (B, C, H, W)
            norm (nn.GroupNorm): Normalization module
            d_proj (nn.Linear): Linear projection of d_emb
            d_emb: (torch.Tensor): Domain embedding
        
        Returns:
            h (torch.Tensor): FiLM-modulated output using d_emb
        """
        h = norm(x)
        gamma, beta = d_proj(d_emb).chunk(2, dim=1) # (B, C), (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return h * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, d_emb: torch.Tensor | None) -> torch.Tensor:
        # -- Skip connection projection
        skip = self.skip(x)

        # -------------------------
        # Block 1
        # -------------------------
        # -- AdaGN or only normalization
        if self.use_adagn and d_emb is not None:
            h = self._apply_adagn(x, self.norm1, self.d1, d_emb)
        else:
            h = self.norm1(x)
        # -- Activation + conv
        h = self.act(h)
        h = self.conv1(h)

        # -- Add time embedding
        h += self.t_proj(t_emb)[:, :, None, None]

        # -------------------------
        # Block 2
        # -------------------------
        # -- AdaGN or only normalization
        if self.use_adagn and d_emb is not None:
            h = self._apply_adagn(h, self.norm2, self.d2, d_emb)
        else:
            h = self.norm2(h)
        # -- Activation + dropout + conv
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)

        # -- Add residual
        return h + skip
