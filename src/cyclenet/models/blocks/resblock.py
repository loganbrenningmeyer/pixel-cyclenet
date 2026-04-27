import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import zero_module


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

    def _adagn_mod(
        self,
        d_proj: nn.Linear,
        d_emb: torch.Tensor,
    ):
        """
        Applys AdaGN (GroupNorm + FiLM) using the domain embedding d_emb.

        Args:
            d_proj (nn.Linear): Linear projection of d_emb
            d_emb: (torch.Tensor): Domain embedding

        Returns:
            h (torch.Tensor): FiLM-modulated output using d_emb
        """
        gamma, beta = d_proj(d_emb).chunk(2, dim=1)  # (B, C), (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return gamma, beta

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, d_emb: torch.Tensor
    ) -> torch.Tensor:
        # -- Skip connection projection
        skip = self.skip(x)

        # -------------------------
        # Block 1
        # -------------------------
        h = self.norm1(x)

        # -- AdaGN modulation
        if self.use_adagn and d_emb is not None:
            gamma, beta = self._adagn_mod(self.d1, d_emb)
            h = h * (1 + gamma) + beta

        # -- Activation + conv
        h = self.act(h)
        h = self.conv1(h)

        # -- Add time embedding
        h += self.t_proj(t_emb)[:, :, None, None]

        # -------------------------
        # Block 2
        # -------------------------
        h = self.norm2(h)

        # -- AdaGN or only normalization
        if self.use_adagn and d_emb is not None:
            gamma, beta = self._adagn_mod(self.d2, d_emb)
            h = h * (1 + gamma) + beta

        # -- Activation + dropout + conv
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)

        # -- Add residual
        return h + skip


class SPADEResBlock(nn.Module):
    """


    Parameters:
        in_ch (int):
        out_ch (int):
        seg_ch (int):
        t_dim (int):
        d_dim (int):
        s_dim (int):
        dropout (float):
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        seg_ch: int,
        t_dim: int,
        d_dim: int | None = None,
        s_dim: int | None = None,
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
        self.norm1 = nn.GroupNorm(32, in_ch, affine=False)
        self.norm2 = nn.GroupNorm(32, out_ch, affine=False)

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

        # -------------------------
        # Segmentation SPADE blocks
        # -------------------------
        self.use_spade = s_dim is not None

        if self.use_spade:
            # -- SPADE block 1
            self.s1 = nn.Sequential(
                nn.Conv2d(seg_ch, s_dim, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.s_gamma1 = nn.Conv2d(s_dim, in_ch, kernel_size=3, padding=1)
            self.s_beta1 = nn.Conv2d(s_dim, in_ch, kernel_size=3, padding=1)

            # -- SPADE block 2
            self.s2 = nn.Sequential(
                nn.Conv2d(seg_ch, s_dim, kernel_size=3, padding=1),
                nn.SiLU(),
            )
            self.s_gamma2 = nn.Conv2d(s_dim, out_ch, kernel_size=3, padding=1)
            self.s_beta2 = nn.Conv2d(s_dim, out_ch, kernel_size=3, padding=1)

            # -- Initialize gamma/beta heads to zeros to start identity
            self.s_gamma1 = zero_module(self.s_gamma1)
            self.s_gamma2 = zero_module(self.s_gamma2)
            self.s_beta1 = zero_module(self.s_beta1)
            self.s_beta2 = zero_module(self.s_beta2)

    def _adagn_mod(
        self, d_emb: torch.Tensor, d_proj: nn.Linear
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ """
        gamma_d, beta_d = d_proj(d_emb).chunk(2, dim=1)  # (B, C), (B, C)
        gamma_d = gamma_d.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta_d = beta_d.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return gamma_d, beta_d

    def _spade_mod(
        self,
        seg: torch.Tensor,
        shared: nn.Sequential,
        gamma_head: nn.Conv2d,
        beta_head: nn.Conv2d,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        seg: [B, C_seg, H_seg, W_seg]
        """
        seg = F.interpolate(seg.float(), size=size, mode="nearest")  # [B, C_seg, H, W]
        h = shared(seg)  # [B, s_dim, H, W]
        gamma_s = gamma_head(h)  # [B, C, H, W]
        beta_s = beta_head(h)  # [B, C, H, W]
        return gamma_s, beta_s

    def forward(
        self,
        x: torch.Tensor,
        seg: torch.Tensor,
        t_emb: torch.Tensor,
        d_emb: torch.Tensor,
    ) -> torch.Tensor:
        # -- Skip connection projection
        skip = self.skip(x)

        # -------------------------
        # Block 1
        # -------------------------
        h = self.norm1(x)

        # -- AdaGN domain modulation
        if self.use_adagn and d_emb is not None:
            gamma_d, beta_d = self._adagn_mod(d_emb, self.d1)
        else:
            gamma_d, beta_d = 0.0, 0.0

        # -- SPADE segmentation modulation
        if self.use_spade and seg is not None:
            gamma_s, beta_s = self._spade_mod(
                seg,
                shared=self.s1,
                gamma_head=self.s_gamma1,
                beta_head=self.s_beta1,
                size=x.shape[-2:],
            )
        else:
            gamma_s, beta_s = 0.0, 0.0

        h = h * (1 + gamma_d + gamma_s) + (beta_d + beta_s)

        # -- Activation + conv
        h = self.act(h)
        h = self.conv1(h)

        # -- Add time embedding
        h += self.t_proj(t_emb)[:, :, None, None]

        # -------------------------
        # Block 2
        # -------------------------
        h = self.norm2(h)

        # -- AdaGN domain modulation
        if self.use_adagn and d_emb is not None:
            gamma_d, beta_d = self._adagn_mod(d_emb, self.d2)
        else:
            gamma_d, beta_d = 0.0, 0.0

        # -- SPADE segmentation modulation
        if self.use_spade and seg is not None:
            gamma_s, beta_s = self._spade_mod(
                seg,
                shared=self.s2,
                gamma_head=self.s_gamma2,
                beta_head=self.s_beta2,
                size=h.shape[-2:],
            )
        else:
            gamma_s, beta_s = 0.0, 0.0

        h = h * (1 + gamma_d + gamma_s) + (beta_d + beta_s)

        # -- Activation + dropout + conv
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)

        # -- Add residual
        return h + skip
