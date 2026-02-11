import torch
import torch.nn as nn


class DomainEmbedding(nn.Module):
    """
    Stores embeddings for source/target domains for domain-to-domain translation

    Parameters:
        d_dim (int): Dimensionality of domain embeddings
        init_std (float): Initial embedding standard deviation
    """

    def __init__(
        self,
        d_dim: int,
        cond_idx: int = 1,
        uncond_idx: int = 0,
        init_std: float = 0.02,
    ):
        super().__init__()
        # -------------------------
        # Define cond/uncond embedding indices
        # -------------------------
        # -- indices: cond = 1 (target domain), uncond = 0 (source domain)
        self.register_buffer("cond_idx", torch.tensor(cond_idx, dtype=torch.long))
        self.register_buffer("uncond_idx", torch.tensor(uncond_idx, dtype=torch.long))

        # -------------------------
        # Initialize embedding (2 domains)
        # -------------------------
        self.embed = nn.Embedding(2, d_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)

    def forward(self, domain_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns embeddings for given domain_idx of shape (B, d_dim)

        Args:
            domain_idx (torch.Tensor): Embedding domain indices of shape (B,)

        Returns:
            emb (torch.Tensor): Embeddings for specified domain indices of shape (B, d_dim)
        """
        # -- Ensure indices are torch.long
        if domain_idx.dtype != torch.long:
            domain_idx = domain_idx.long()
        return self.embed(domain_idx)

    def get_tokens(self, domain_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns embeddings as tokens for the given domain indices of shape (B, 1, d_dim)

        Args:
            domain_indices (torch.Tensor): Embedding domain indices of shape (B,)

        Returns:
            (torch.Tensor): Tokens for specified domain indices of shape (B, 1, d_dim)
        """
        return self.forward(domain_idx).unsqueeze(1)

    def drop_cond_idx(
        self, cond_idx: torch.Tensor, uncond_idx: torch.Tensor, p_dropout: float
    ) -> torch.Tensor:
        """
        CycleNet-style CFG dropout: replace *conditional* domain indices with uncond_idx
        with probability p_dropout. Non-conditional indices are left unchanged.

        Args:
            cond_idx (torch.Tensor): Domain indices for conditional domain per-sample (B,)
            uncond_idx (torch.Tensor): Domain indices for unconditional domain per-sample (B,)
            p_dropout (float): Probability of swapping cond_idx for uncond_idx
        """
        if p_dropout <= 0.0 or not self.training:
            return cond_idx

        B = cond_idx.shape[0]

        # -- Randomly replace cond indices with uncond_idx
        drop = (torch.rand(B, device=cond_idx.device) < p_dropout)
        out = cond_idx.clone()
        out[drop] = uncond_idx[drop]

        return out
