import torch
import torch.nn as nn


class DomainEmbedding(nn.Module):
    """
    Stores embeddings for source/target domains for domain-to-domain translation

    Parameters:
        d_dim (int): Dimensionality of domain embeddings
        cond_label (str): Label used to access the conditional embedding (idx 0)
        uncond_label (str): Label used to access the unconditional embedding (idx 1)
        init_std (float): Initial embedding standard deviation
    """

    def __init__(
        self,
        d_dim: int,
        cond_idx: int = 0,
        uncond_idx: int = 1,
        init_std: float = 0.02,
    ):
        super().__init__()
        # -------------------------
        # Define cond/uncond embedding indices
        # -------------------------
        # -- indices: 0 = cond (target domain), 1 = uncond (source domain)
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

    def get_pair(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a batch of cond/uncond embeddings each (B, d_dim)

        Args:
            batch_size (int): Training / inference batch size

        Returns:
            cond_dim (torch.Tensor): Batch of conditional embeddings of shape (B, d_dim)
            uncond_dim (torch.Tensor): Batch of unconditional embeddings of shape (B, d_dim)
        """
        device = self.embed.weight.device

        cond_idx = self.cond_idx.expand(batch_size).to(device)  # (B]
        uncond_idx = self.uncond_idx.expand(batch_size).to(device)  # (B]

        cond_emb = self.embed(cond_idx)  # (B, d_dim)
        uncond_emb = self.embed(uncond_idx)  # (B, d_dim)

        # -- Return embeddings: (B, d_dim), (B, d_dim)
        return cond_emb, uncond_emb
