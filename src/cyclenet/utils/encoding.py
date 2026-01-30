import math
import torch

def sinusoidal_encoding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes sinusoidal time encodings for t timesteps with dim dimensionality.
    
    Args:
        t (torch.Tensor): Tensor of shape (B,) containing integer timesteps
        dim (int): Dimensionality of the time embedding
    
    Returns:
        t_emb (torch.Tensor): Tensor of shape (B, dim) containing sinusoidal time embeddings
    """
    half_dim = dim // 2
    
    # -------------------------
    # => k = 0,1,2,\ldots,\frac{d}{2}-1
    # -------------------------
    k_vals = torch.arange(half_dim, device=t.device)

    # -------------------------
    # => \omega_k t = \frac{t}{10000^{2k/d}} = \exp(-2k/d \cdot \ln(10000))
    # -------------------------
    freqs = torch.exp(-2*k_vals/dim * math.log(10000))  # (dim/2,)
    freqs = freqs[None, :] * t[:, None].float()         # (B, dim/2)

    # -------------------------
    # => \text{PE}(t) = \{\sin(\omega_0 t),\sin(\omega_1 t),\ldots,\sin(\omega_{d/2-1}t),\cos(\omega_0 t),\cos(\omega_1 t),\ldots,\cos(\omega_{d/2-1}t)\}
    # -------------------------
    t_emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=1)  # (B, dim)

    return t_emb