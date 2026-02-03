import torch


def linear_beta_schedule(T: int, beta_start: float, beta_end: float):
    """


    Args:


    Returns:

    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008):
    """


    Args:


    Returns:

    """
    t = torch.arange(0, T + 1)
    # => f(t) = \cos^2\Bigl(\frac{t/T+s}{1+s} \cdot \frac{\pi}{2}\Bigr)
    f_t = torch.cos((t / T + s) / (1 + s) * torch.pi / 2) ** 2
    # => \bar\alpha_t = \frac{f(t)}{f(0)}
    alpha_bars = f_t / f_t[0]
    # => \beta_t = 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}
    betas = 1.0 - alpha_bars[1:] / alpha_bars[0:T]
    # -- Clamp betas to avoid 0 and 1
    betas = betas.clamp(min=1e-12, max=0.999)
    return betas
