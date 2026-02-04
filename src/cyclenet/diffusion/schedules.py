import torch


class DiffusionSchedule:
    def __init__(self, schedule: str, T: int, beta_start: float, beta_end: float, device: torch.device, s: float = 0.008):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule = schedule

        # -------------------------
        # Create beta schedule
        # -------------------------
        if schedule == "linear":
            self.betas = linear_beta_schedule(T, beta_start, beta_end).to(device)
        elif schedule == "cosine": 
            self.betas = cosine_beta_schedule(T, s).to(device)
        else:
            raise ValueError("Beta schedule must be 'linear' or 'cosine'.")
        
        # -------------------------
        # Precompute alphas / alpha_bars
        # -------------------------
        self.alphas = get_alphas(self.betas)
        self.sqrt_alphas = torch.sqrt(self.alphas)

        self.alpha_bars = get_alpha_bars(self.betas)
        self.one_minus_alpha_bars = 1.0 - self.alpha_bars

        self.alpha_bars_prev = torch.cat([
            self.alpha_bars.new_ones(1), 
            self.alpha_bars[:-1]
        ])
        self.one_minus_alpha_bars_prev = 1.0 - self.alpha_bars_prev
        self.sqrt_alpha_bars_prev = torch.sqrt(self.alpha_bars_prev)
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def ddim_timesteps(self, num_steps: int):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        return torch.linspace(0, self.T-1, num_steps, dtype=torch.long)
    

def expand(v: torch.Tensor) -> torch.Tensor:
    """
    Reshapes tensor of shape (B,) to (B,1,1,1)
    """
    return v[:, None, None, None]


def linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """

    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """

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


def get_alphas(betas: torch.Tensor) -> torch.Tensor:
    """
    Converts beta schedule to alphas
    => \alpha_t = 1 - \beta_t
    """
    return 1.0 - betas


def get_alpha_bars(betas: torch.Tensor) -> torch.Tensor:
    """
    Converts beta schedule to alpha bars
    => \bar \alpha_t = \prod_{s=1}^t \alpha_s
    """
    alphas = get_alphas(betas)
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars