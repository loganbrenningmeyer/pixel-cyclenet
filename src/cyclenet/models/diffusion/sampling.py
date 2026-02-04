import torch

from cyclenet.models.diffusion import DiffusionSchedule
from cyclenet.models import CycleNet


def predict_cyclenet_cfg(
    model: CycleNet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    c_img: torch.Tensor,
    cond_idx: torch.Tensor,
    uncond_idx: torch.Tensor,
    w: float = 1.0,
) -> torch.Tensor:
    """


    Args:


    Returns:

    """
    c_idx_cond   = {"cond": cond_idx,   "uncond": uncond_idx}
    c_idx_uncond = {"cond": uncond_idx, "uncond": uncond_idx}

    # -------------------------
    # Perform conditional / unconditional passes
    # -------------------------
    eps_cond   = model.forward(x_t, t, c_img, c_idx=c_idx_cond)
    eps_uncond = model.forward(x_t, t, c_img, c_idx=c_idx_uncond)

    # -------------------------
    # CFG weighted sum
    # -------------------------
    return eps_uncond + w * (eps_cond - eps_uncond)


def q_sample(
    x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, sched: DiffusionSchedule
) -> torch.Tensor:
    """
    Computes forward q sample (x_t), noising the clean sample x0 to timestep t
    => x_t = \sqrt{\bar \alpha_t}x_0 + \sqrt{1 - \bar \alpha_t}\epsilon

    Args:


    Returns:

    """
    return sched.sqrt_alpha_bars[t] * x_0 + sched.sqrt_one_minus_alpha_bars[t] * eps


def predict_x0_from_eps(
    x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, sched: DiffusionSchedule
) -> torch.Tensor:
    """
    Inverts the forward q sample equation to solve for x0 given the eps noise prediction
    => x_0 = \frac{1}{\sqrt{\bar \alpha_t}}x_t - \frac{\sqrt{1 - \bar \alpha_t}}{\sqrt{\bar \alpha_t}}\epsilon

    Args:


    Returns:

    """
    return (1.0 / sched.sqrt_alpha_bars[t]) * x_t - (
        sched.sqrt_one_minus_alpha_bars[t] / sched.sqrt_alpha_bars[t]
    ) * eps


def p_mean_variance(
    x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, sched: DiffusionSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """


    Args:


    Returns:

    """
    # -------------------------
    # Posterior mean
    # => \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon_\theta(x_t,t)\Bigr)
    # -------------------------
    mu = (1.0 / sched.alphas[t]) * (
        x_t - (sched.betas[t] / sched.sqrt_one_minus_alpha_bars[t]) * eps
    )

    # -------------------------
    # Posterior variance
    # => \sigma_t^2 = \beta_t \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}
    # -------------------------
    var = (
        sched.betas[t]
        * (sched.one_minus_alpha_bars_prev)
        / (sched.one_minus_alpha_bars)
    )

    return mu, var
