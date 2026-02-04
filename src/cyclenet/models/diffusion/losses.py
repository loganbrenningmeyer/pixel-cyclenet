import torch
import torch.nn as nn
import torch.nn.functional as F

from cyclenet.models import UNet, CycleNet
from cyclenet.models.diffusion import DiffusionSchedule
from cyclenet.models.diffusion import q_sample, predict_x0_from_eps


# =========================
# UNet Loss
# =========================
def diffusion_loss(
    model: UNet,
    x_0: torch.Tensor,
    t: torch.Tensor,
    d_emb: torch.Tensor,
    sched: DiffusionSchedule,
):
    """


    Args:


    Returns:

    """
    # -------------------------
    # Noise input image / predict noise
    # -------------------------
    eps = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps, sched)

    eps_pred = model(x_t, t, d_emb)

    # -- MSE noise prediction loss
    return F.mse_loss(eps_pred, eps)


# =========================
# CycleNet Loss Functions
# =========================
def cyclenet_loss(
    model: CycleNet,
    x_0: torch.Tensor,
    t: torch.Tensor,
    cond_idx: torch.Tensor,
    uncond_idx: torch.Tensor,
    sched: DiffusionSchedule,
):
    """


    Args:


    Returns:

    """
    # -------------------------
    # Define c_y (x -> y), c_x (y -> x) prompt pairs
    # -------------------------
    c_y = {"cond": cond_idx, "uncond": uncond_idx}
    c_x = {"cond": uncond_idx, "uncond": cond_idx}

    # -------------------------
    # Noise input image
    # -------------------------
    eps_x = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps_x, sched)

    # -------------------------
    # Reconstruction Loss
    # => \mathcal{L}_{x \to x} = \mathbb{E}_{x_0,\epsilon_x} \Vert \epsilon_\theta(x_t,c_x,x_0) - \epsilon_x \Vert_2^2 \\
    # -------------------------
    # -- eps(x_t, c_x, x_0)
    eps_pred_rec = model.forward(x_t, t, c_idx=c_x, c_img=x_0)

    rec_loss = reconstruction_loss(eps_pred_rec, eps_x)

    # -------------------------
    # Cycle Consistency Loss
    # => \mathcal{L}_{x \to y \to x} = \mathbb{E}_{x_0,\epsilon_x,\epsilon_y} \Vert \epsilon_\theta(y_t,c_x,x_0) + \epsilon_\theta(x_t,c_y,x_0) - \epsilon_x - \epsilon_y \Vert_2^2 \\
    # -------------------------
    # -- eps(x_t, c_y, x_0)
    eps_pred_cyc_x = model.forward(x_t, t, c_idx=c_y, c_img=x_0)

    # -- Predict clean y_0 / add noise
    y_0_bar = predict_x0_from_eps(x_t, t, eps_pred_cyc_x, sched)

    eps_y = torch.randn_like(y_0_bar)
    y_t = q_sample(y_0_bar, t, eps_y, sched)

    # -- eps(y_t, c_x, x_0)
    eps_pred_cyc_y = model.forward(y_t, t, c_idx=c_x, c_img=x_0)

    cyc_loss = cycle_consistency_loss(eps_pred_cyc_x, eps_pred_cyc_y, eps_x, eps_y)

    # -------------------------
    # Invariance Loss
    # => \mathcal{L}_{x \to y \to y} = \mathbb{E}_{x_0,\epsilon_x} \Vert \epsilon_\theta(x_t,c_y,x_0) - \epsilon_\theta(x_t,c_y,\bar{y}_0) \Vert_2^2 \\
    # -------------------------
    # -- eps(x_t, c_y, y_0_bar)
    eps_pred_inv_y = model.forward(x_t, t, c_idx=c_y, c_img=y_0_bar)

    inv_loss = invariance_loss(eps_pred_cyc_x, eps_pred_inv_y)

    return {"rec_loss": rec_loss, "cyc_loss": cyc_loss, "inv_loss": inv_loss}


def reconstruction_loss(eps_pred: torch.Tensor, eps: torch.Tensor):
    """
    => \mathcal{L}_{x \to x} = \mathbb{E}_{x_0,\epsilon_x} \Vert \epsilon_\theta(x_t,c_x,x_0) - \epsilon_x \Vert_2^2 \\
    
    Args:
    
    
    Returns:
    
    """
    return F.mse_loss(eps_pred, eps)


def cycle_consistency_loss(
    eps_pred_x: torch.Tensor,
    eps_pred_y: torch.Tensor,
    eps_x: torch.Tensor,
    eps_y: torch.Tensor,
):
    """
    => \mathcal{L}_{x \to y \to x} = \mathbb{E}_{x_0,\epsilon_x,\epsilon_y} \Vert \epsilon_\theta(y_t,c_x,x_0) + \epsilon_\theta(x_t,c_y,x_0) - \epsilon_x - \epsilon_y \Vert_2^2 \\
    
    Args:
    
    
    Returns:
    
    """
    return F.mse_loss((eps_pred_x + eps_pred_y), (eps_x + eps_y))


def invariance_loss(eps_pred_x: torch.Tensor, eps_pred_y: torch.Tensor):
    """
    => \mathcal{L}_{x \to y \to y} = \mathbb{E}_{x_0,\epsilon_x} \Vert \epsilon_\theta(x_t,c_y,x_0) - \epsilon_\theta(x_t,c_y,\bar{y}_0) \Vert_2^2 \\
    
    Args:
    
    
    Returns:
    
    """
    return F.mse_loss(eps_pred_x, eps_pred_y)
