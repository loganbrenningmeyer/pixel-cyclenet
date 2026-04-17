import torch
import torch.nn.functional as F

from cyclenet.models import UNet, CycleNet
from cyclenet.diffusion import DiffusionSchedule
from cyclenet.diffusion import q_sample, x0_from_eps


# =========================
# UNet Loss
# =========================
def unet_loss(
    model: UNet,
    x_0: torch.Tensor,
    t: torch.Tensor,
    d_emb: torch.Tensor,
    sched: DiffusionSchedule,
) -> torch.Tensor:
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
# CycleNet Loss ( Only image conditioning )
# =========================
def cyclenet_loss(
    model: CycleNet,
    x_0: torch.Tensor,
    t: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    sched: DiffusionSchedule,
    invar_unet_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """


    Args:
        model (CycleNet): 
        x_0 (torch.Tensor): 
        t (torch.Tensor): 
        src_idx (torch.Tensor): Array of shape (B,) labeling 1 for samples from the source domain (X)
        tgt_idx (torch.Tensor): Array of shape (B,) labeling 1 for samples from the target domain (Y)
        sched (DiffusionSchedule): 
        recon_weight (float): Reconstruction loss weight
        cycle_weight (float): Cycle loss weight
        consis_weight (float): Consistency loss weight
        invar_weight (float): Invariance loss weight


    Returns:

    """
    # -------------------------
    # Noise input image
    # -------------------------
    eps_x = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps_x, sched)

    # -- Normalize x_0 [0,1] for c_img
    x_0_ctrl = ((x_0 + 1.0) / 2.0).clamp(0.0, 1.0)

    # -------------------------
    # Reconstruction Loss (x -> x)
    # => \mathcal{L}_\text{rec} = \mathbb{E}_{x_0,\epsilon_x} \Vert \epsilon_\theta(x_t,c_{x \to x},x_0) - \epsilon_x \Vert_2^2 \\
    # -------------------------
    # => \epsilon_\theta(x_t, c_{x \to x}, x_0)
    eps_xt_x2x_x0 = model.forward(
        x_t=x_t, 
        t=t, 
        from_idx=src_idx,
        to_idx=src_idx,
        c_img=x_0_ctrl,
    )

    recon_loss = F.mse_loss(eps_xt_x2x_x0, eps_x)

    # -------------------------
    # Cycle Loss
    # => L_\text{cycle} = \mathbb{E}_{x_0,\epsilon_x,\epsilon_y}\Vert \epsilon_\theta(x_t, c_{x \to y}, x_0) + \epsilon_\theta(y_t, c_{y \to x}, \bar y_0) - \epsilon_x - \epsilon_y \Vert_2^2
    #
    # Compute x->y translation eps with disabled UNet gradients
    # => \epsilon_\theta(x_t, c_{x \to y}, x_0)
    # -------------------------
    eps_xt_x2y_x0 = model.forward(
        x_t=x_t, 
        t=t, 
        from_idx=src_idx,
        to_idx=tgt_idx,
        c_img=x_0_ctrl,
        enable_unet_grad=False,
    )

    # -- Predict clean y_0 / detached y_0 for c_img conditioning
    y_0 = x0_from_eps(x_t, t, eps_xt_x2y_x0, sched)
    y_0_cond = ((y_0.detach() + 1.0) / 2.0).clamp(0.0, 1.0)

    # -- Noise y_0 -> detached y_t / non-detached y_t_c
    eps_y = torch.randn_like(y_0)

    y_t = q_sample(y_0.detach(), t, eps_y, sched)
    y_t_c = q_sample(y_0, t, eps_y, sched)

    # => \epsilon_\theta(y_t, c_{y \to x}, \bar y_0)
    eps_yt_y2x_y0 = model.forward(
        x_t=y_t_c, 
        t=t,
        from_idx=tgt_idx,
        to_idx=src_idx,
        c_img=y_0_cond,
    )

    cycle_loss = F.mse_loss((eps_xt_x2y_x0.detach() + eps_yt_y2x_y0), (eps_x + eps_y))

    # -------------------------
    # Consistency Loss
    # => L_\text{consis} = \mathbb{E}_{x_0,\epsilon_x,\epsilon_y}\Vert \epsilon_\theta(x_t, c_{x \to y}, x_0) + \epsilon_\theta(y_t, c_{x \to x}, x_0) - \epsilon_x - \epsilon_y \Vert_2^2
    # -------------------------
    # => \epsilon_\theta(y_t, c_{x \to x}, x_0)
    eps_yt_x2x_x0 = model.forward(
        x_t=y_t,
        t=t,
        from_idx=src_idx,
        to_idx=src_idx,
        c_img=x_0_ctrl,
    )

    consis_loss = F.mse_loss((eps_xt_x2y_x0.detach() + eps_yt_x2x_x0), (eps_x + eps_y))

    # -------------------------
    # Invariance Loss
    # => \mathcal{L}_{x \to y \to y} = \mathbb{E}_{x_0,\epsilon_x} \Vert \epsilon_\theta(x_t,c_y,x_0) - \epsilon_\theta(x_t,c_y,\bar{y}_0) \Vert_2^2 \\
    # -------------------------
    # => \epsilon_\theta(x_t, c_{x \to y}, \bar y_0)

    # -- x2y forward pass with gradients enabled
    if invar_unet_grad:
        eps_xt_x2y_x0_invar = model.forward(
            x_t=x_t, 
            t=t, 
            from_idx=src_idx,
            to_idx=tgt_idx,
            c_img=x_0_ctrl,
            enable_unet_grad=True,     # allow forward pass grads to invariance loss
        )
    # -- If no U-Net grad, use previous no grad pass
    else:
        eps_xt_x2y_x0_invar = eps_xt_x2y_x0

    eps_xt_y2y_y0 = model.forward(
        x_t=x_t,
        t=t,
        from_idx=tgt_idx,
        to_idx=tgt_idx,
        c_img=y_0_cond,
    )

    invar_loss = F.mse_loss(eps_xt_x2y_x0_invar, eps_xt_y2y_y0.detach())

    return {
        "recon": recon_loss,
        "cycle": cycle_loss,
        "consis": consis_loss,
        "invar": invar_loss,
    }


# =========================
# CycleNet Loss ( Image + Segmentation mask conditioning )
# =========================
def build_seg_condition(img: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """
    Args:
        img: (B, 3, H, W) in [-1, 1]
        seg: (B, Cseg, H, W) in {0, 1}

    Returns:
        cond: (B, 3 + Cseg, H, W)
    """
    img_norm = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
    cond = torch.cat([img_norm, seg.float()], dim=1)    # (B, 3 + Cseg, H, W)
    return cond


def cyclenet_loss_seg(
    model: CycleNet,
    x_0: torch.Tensor,
    seg_0: torch.Tensor,
    t: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    sched: DiffusionSchedule,
    invar_unet_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """


    Args:
        model (CycleNet): 
        x_0 (torch.Tensor): 
        seg_0 (torch.Tensor): 
        t (torch.Tensor): 
        src_idx (torch.Tensor): Array of shape (B,) labeling 1 for samples from the source domain (X)
        tgt_idx (torch.Tensor): Array of shape (B,) labeling 1 for samples from the target domain (Y)
        sched (DiffusionSchedule): 
        recon_weight (float): Reconstruction loss weight
        cycle_weight (float): Cycle loss weight
        consis_weight (float): Consistency loss weight
        invar_weight (float): Invariance loss weight


    Returns:

    """
    # -------------------------
    # Noise input image
    # -------------------------
    eps_x = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps_x, sched)

    # -- Build x_0 + seg_0 conditioning image
    c_x0_seg = build_seg_condition(x_0, seg_0)

    # -------------------------
    # Reconstruction Loss (x -> x)
    # => \mathcal{L}_\text{rec} = \mathbb{E}_{x_0,\epsilon_x} \Vert \epsilon_\theta(x_t,c_{x \to x},x_0) - \epsilon_x \Vert_2^2 \\
    # -------------------------
    # => \epsilon_\theta(x_t, c_{x \to x}, x_0)
    eps_xt_x2x_x0 = model.forward(
        x_t=x_t, 
        t=t, 
        from_idx=src_idx,
        to_idx=src_idx,
        c_img=c_x0_seg,
    )

    recon_loss = F.mse_loss(eps_xt_x2x_x0, eps_x)

    # -------------------------
    # Cycle Loss
    # => L_\text{cycle} = \mathbb{E}_{x_0,\epsilon_x,\epsilon_y}\Vert \epsilon_\theta(x_t, c_{x \to y}, x_0) + \epsilon_\theta(y_t, c_{y \to x}, \bar y_0) - \epsilon_x - \epsilon_y \Vert_2^2
    #
    # Compute x->y translation eps with disabled UNet gradients
    # => \epsilon_\theta(x_t, c_{x \to y}, x_0)
    # -------------------------
    eps_xt_x2y_x0 = model.forward(
        x_t=x_t, 
        t=t, 
        from_idx=src_idx,
        to_idx=tgt_idx,
        c_img=c_x0_seg,
        enable_unet_grad=False,
    )

    # -- Predict clean y_0 / detached y_0 for c_img conditioning
    y_0 = x0_from_eps(x_t, t, eps_xt_x2y_x0, sched)
    c_y0_seg = build_seg_condition(y_0.detach(), seg_0)

    # -- Noise y_0 -> detached y_t / non-detached y_t_c
    eps_y = torch.randn_like(y_0)

    y_t = q_sample(y_0.detach(), t, eps_y, sched)
    y_t_c = q_sample(y_0, t, eps_y, sched)

    # => \epsilon_\theta(y_t, c_{y \to x}, \bar y_0)
    eps_yt_y2x_y0 = model.forward(
        x_t=y_t_c, 
        t=t,
        from_idx=tgt_idx,
        to_idx=src_idx,
        c_img=c_y0_seg,
    )

    cycle_loss = F.mse_loss((eps_xt_x2y_x0.detach() + eps_yt_y2x_y0), (eps_x + eps_y))

    # -------------------------
    # Consistency Loss
    # => L_\text{consis} = \mathbb{E}_{x_0,\epsilon_x,\epsilon_y}\Vert \epsilon_\theta(x_t, c_{x \to y}, x_0) + \epsilon_\theta(y_t, c_{x \to x}, x_0) - \epsilon_x - \epsilon_y \Vert_2^2
    # -------------------------
    # => \epsilon_\theta(y_t, c_{x \to x}, x_0)
    eps_yt_x2x_x0 = model.forward(
        x_t=y_t,
        t=t,
        from_idx=src_idx,
        to_idx=src_idx,
        c_img=c_x0_seg,
    )

    consis_loss = F.mse_loss((eps_xt_x2y_x0.detach() + eps_yt_x2x_x0), (eps_x + eps_y))

    # -------------------------
    # Invariance Loss
    # => \mathcal{L}_{x \to y \to y} = \mathbb{E}_{x_0,\epsilon_x} \Vert \epsilon_\theta(x_t,c_y,x_0) - \epsilon_\theta(x_t,c_y,\bar{y}_0) \Vert_2^2 \\
    # -------------------------
    # => \epsilon_\theta(x_t, c_{x \to y}, \bar y_0)

    # -- x2y forward pass with gradients enabled
    if invar_unet_grad:
        eps_xt_x2y_x0_invar = model.forward(
            x_t=x_t,
            t=t,
            from_idx=src_idx,
            to_idx=tgt_idx,
            c_img=c_x0_seg,
            enable_unet_grad=True,
        )
    # -- If no U-Net grad, use previous no grad pass
    else:
        eps_xt_x2y_x0_invar = eps_xt_x2y_x0

    eps_xt_y2y_y0 = model.forward(
        x_t=x_t,
        t=t,
        from_idx=tgt_idx,
        to_idx=tgt_idx,
        c_img=c_y0_seg,
    )

    invar_loss = F.mse_loss(eps_xt_x2y_x0_invar, eps_xt_y2y_y0.detach())

    return {
        "recon": recon_loss,
        "cycle": cycle_loss,
        "consis": consis_loss,
        "invar": invar_loss,
    }



def unwrap(m):
    """
    Unwrap DDP module if it is wrapped
    """
    return m.module if hasattr(m, "module") else m
