import torch
from tqdm import tqdm

from cyclenet.diffusion import DiffusionSchedule, expand
from cyclenet.models import UNet, CycleNet


@torch.no_grad()
def cyclenet_ddpm_step(
    model: CycleNet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    c_img: torch.Tensor,
    sched: DiffusionSchedule,
    w: float = 1.0,
) -> torch.Tensor:
    """
    Performs a single DDPM denoising step for `CycleNet` model, translating in the direction
    of "uncond" -> "cond" as defined by `c_idx_cond` with CFG weighted by `w`

    Args:
        model (CycleNet): CycleNet model used for translation
        x_t (torch.Tensor): Noised source image (B, C, H, W)
        t (torch.Tensor): Current denoising timestep batch (B,)
        c_img (torch.Tensor): Conditioning image for ControlNet in range [0,1] and shape (B, C, H, W)
        sched (DiffusionSchedule): Object containing diffusion schedule information
        w (float): Classifier-free guidance weight

    Returns:
        (torch.Tensor): Denoised previous sample x_t-1 (B, C, H, W)
    """
    # -------------------------
    # Classifier-Free Guidance: Predict noise
    # -------------------------
    eps_pred = predict_cyclenet_cfg(model, x_t, t, src_idx, tgt_idx, c_img, w)

    # -------------------------
    # Predict x_t-1 from eps_pred
    # -------------------------
    return ddpm_x_prev_from_eps(x_t, t, eps_pred, sched)


@torch.no_grad()
def cyclenet_ddpm_loop(
    model: CycleNet,
    x_src: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    c_img: torch.Tensor,
    sched: DiffusionSchedule,
    w: float = 1.0,
    strength: float = 0.5,
):
    """
    Performs full DDPM translation for `CycleNet` model, translating in the direction
    of "uncond" -> "cond" as defined by `c_idx_cond` with CFG weighted by `w`

    Args:
        model (CycleNet): CycleNet model used for translation
        x_src (torch.Tensor): Source (uncond) image to be translated to target (cond) domain (B, C, H, W)
        c_img (torch.Tensor): Conditioning image for ControlNet in range [0,1] and shape (B, C, H, W)
        sched (DiffusionSchedule): Object containing diffusion schedule information
        w (float): Classifier-free guidance weight
        strength (float): Proportion of DDPM steps to noise source before denoising during translation

    Returns:
        (torch.Tensor): Translated image in target (cond) domain (B, C, H, W)
    """
    B = x_src.shape[0]
    device = x_src.device

    # -------------------------
    # Noise x_src based on strength
    # -------------------------
    t_steps = ddpm_steps_from_strength(sched, strength)
    t_noise = torch.full((B,), t_steps[-1], device=device, dtype=torch.long)

    eps = torch.randn_like(x_src)
    x_t = q_sample(x_src, t_noise, eps, sched) if strength > 0 else x_src

    # -------------------------
    # Iteratively denoise: t_i = T-1, ..., 0
    # -------------------------
    for t_i in tqdm(reversed(t_steps)):
        # -------------------------
        # Create timestep batch
        # -------------------------
        t = torch.full((B,), t_i, device=device, dtype=torch.long)

        # -------------------------
        # Perform DDPM step
        # -------------------------
        x_t = cyclenet_ddpm_step(model, x_t, t, src_idx, tgt_idx, c_img, sched, w)

    return x_t


@torch.no_grad()
def cyclenet_ddim_step(
    model: CycleNet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    c_img: torch.Tensor,
    sched: DiffusionSchedule,
    w: float = 1.0,
    eta: float = 0.0,
):
    """
    Performs a single DDIM denoising step for `CycleNet` model, translating in the direction
    of "uncond" -> "cond" as defined by `c_idx_cond` with CFG weighted by `w`

    Args:
        model (CycleNet): CycleNet model used for translation
        x_t (torch.Tensor): Noised source image (B, C, H, W)
        t (torch.Tensor): Current denoising timestep batch (B,)
        t_prev (torch.Tensor): Previous DDIM timestep batch (B,)
        c_img (torch.Tensor): Conditioning image for ControlNet in range [0,1] and shape (B, C, H, W)
        sched (DiffusionSchedule): Object containing diffusion schedule information
        w (float): Classifier-free guidance weight
        eta (float): DDIM stochastic noise weight (`eta == 0` is deterministic)

    Returns:
        (torch.Tensor): Denoised sample x_prev from previous DDIM timestep (B, C, H, W)
    """
    # -------------------------
    # Classifier-Free Guidance: Predict noise
    # -------------------------
    eps_pred = predict_cyclenet_cfg(model, x_t, t, src_idx, tgt_idx, c_img, w)

    # -------------------------
    # Compute x_prev (prev DDIM step)
    # -------------------------
    return ddim_x_prev_from_eps(x_t, t, t_prev, eps_pred, sched, eta)


@torch.no_grad()
def cyclenet_ddim_loop(
    model: CycleNet,
    x_src: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    c_img: torch.Tensor,
    sched: DiffusionSchedule,
    w: float = 1.0,
    strength: float = 0.5,
    num_steps: int = 100,
    eta: float = 0.0,
):
    """
    Performs full DDIM translation for `CycleNet` model, translating in the direction
    of "uncond" -> "cond" as defined by `c_idx_cond` with CFG weighted by `w`

    Args:
        model (CycleNet): CycleNet model used for translation
        x_src (torch.Tensor): Source (uncond) image to be translated to target (cond) domain (B, C, H, W)
        c_img (torch.Tensor): Conditioning image for ControlNet in range [0,1] and shape (B, C, H, W)
        sched (DiffusionSchedule): Object containing diffusion schedule information
        w (float): Classifier-free guidance weight
        strength (float): Proportion of DDPM steps to noise source before denoising during translation
        num_steps (int): Number of DDIM steps to uniformly divide the full T timesteps
        eta (float): DDIM stochastic noise weight (`eta == 0` is deterministic)

    Returns:
        (torch.Tensor): Translated image in target (cond) domain (B, C, H, W)
    """
    B = x_src.shape[0]
    device = x_src.device

    # -------------------------
    # Noise x_src based on strength
    # -------------------------
    t_steps = ddim_steps_from_strength(sched, num_steps, strength)
    t_noise = torch.full((B,), t_steps[-1], device=device, dtype=torch.long)

    eps = torch.randn_like(x_src)
    x_t = q_sample(x_src, t_noise, eps, sched) if strength > 0 else x_src

    # -------------------------
    # Uniformly sample reverse DDIM steps from t_noise
    # -------------------------
    t_steps_rev = list(reversed(t_steps))

    # -------------------------
    # Iteratively denoise: t = t_noise, ..., 0
    # -------------------------
    for i, t_i in tqdm(enumerate(t_steps_rev[:-1])):
        # -------------------------
        # Create t / t_prev batch
        # -------------------------
        t = torch.full((B,), t_i, device=device, dtype=torch.long)

        t_prev_i = t_steps_rev[i + 1]
        t_prev = torch.full((B,), t_prev_i, device=device, dtype=torch.long)

        # -------------------------
        # Perform DDIM step
        # -------------------------
        x_t = cyclenet_ddim_step(
            model, x_t, t, t_prev, src_idx, tgt_idx, c_img, sched, w, eta
        )

    return x_t


@torch.no_grad()
def unet_ddpm_step(
    model: UNet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    d_emb: torch.Tensor,
    sched: DiffusionSchedule,
) -> torch.Tensor:
    """
    Performs a single DDPM denoising step for `UNet` model, denoising to the target domain
    as guided by `d_emb`.

    Args:
        model (UNet): UNet model used for denoising
        x_t (torch.Tensor): Noised image (B, C, H, W)
        t (torch.Tensor): Current denoising timestep batch (B,)
        d_emb (torch.Tensor): Embedding of target domain (d_dim,)
        sched (DiffusionSchedule): Object containing diffusion schedule information

    Returns:
        (torch.Tensor): Denoised previous sample x_t-1 (B, C, H, W)
    """
    # -------------------------
    # Predict noise
    # -------------------------
    eps_pred = model.forward(x_t, t, d_emb)

    # -------------------------
    # Predict x_t-1 from eps_pred
    # -------------------------
    return ddpm_x_prev_from_eps(x_t, t, eps_pred, sched)


@torch.no_grad()
def unet_ddpm_loop(
    model: UNet,
    x_ref: torch.Tensor,
    d_emb: torch.Tensor,
    sched: DiffusionSchedule,
) -> torch.Tensor:
    """
    Performs full DDPM generation for `UNet` model, denoising to the target domain
    as guided by `d_emb`.

    Args:
        model (UNet): UNet model used for denoising
        x_ref (torch.Tensor): Empty tensor to reference image shape, device, and dtype (B, C, H, W)
        d_emb (torch.Tensor): Embedding of target domain (d_dim,)
        sched (DiffusionSchedule): Object containing diffusion schedule information

    Returns:
        (torch.Tensor): Generated image in target domain (B, C, H, W)
    """
    B = x_ref.shape[0]
    device = x_ref.device

    # -------------------------
    # Sample Gaussian noise (x_T)
    # -------------------------
    x_t = torch.randn_like(x_ref)

    # -------------------------
    # Iteratively denoise: t_i = T-1, ..., 0
    # -------------------------
    for t_i in tqdm(reversed(range(sched.T))):
        # -------------------------
        # Create timestep batch
        # -------------------------
        t = torch.full((B,), t_i, device=device, dtype=torch.long)

        # -------------------------
        # Perform DDPM step
        # -------------------------
        x_t = unet_ddpm_step(model, x_t, t, d_emb, sched)

    return x_t


@torch.no_grad()
def unet_ddim_step(
    model: UNet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    d_emb: torch.Tensor,
    sched: DiffusionSchedule,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    Performs a single DDIM denoising step for `UNet` model, denoising to the target domain
    as guided by `d_emb`.

    Args:
        model (UNet): UNet model used for denoising
        x_t (torch.Tensor): Noised image (B, C, H, W)
        t (torch.Tensor): Current denoising timestep batch (B,)
        t_prev (torch.Tensor): Previous DDIM timestep batch (B,)
        d_emb (torch.Tensor): Embedding of target domain (d_dim,)
        sched (DiffusionSchedule): Object containing diffusion schedule information
        eta (float): DDIM stochastic noise weight (`eta == 0` is deterministic)

    Returns:
        (torch.Tensor): Denoised previous sample x_t-1 (B, C, H, W)
    """
    # -------------------------
    # Predict noise
    # -------------------------
    eps_pred = model.forward(x_t, t, d_emb)

    # -------------------------
    # Compute x_prev (prev DDIM step)
    # -------------------------
    return ddim_x_prev_from_eps(x_t, t, t_prev, eps_pred, sched, eta)


@torch.no_grad()
def unet_ddim_loop(
    model: UNet,
    x_ref: torch.Tensor,
    d_emb: torch.Tensor,
    sched: DiffusionSchedule,
    num_steps: int = 100,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    Performs full DDIM generation for `UNet` model, denoising to the target domain
    as guided by `d_emb`.

    Args:
        model (UNet): UNet model used for denoising
        x_ref (torch.Tensor): Empty tensor to reference image shape, device, and dtype (B, C, H, W)
        d_emb (torch.Tensor): Embedding of target domain (d_dim,)
        sched (DiffusionSchedule): Object containing diffusion schedule information
        num_steps (int): Number of DDIM steps to uniformly divide the full T timesteps
        eta (float): DDIM stochastic noise weight (`eta == 0` is deterministic)

    Returns:
        (torch.Tensor): Generated image in target domain (B, C, H, W)
    """
    B = x_ref.shape[0]
    device = x_ref.device

    # -------------------------
    # Sample Gaussian noise (x_T)
    # -------------------------
    x_t = torch.randn_like(x_ref)

    # -------------------------
    # Uniformly sample reverse DDIM steps
    # -------------------------
    t_steps_rev = list(reversed(sched.ddim_timesteps(num_steps)))

    # -------------------------
    # Iteratively denoise: t = T-1, ..., 0
    # -------------------------
    for i, t_i in tqdm(enumerate(t_steps_rev[:-1])):
        # -------------------------
        # Create t / t_prev batch
        # -------------------------
        t = torch.full((B,), t_i, device=device, dtype=torch.long)

        t_prev_i = t_steps_rev[i + 1]
        t_prev = torch.full((B,), t_prev_i, device=device, dtype=torch.long)

        # -------------------------
        # Perform DDIM step
        # -------------------------
        x_t = unet_ddim_step(model, x_t, t, t_prev, d_emb, sched, eta)

    return x_t


def ddim_sigma(
    t: torch.Tensor, t_prev: torch.Tensor, sched: DiffusionSchedule, eta: float = 0.0
) -> torch.Tensor:
    """
    Computes the standard deviation of the injected DDIM stochastic noise.

    Args:
        t (torch.Tensor): Current denoising timestep batch (B,)
        t_prev (torch.Tensor): Previous DDIM timestep batch (B,)
        sched (DiffusionSchedule): Object containing diffusion schedule information
        eta (float): DDIM stochastic noise weight (`eta == 0` is deterministic)

    Returns:
        (torch.Tensor): Standard deviation of stochastic noise (B, 1, 1, 1)
    """
    return (
        eta
        * torch.sqrt(
            expand(sched.one_minus_alpha_bars[t_prev])
            / expand(sched.one_minus_alpha_bars[t])
        )
        * torch.sqrt(
            1.0 - expand(sched.alpha_bars[t]) / expand(sched.alpha_bars[t_prev])
        )
    )


def predict_cyclenet_cfg(
    model: CycleNet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    c_img: torch.Tensor,
    w: float = 1.0,
) -> torch.Tensor:
    """
    Returns the CycleNet noise prediction using CFG given two passes:
        - 1. Conditional Pass:   "cond" -> UNet,   "uncond" -> ControlNet
        - 2. Unconditional Pass: "uncond" -> UNet, "uncond" -> ControlNet

    Args:
        model (CycleNet): CycleNet model used for translation
        x_t (torch.Tensor): Noised source image (B, C, H, W)
        t (torch.Tensor): Current denoising timestep batch (B,)
        c_img (torch.Tensor): Conditioning image for ControlNet (B, C, H, W)
        w (float): Classifier-free guidance weight

    Returns:
        (torch.Tensor): Predicted noise after CFG weighting (B, C, H, W)
    """
    # -------------------------
    # Perform conditional / unconditional passes
    # -------------------------
    # -- [Conditional]: UNet Backbone (target), ControlNet (source)
    eps_cond = model.forward(
        x_t=x_t, 
        t=t,
        from_idx=src_idx,
        to_idx=tgt_idx,
        c_img=c_img
    )

    # -- [Unconditional]: UNet Backbone (source), ControlNet (source)
    eps_uncond = model.forward(
        x_t=x_t, 
        t=t, 
        from_idx=src_idx,
        to_idx=src_idx,
        c_img=c_img
    )

    # -------------------------
    # CFG weighted sum
    # -------------------------
    return eps_uncond + w * (eps_cond - eps_uncond)


def ddpm_x_prev_from_eps(
    x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor, sched: DiffusionSchedule
) -> torch.Tensor:
    """
    Computes the previous x_t-1 given the eps noise prediction

    Args:
        x_t (torch.Tensor): Noised source image (B, C, H, W)
        t (torch.Tensor): Current denoising timestep batch (B,)
        eps_pred (torch.Tensor): Predicted noise (B, C, H, W)
        sched (DiffusionSchedule): Object containing diffusion schedule information

    Returns:
        (torch.Tensor): Denoised previous sample x_t-1 (B, C, H, W)
    """
    # -------------------------
    # Compute posterior mean / variance
    # -------------------------
    mu, var = p_mean_variance(x_t, t, eps_pred, sched)

    # -------------------------
    # Sample noise / compute x_t-1
    # => x_{t-1} = \mu_\theta(x_t,t) + \sqrt{\tilde\beta_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)\quad \text{if}\ t > 0\ \text{else}\ \epsilon=0
    # -------------------------
    eps = torch.randn_like(x_t) if t[0].item() > 0 else torch.zeros_like(x_t)

    return mu + torch.sqrt(var) * eps


def ddim_x_prev_from_eps(
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    eps_pred: torch.Tensor,
    sched: DiffusionSchedule,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    Computes the previous x_t (prev DDIM step) given the eps noise prediction

    Args:
        x_t (torch.Tensor): Noised source image (B, C, H, W)
        t (torch.Tensor): Current denoising timestep batch (B,)
        t_prev (torch.Tensor): Previous DDIM timestep batch (B,)
        eps_pred (torch.Tensor): Predicted noise (B, C, H, W)
        sched (DiffusionSchedule): Object containing diffusion schedule information
        eta (float): DDIM stochastic noise weight (`eta == 0` is deterministic)

    Returns:
        (torch.Tensor): Denoised sample x_prev from previous DDIM timestep (B, C, H, W)
    """
    # -------------------------
    # Compute clean x_0
    # -------------------------
    x_0 = x0_from_eps(x_t, t, eps_pred, sched)

    # -------------------------
    # Compute std of added DDIM stochastic noise
    # -------------------------
    sigma = ddim_sigma(t, t_prev, sched, eta)

    # -------------------------
    # Compute x_prev
    # -------------------------
    # -- Inject stochastic noise
    z = torch.randn_like(x_t) if eta != 0 else torch.zeros_like(x_t)

    return (
        expand(sched.sqrt_alpha_bars[t_prev]) * x_0
        + torch.sqrt(expand(sched.one_minus_alpha_bars[t_prev]) - sigma**2) * eps_pred
        + sigma * z
    )


def q_sample(
    x_0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, sched: DiffusionSchedule
) -> torch.Tensor:
    """
    Computes forward q sample (x_t), noising the clean sample x0 to timestep t
    => x_t = \sqrt{\bar \alpha_t}x_0 + \sqrt{1 - \bar \alpha_t}\epsilon

    Args:


    Returns:

    """
    return (
        expand(sched.sqrt_alpha_bars[t]) * x_0
        + expand(sched.sqrt_one_minus_alpha_bars[t]) * eps
    )


def x0_from_eps(
    x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor, sched: DiffusionSchedule
) -> torch.Tensor:
    """
    Inverts the forward q sample equation to solve for x0 given the eps noise prediction
    => x_0 = \frac{1}{\sqrt{\bar \alpha_t}}x_t - \frac{\sqrt{1 - \bar \alpha_t}}{\sqrt{\bar \alpha_t}}\epsilon

    Args:


    Returns:

    """
    return (1.0 / expand(sched.sqrt_alpha_bars[t])) * x_t - (
        expand(sched.sqrt_one_minus_alpha_bars[t]) / expand(sched.sqrt_alpha_bars[t])
    ) * eps_pred


def p_mean_variance(
    x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor, sched: DiffusionSchedule
) -> tuple[torch.Tensor, torch.Tensor]:
    """


    Args:


    Returns:

    """
    # -------------------------
    # Posterior mean
    # => \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\Bigl(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon_\theta(x_t,t)\Bigr)
    # -------------------------
    mu = (1.0 / expand(sched.sqrt_alphas[t])) * (
        x_t
        - (expand(sched.betas[t]) / expand(sched.sqrt_one_minus_alpha_bars[t]))
        * eps_pred
    )

    # -------------------------
    # Posterior variance
    # => \sigma_t^2 = \beta_t \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}
    # -------------------------
    var = (
        expand(sched.betas[t])
        * expand(sched.one_minus_alpha_bars_prev[t])
        / expand(sched.one_minus_alpha_bars[t])
    )

    return mu, var


def ddpm_steps_from_strength(sched: DiffusionSchedule, strength: float) -> list[int]:
    """


    Args:


    Returns:

    """
    # -- Clamp strength to [0, 1]
    strength = max(0.0, min(1.0, strength))

    t_step_max = min(int(strength * (sched.T - 1)), int(sched.T - 1))
    return list(range(t_step_max + 1))


def ddim_steps_from_strength(
    sched: DiffusionSchedule, num_steps: int, strength: float
) -> list[int]:
    """


    Args:


    Returns:

    """
    # -- Clamp strength to [0, 1]
    strength = max(0.0, min(1.0, strength))

    t_steps = sched.ddim_timesteps(num_steps)
    # -- Determine max DDIM step based on strength
    t_idx_max = int(strength * (num_steps - 1))
    return t_steps[: t_idx_max + 1].tolist()
