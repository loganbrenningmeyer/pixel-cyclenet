import torch
import torch.nn.functional as F


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image tensor to luminance.

    Args:
        x: [B, 3, H, W] in [-1, 1] or [0, 1]

    Returns:
        y: [B, H, W]
    """
    if x.ndim != 4 or x.size(1) != 3:
        raise ValueError(f"Expected [B,3,H,W], got {tuple(x.shape)}")

    r = x[:, 0]
    g = x[:, 1]
    b = x[:, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def fftshift2(x: torch.Tensor) -> torch.Tensor:
    """
    Shift zero-frequency component to the center for the last 2 dims.
    """
    shift_y = x.size(-2) // 2
    shift_x = x.size(-1) // 2
    return torch.roll(torch.roll(x, shifts=shift_y, dims=-2), shifts=shift_x, dims=-1)


def sample_random_patches(
    imgs: torch.Tensor,
    patch_size: int,
    patches_per_image: int,
) -> torch.Tensor:
    """
    Sample random square patches from a batch of images.

    Args:
        imgs: [B, H, W]
        patch_size: int
        patches_per_image: int

    Returns:
        patches: [B * patches_per_image, patch_size, patch_size]
    """
    if imgs.ndim != 3:
        raise ValueError(f"Expected [B,H,W], got {tuple(imgs.shape)}")

    B, H, W = imgs.shape
    if H < patch_size or W < patch_size:
        raise ValueError(f"Patch size {patch_size} exceeds image size {H}x{W}")

    patches = []
    max_top = H - patch_size
    max_left = W - patch_size

    for i in range(B):
        for _ in range(patches_per_image):
            top = torch.randint(0, max_top + 1, (1,), device=imgs.device).item()
            left = torch.randint(0, max_left + 1, (1,), device=imgs.device).item()
            patches.append(imgs[i, top:top + patch_size, left:left + patch_size])

    return torch.stack(patches, dim=0)


def radial_power_spectrum_batch(
    patches: torch.Tensor,
    *,
    remove_mean: bool = True,
    normalize_std: bool = True,
    window: str = "hann",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    patches: [B, N, N]
    returns:
        radial: [B, R]
        r: [R]
    """
    if patches.ndim != 3:
        raise ValueError(f"Expected [B,N,N], got {tuple(patches.shape)}")

    B, N, M = patches.shape
    if N != M:
        raise ValueError(f"Expected square patches, got {N}x{M}")

    x = patches.float()

    if remove_mean:
        x = x - x.mean(dim=(-2, -1), keepdim=True)

    if normalize_std:
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
        x = x / (std + eps)

    if window is None:
        xw = x
    elif window == "hann":
        w1 = torch.hann_window(N, periodic=False, device=x.device, dtype=x.dtype)
        w2d = torch.outer(w1, w1)
        xw = x * w2d
    else:
        raise ValueError("window must be 'hann' or None")

    X = torch.fft.fft2(xw, dim=(-2, -1))
    X = torch.fft.fftshift(X, dim=(-2, -1))
    P = X.real.square() + X.imag.square()   # [B,N,N]

    c = N // 2
    yy = torch.arange(N, device=x.device) - c
    xx = torch.arange(N, device=x.device) - c
    Y, Xg = torch.meshgrid(yy, xx, indexing="ij")
    R = torch.sqrt(Xg.float().square() + Y.float().square())
    rbin = torch.floor(R).long()            # [N,N]

    nbins = int(rbin.max().item()) + 1
    r = torch.arange(nbins, device=x.device)

    # Flatten spatial dims
    rflat = rbin.reshape(-1)                # [N*N]
    pflat = P.reshape(B, -1)                # [B, N*N]

    # Counts are constant, no grad needed
    counts = torch.bincount(rflat, minlength=nbins).to(x.dtype)  # [R]

    # Differentiable accumulation
    radial = torch.zeros(B, nbins, device=x.device, dtype=x.dtype)
    radial.scatter_add_(
        dim=1,
        index=rflat.unsqueeze(0).expand(B, -1),
        src=pflat,
    )

    radial = radial / (counts.unsqueeze(0) + eps)
    return radial, r


def spectral_loss(
    translated_rgb: torch.Tensor,
    real_rgb: torch.Tensor,
    *,
    patch_size: int = 64,
    patches_per_image: int = 4,
    remove_mean: bool = True,
    normalize_std: bool = True,
    use_log: bool = True,
    mid_high_emphasis: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Patch-level log-radial-spectrum matching loss between translated images and real images.

    Args:
        translated_rgb: [B,3,H,W] translated outputs, ideally in [-1,1]
        real_rgb: [B,3,H,W] real target-domain images, ideally in [-1,1]
        patch_size: patch size for local spectrum computation
        patches_per_image: number of random patches per image
        remove_mean: remove DC per patch
        normalize_std: normalize per-patch std
        use_log: compare in log-spectrum space
        mid_high_emphasis: weight mid/high bins more heavily
        eps: numerical stability

    Returns:
        scalar loss
    """
    # Convert to luminance
    y_trans = rgb_to_luminance(translated_rgb)   # [B,H,W]
    y_real = rgb_to_luminance(real_rgb)          # [B,H,W]

    # Sample patches
    trans_patches = sample_random_patches(
        y_trans, patch_size=patch_size, patches_per_image=patches_per_image
    )  # [Bp,N,N]

    real_patches = sample_random_patches(
        y_real, patch_size=patch_size, patches_per_image=patches_per_image
    )  # [Bp,N,N]

    # Compute radial spectra
    trans_radial, r = radial_power_spectrum_batch(
        trans_patches,
        remove_mean=remove_mean,
        normalize_std=normalize_std,
        window="hann",
        eps=eps,
    )
    real_radial, _ = radial_power_spectrum_batch(
        real_patches,
        remove_mean=remove_mean,
        normalize_std=normalize_std,
        window="hann",
        eps=eps,
    )

    # Average across sampled patches in the batch
    trans_mean = trans_radial.mean(dim=0)   # [R]
    real_mean = real_radial.mean(dim=0)     # [R]

    # Optionally compare in log space
    if use_log:
        trans_mean = torch.log(trans_mean + eps)
        real_mean = torch.log(real_mean + eps)

    diff = torch.abs(trans_mean - real_mean)   # [R]

    # Optional weighting: de-emphasize DC, emphasize mid/high bins
    if mid_high_emphasis:
        # Build simple smooth weights from radius
        r = r.float()
        r_norm = r / max(r.max(), torch.tensor(1.0, device=r.device))
        weights = 0.25 + 0.75 * (r_norm ** 0.75)

        # Zero or strongly downweight DC
        weights[0] = 0.0
        diff = diff * weights
        loss = diff.sum() / (weights.sum() + eps)
    else:
        diff = diff[1:]  # drop DC bin
        loss = diff.mean()

    return loss