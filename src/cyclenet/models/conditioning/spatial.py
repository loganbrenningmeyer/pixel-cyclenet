import torch


def build_condition_input(
    img: torch.Tensor | None,
    seg: torch.Tensor | None,
    cond_mode: str,
) -> torch.Tensor | None:
    """

    """
    if cond_mode == "rgb":
        if img is None:
            raise ValueError("cond_mode='rgb' requires img")
        img_norm = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
        return img_norm

    if cond_mode == "seg":
        if seg is None:
            raise ValueError("cond_mode='seg' requires seg")
        return seg.float()

    if cond_mode == "rgb_seg":
        if img is None or seg is None:
            raise ValueError("cond_mode='rgb_seg' requires both img and seg")
        img_norm = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
        return torch.cat([img_norm, seg.float()], dim=1)

    raise ValueError(f"Unknown cond_mode: {cond_mode}")


def build_seg_modulation_input(
    seg: torch.Tensor | None,
    use_spade: bool,
) -> torch.Tensor | None:
    """
    
    """
    # -------------------------
    # SPADE segmentation mask modulation
    # -------------------------
    if use_spade:
        if seg is None:
            raise ValueError("use_space=True requires seg")
        return seg.float()
    # -------------------------
    # No segmentation mask modulation
    # -------------------------
    else:
        return None


def control_in_channels(cond_mode: str, num_seg_classes: int) -> int:
    """
    
    """
    if cond_mode == "rgb":
        return 3
    elif cond_mode == "seg": 
        return num_seg_classes
    elif cond_mode == "rgb_seg":
        return 3 + num_seg_classes
    else:
        raise ValueError(f"Unknown cond_mode: {cond_mode}")