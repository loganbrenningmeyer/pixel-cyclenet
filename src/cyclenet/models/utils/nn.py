import torch
import torch.nn as nn


class ContextIdentity(nn.Module):
    """
    Acts as nn.Identity() when passed both x and ctx as input.
    """
    def forward(self, x, ctx=None):
        return x


def zero_module(module: nn.Module):
    """
    Zero out the parameters of a torch Module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def zero_skip_list(skips: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Converts the list of skips to zero tensors 
    """
    return [torch.zeros_like(skip) for skip in skips]


def unwrap(m: nn.Module):
    """
    Unwraps DDP module if it is wrapped
    """
    return m.module if hasattr(m, "module") else m