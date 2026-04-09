from pathlib import Path
import lpips
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = lpips.LPIPS(net="alex").to(device)
loss_fn.eval()

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5,)*3, (0.5,)*3),     # [-1, 1]
])


def load_img(path: str | Path) -> torch.Tensor:
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


def lpips_pair(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Computes LPIPS loss for single image tensor pair
    """
    x = img1.unsqueeze(0).to(device)    # (1,3,H,W)
    y = img2.unsqueeze(0).to(device)    # (1,3,H,W)
    with torch.no_grad():
        d = loss_fn(x, y)
    return float(d.item())


def lpips_batch(b1: torch.Tensor, b2: torch.Tensor) -> np.ndarray:
    """
    Computes LPIPS loss for each image pair in a batch of tensors (B,3,H,W)
    """
    with torch.no_grad():
        d: torch.Tensor = loss_fn(b1.to(device), b2.to(device))
    return d.view(-1).cpu().numpy()     # (B,)