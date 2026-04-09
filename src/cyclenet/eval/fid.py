from pathlib import Path
from PIL import Image
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as T


# -------------------------
# InceptionV3-compatible transform
# -------------------------
transform = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
])


def iter_folder(folder: str | Path, exts: set[str] = {".png", ".jpg", ".tif", ".tiff"}):
    folder = Path(folder)

    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in exts:
            with Image.open(path) as img:
                yield transform(img.convert("RGB"))


def compute_fid(real_dir: str | Path, fake_dir: str | Path) -> float:
    """
    Given paths to real / fake directories of images, computes and returns FID score 
    -- ( Recommended ): 50k images per domain
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    with torch.no_grad():
        # -- Real
        for x in iter_folder(real_dir):
            fid.update(x.unsqueeze(0).to(device), real=True)
        # -- Translated sim
        for x in iter_folder(fake_dir):
            fid.update(x.unsqueeze(0).to(device), real=False)

    return float(fid.compute())


def main():
    real_dir = ""
    fake_dir = ""

    fid_score = compute_fid(real_dir, fake_dir)

    print("FID:", fid_score)


if __name__ == "__main__":
    main()