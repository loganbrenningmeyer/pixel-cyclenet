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
    return FIDComputer().compute(real_dir, fake_dir)


class FIDComputer:
    def __init__(self, device: str | torch.device | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fid = self._build_metric().to(self.device)

    def _build_metric(self) -> FrechetInceptionDistance:
        """
        TorchMetrics syncs metric states across an initialized process group by
        default during compute(). The translate sweep computes FID on rank 0
        only, so distributed sync must stay disabled to avoid DDP hangs.
        """
        try:
            return FrechetInceptionDistance(
                feature=2048,
                normalize=True,
                sync_on_compute=False,
            )
        except TypeError:
            metric = FrechetInceptionDistance(feature=2048, normalize=True)
            if hasattr(metric, "sync_on_compute"):
                metric.sync_on_compute = False
            return metric

    def compute(self, real_dir: str | Path, fake_dir: str | Path) -> float:
        """
        Computes FID while reusing the same Inception model across calls.
        """
        self.fid.reset()

        with torch.no_grad():
            # -- Real
            for x in iter_folder(real_dir):
                self.fid.update(x.unsqueeze(0).to(self.device), real=True)
            # -- Translated sim
            for x in iter_folder(fake_dir):
                self.fid.update(x.unsqueeze(0).to(self.device), real=False)

        return float(self.fid.compute())


def main():
    real_dir = ""
    fake_dir = ""

    fid_score = compute_fid(real_dir, fake_dir)

    print("FID:", fid_score)


if __name__ == "__main__":
    main()
