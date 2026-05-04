import csv
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


def parse_prefixed_float(name: str, prefix: str) -> float:
    if not name.startswith(prefix):
        raise ValueError(f"Expected '{name}' to start with '{prefix}'.")
    return float(name[len(prefix) :])


def parse_step_index(name: str) -> int:
    if not name.startswith("step-"):
        raise ValueError(f"Expected '{name}' to start with 'step-'.")
    return int(name.removeprefix("step-"))


def iter_candidate_dirs(step_dir: str | Path) -> list[tuple[int, float, float, Path]]:
    root = Path(step_dir)
    if not root.exists():
        raise FileNotFoundError(f"step_dir does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"step_dir must be a directory, got: {root}")
    if not root.name.startswith("step-"):
        raise ValueError(f"Expected step_dir name like 'step-*', got '{root.name}'")

    candidates: list[tuple[int, float, float, Path]] = []
    step = parse_step_index(root.name)

    strength_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("strength-")],
        key=lambda path: parse_prefixed_float(path.name, "strength-"),
    )
    for strength_dir in strength_dirs:
        noise_strength = parse_prefixed_float(strength_dir.name, "strength-")
        cfg_dirs = sorted(
            [path for path in strength_dir.iterdir() if path.is_dir() and path.name.startswith("cfg-")],
            key=lambda path: parse_prefixed_float(path.name, "cfg-"),
        )
        for cfg_dir in cfg_dirs:
            cfg_weight = parse_prefixed_float(cfg_dir.name, "cfg-")
            candidates.append((step, noise_strength, cfg_weight, cfg_dir))

    if not candidates:
        raise ValueError(
            f"No strength/cfg directories were found under {root}. "
            "Expected a layout like step-*/strength-*/cfg-*."
        )

    return candidates


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


def write_rows_to_csv(rows: list[dict[str, object]], csv_out_path: Path) -> None:
    if not rows:
        raise ValueError("No FID stats were computed.")

    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
        )
        writer.writeheader()
        writer.writerows(rows)


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
    

def fid_sweep(reference_dir: Path | str, cyclenet_sim_dir: Path | str, steps: list[int]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid_computer = FIDComputer(device)

    for step in steps:

        step_dir = Path(cyclenet_sim_dir) / f"step-{step}"

        # CSV path where the aggregated per-setting FID stats for this step will be saved.
        sweep_csv_out_path = step_dir / "fid_stats.csv"

        if not reference_dir or not step_dir:
            raise ValueError(
                "Set reference_dir and step_dir in main() for `translated_step_vs_reference` mode."
            )

        summary_rows: list[dict[str, object]] = []
        for step, noise_strength, cfg_weight, translated_dir in iter_candidate_dirs(step_dir):
            fid = fid_computer.compute(reference_dir, translated_dir)

            print(
                f"step-{step} / strength-{noise_strength:.1f} / cfg-{cfg_weight:.1f}".center(50, "=")
            )
            print(f"[ FID ]: {fid:.6f}")

            summary_rows.append(
                {
                    "reference_dir": str(reference_dir),
                    "step": step,
                    "noise_strength": noise_strength,
                    "cfg_weight": cfg_weight,
                    "translated_dir": str(translated_dir),
                    "fid": fid,
                }
            )

        write_rows_to_csv(summary_rows, sweep_csv_out_path)
        print(f"\nSaved FID stats CSV to {sweep_csv_out_path}")


def fid_direct_pair():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid_computer = FIDComputer(device)

    reference_dir = "/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/projection/oem_proj"
    # First dataset in a direct baseline comparison. For example, simulated images.
    direct_fake_dir = "/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/projection/sim_proj"
    # CSV path for the one-off direct baseline result, for example `sim vs real`.
    direct_csv_out_path = Path("/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/projection/fid_baselines/sim_vs_real_fid.csv")

    if not reference_dir or not direct_fake_dir:
        raise ValueError(
            "Set reference_dir and direct_fake_dir in main() for `direct_pair` mode."
        )

    fid = fid_computer.compute(reference_dir, direct_fake_dir)
    print(f"[ FID ] {Path(direct_fake_dir).name} vs {Path(reference_dir).name}: {fid:.6f}")

    rows = [
        {
            "reference_dir": str(reference_dir),
            "fake_dir": str(direct_fake_dir),
            "fid": fid,
        }
    ]
    write_rows_to_csv(rows, direct_csv_out_path)
    print(f"\nSaved direct-pair FID CSV to {direct_csv_out_path}")
    return
    