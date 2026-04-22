import csv
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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


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


def collect_images_by_name(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    images_by_name: dict[str, Path] = {}
    duplicate_names: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        name = path.name
        if name in images_by_name:
            duplicate_names.append(name)
            continue
        images_by_name[name] = path

    if duplicate_names:
        dupes = ", ".join(sorted(set(duplicate_names))[:10])
        raise ValueError(
            f"Found duplicate filenames under {root}. "
            f"Pairing by filename would be ambiguous. Examples: {dupes}"
        )

    if not images_by_name:
        raise ValueError(f"No images found under {root}")

    return images_by_name


def pair_images_by_name(
    sim_dir: str | Path,
    translated_dir: str | Path,
) -> list[tuple[Path, Path]]:
    sim_images = collect_images_by_name(sim_dir)
    translated_images = collect_images_by_name(translated_dir)

    shared_names = sorted(set(sim_images) & set(translated_images))
    if not shared_names:
        raise ValueError(
            "No shared filenames were found between "
            f"{Path(sim_dir)} and {Path(translated_dir)}"
        )

    return [(sim_images[name], translated_images[name]) for name in shared_names]


def compute_average_lpips(
    sim_dir: str | Path,
    translated_dir: str | Path,
    batch_size: int = 32,
) -> tuple[float, np.ndarray, list[tuple[Path, Path]]]:
    pairs = pair_images_by_name(sim_dir, translated_dir)
    values_all: list[np.ndarray] = []

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        sim_batch = torch.stack([load_img(sim_path) for sim_path, _ in batch_pairs], dim=0)
        translated_batch = torch.stack([load_img(translated_path) for _, translated_path in batch_pairs], dim=0)
        values_all.append(lpips_batch(sim_batch, translated_batch))

    values = np.concatenate(values_all, axis=0)
    return float(values.mean()), values, pairs


def main() -> None:
    # Number of image pairs to score together on each forward pass.
    batch_size = 32

    # Directory containing the source sim images to compare.
    sim_dir = "/develop/data/remote_sensing/tiled/projection/sim_proj"

    # Single `step-{step}` directory whose `strength-{strength}/cfg-{cfg}` subdirectories
    # contain translated outputs to compare against the source sim images.
    # Example:
    # `/.../all_real_ft_invar/step-30000`
    # `/.../oem_only/ema/step-2500`
    step_dir = Path("/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/projection/cyclenet_sim_proj/oem_only/ema/step-15000")

    # CSV path where the aggregated per-setting LPIPS stats for this step will be saved.
    csv_out_path = step_dir / "lpips_stats.csv"
    summary_rows: list[dict[str, object]] = []

    if not sim_dir or not step_dir:
        raise ValueError("Set sim_dir and step_dir in main() before running this script.")

    for step, noise_strength, cfg_weight, translated_dir in iter_candidate_dirs(step_dir):
        average_lpips, values, pairs = compute_average_lpips(
            sim_dir=sim_dir,
            translated_dir=translated_dir,
            batch_size=batch_size,
        )

        print(
            f"step-{step} / strength-{noise_strength:.1f} / cfg-{cfg_weight:.1f}".center(50, '=')
        )
        print(f"paired_images: {len(pairs)}")
        print(f"[ Average LPIPS ]: {average_lpips:.6f}")
        print(f"[ LPIPS STD     ]: {float(values.std()):.6f}")

        summary_rows.append(
            {
                "step": step,
                "noise_strength": noise_strength,
                "cfg_weight": cfg_weight,
                "translated_dir": str(translated_dir),
                "paired_images": len(pairs),
                "lpips_mean": average_lpips,
                "lpips_std": float(values.std()),
                "lpips_min": float(values.min()),
                "lpips_max": float(values.max()),
            }
        )

    if not summary_rows:
        raise ValueError("No LPIPS stats were computed.")

    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "noise_strength",
                "cfg_weight",
                "translated_dir",
                "paired_images",
                "lpips_mean",
                "lpips_std",
                "lpips_min",
                "lpips_max",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved LPIPS stats CSV to {csv_out_path}")


if __name__ == "__main__":
    main()
