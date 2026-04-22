import csv
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".png", ".tif", ".tiff"}


def sample_images(root_dir: str, num_samples: int) -> list[Path]:
    paths_by_name: dict[str, list[Path]] = {}
    for path in Path(root_dir).rglob("*"):
        if path.parent.name != "opt" or path.suffix.lower() not in IMAGE_EXTS:
            continue

        paths_by_name.setdefault(path.name, []).append(path)

    unique_filenames = sorted(paths_by_name)
    if num_samples > len(unique_filenames):
        raise ValueError(
            f"Requested {num_samples} samples, but only found {len(unique_filenames)} "
            f"unique filenames under {root_dir}."
        )

    selected_filenames = random.sample(unique_filenames, k=num_samples)
    sample_paths = [
        random.choice(paths_by_name[filename])
        for filename in selected_filenames
    ]

    return sample_paths


def resolve_mask_path(img_path: Path, label_parent_dir: str = "gt_ss_mask") -> Path:
    if img_path.parent.name != "opt":
        raise ValueError(f"Expected RGB image under 'opt', got {img_path}")

    mask_dir = img_path.parent.parent / label_parent_dir
    mask_path = mask_dir / img_path.name
    if mask_path.exists():
        return mask_path

    matches = [
        path
        for path in sorted(mask_dir.glob(f"{img_path.stem}.*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not matches:
        raise FileNotFoundError(f"Missing mask for {img_path}: expected {mask_path}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple masks found for {img_path}: {matches}")
    return matches[0]


def main():
    seed = 42
    random.seed(seed)

    n_samples = 2000

    data_dir = "/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/sim_subset/sim_test"
    img_out_dir = Path("/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/sim_subset/sim_proj")
    label_out_dir = Path("/cgi/data/nvesd/workspaces/logan/data/remote_sensing/tiled/sim_subset/sim_labels")

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    img_paths = sample_images(data_dir, n_samples)
    manifest_path = img_out_dir / "subset_manifest.csv"
    manifest_rows: list[tuple[str, str, str, str]] = []

    for img_path in tqdm(img_paths, desc="Copying images"):
        mask_path = resolve_mask_path(img_path)
        filename = img_path.name
        img_dst_path = img_out_dir / filename
        label_dst_path = label_out_dir / filename

        shutil.copy(str(img_path), str(img_dst_path))
        shutil.copy(str(mask_path), str(label_dst_path))
        manifest_rows.append(
            (str(img_path), str(img_dst_path), str(mask_path), str(label_dst_path))
        )

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_path", "out_path", "mask_source_path", "mask_out_path"])
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
