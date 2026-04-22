import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cyclenet.eval.plotting.set_style import apply_style

apply_style()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_images(root: Path, file_exts: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in file_exts
    )


def match_key(root: Path, path: Path, match_mode: str) -> str:
    rel_path = path.resolve().relative_to(root.resolve())

    if match_mode == "relative_stem":
        return rel_path.with_suffix("").as_posix()
    if match_mode == "filename_stem":
        return path.stem

    raise ValueError(f"Unsupported match_mode: {match_mode}")


def build_image_map(root: Path, file_exts: set[str], match_mode: str) -> dict[str, Path]:
    image_map: dict[str, Path] = {}
    for path in collect_images(root, file_exts):
        key = match_key(root, path, match_mode)
        if key in image_map:
            raise ValueError(
                f"Duplicate match key '{key}' under {root}. "
                f"Choose a different match_mode or clean up duplicates."
            )
        image_map[key] = path
    return image_map


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def select_pair_keys(common_keys: list[str], num_pairs: int, seed: int) -> list[str]:
    if not common_keys:
        raise ValueError("No matching image pairs found.")

    if num_pairs <= 0:
        raise ValueError("num_pairs must be greater than 0.")

    rng = random.Random(seed)
    n_select = min(num_pairs, len(common_keys))
    return sorted(rng.sample(common_keys, k=n_select))


def plot_pairs(
    sim_map: dict[str, Path],
    translated_map: dict[str, Path],
    pair_keys: list[str],
    out_path: Path,
    figure_title: str | None,
    dpi: int,
):
    n_rows = len(pair_keys)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(10.0, max(2.8 * n_rows, 3.5)),
        dpi=dpi,
        squeeze=False,
        constrained_layout=True,
    )

    if figure_title:
        fig.suptitle(figure_title)

    axes[0, 0].set_title("Sim")
    axes[0, 1].set_title("Translated")

    for row, key in enumerate(pair_keys):
        sim_img = load_rgb(sim_map[key])
        translated_img = load_rgb(translated_map[key])

        axes[row, 0].imshow(sim_img, interpolation="nearest")
        axes[row, 1].imshow(translated_img, interpolation="nearest")

        for col in range(2):
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(key, rotation=0, ha="right", va="center", labelpad=48)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    # Root directory containing the source sim images.
    sim_dir = Path("/path/to/sim")
    # Root directory containing translated outputs to compare against sim.
    translated_dir = Path("/path/to/translated")
    # Output figure path for the sampled side-by-side plot.
    out_path = Path("/tmp/random_translation_pairs.png")
    # Number of random sim/translated pairs to show.
    num_pairs = 8
    # Random seed for reproducible pair sampling.
    seed = 42
    # Matching strategy: "relative_stem" for mirrored directory trees, "filename_stem" for flattened outputs.
    match_mode = "relative_stem"
    # Optional title for the saved figure.
    figure_title = "Random Sim vs Translated Pairs"
    # Save DPI for the output figure.
    dpi = 200

    if not sim_dir.exists():
        raise FileNotFoundError(f"sim_dir does not exist: {sim_dir}")
    if not translated_dir.exists():
        raise FileNotFoundError(f"translated_dir does not exist: {translated_dir}")

    sim_map = build_image_map(sim_dir, IMAGE_EXTS, match_mode)
    translated_map = build_image_map(translated_dir, IMAGE_EXTS, match_mode)

    common_keys = sorted(set(sim_map) & set(translated_map))
    pair_keys = select_pair_keys(common_keys, num_pairs, seed)

    plot_pairs(
        sim_map=sim_map,
        translated_map=translated_map,
        pair_keys=pair_keys,
        out_path=out_path,
        figure_title=figure_title,
        dpi=dpi,
    )

    print(f"Matched {len(common_keys)} pairs between {sim_dir} and {translated_dir}.")
    print(f"Saved {len(pair_keys)} random pairs to {out_path}")


if __name__ == "__main__":
    main()
