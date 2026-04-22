from pathlib import Path

import numpy as np
from PIL import Image


def parse_prefixed_float(name: str, prefix: str) -> float:
    if not name.startswith(prefix):
        raise ValueError(f"Expected '{name}' to start with '{prefix}'.")
    return float(name[len(prefix) :])


def parse_step_index(step_dir: Path) -> int:
    return int(step_dir.name.removeprefix("step-"))


def sorted_step_dirs(figs_dir: Path) -> list[Path]:
    step_dirs = [path for path in figs_dir.iterdir() if path.is_dir() and path.name.startswith("step-")]
    return sorted(step_dirs, key=parse_step_index)


def sorted_strength_dirs(step_dir: Path) -> list[Path]:
    strength_dirs = [path for path in step_dir.iterdir() if path.is_dir() and path.name.startswith("strength-")]
    return sorted(strength_dirs, key=lambda path: parse_prefixed_float(path.name, "strength-"))


def sorted_cfg_dirs(strength_dir: Path) -> list[Path]:
    cfg_dirs = [path for path in strength_dir.iterdir() if path.is_dir() and path.name.startswith("cfg-")]
    return sorted(cfg_dirs, key=lambda path: parse_prefixed_float(path.name, "cfg-"))


def load_image(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


def save_image(image: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_path)


def compose_vertical_pair(top_image: np.ndarray, bottom_image: np.ndarray) -> np.ndarray:
    if top_image.shape != bottom_image.shape:
        raise ValueError(
            "Source and translated images must have the same shape for pixel-perfect composition: "
            f"{top_image.shape} vs {bottom_image.shape}"
        )

    height, width, channels = top_image.shape
    canvas = np.zeros((height * 2, width, channels), dtype=np.uint8)
    canvas[:height] = top_image
    canvas[height:] = bottom_image
    return canvas


def save_cfg_pair_image(
    source_image: np.ndarray,
    translated_path: Path,
    output_name: str,
):
    translated_image = load_image(translated_path)
    pair_image = compose_vertical_pair(source_image, translated_image)
    save_image(pair_image, translated_path.parent / output_name)


def process_step_dir(
    step_dir: Path,
    output_name: str,
):
    source_path = step_dir / "x_src.png"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source image: {source_path}")

    source_image = load_image(source_path)
    saved_count = 0
    missing_paths: list[str] = []

    for strength_dir in sorted_strength_dirs(step_dir):
        for cfg_dir in sorted_cfg_dirs(strength_dir):
            translated_path = cfg_dir / "ema.png"
            if not translated_path.exists():
                missing_paths.append(str(translated_path))
                continue

            save_cfg_pair_image(
                source_image=source_image,
                translated_path=translated_path,
                output_name=output_name,
            )
            saved_count += 1

    if missing_paths:
        print(f"{step_dir.name}: saved {saved_count} pair images with {len(missing_paths)} missing ema.png files.")
        for missing_path in missing_paths:
            print(f"  missing: {missing_path}")
    else:
        print(f"{step_dir.name}: saved {saved_count} pair images")


def main():
    # Training run directory that contains the `training/figs/step-*` folders.
    run_dir = Path("/home/logan/projects/pixel-cyclenet/your-run-dir")
    # Output filename written into each `cfg-*` directory.
    output_name = "source_and_ema.png"

    figs_dir = run_dir / "training" / "figs"
    if not figs_dir.exists():
        raise FileNotFoundError(f"Could not find figs directory: {figs_dir}")

    step_dirs = sorted_step_dirs(figs_dir)
    if not step_dirs:
        raise ValueError(f"No step directories found under {figs_dir}")

    for step_dir in step_dirs:
        process_step_dir(
            step_dir=step_dir,
            output_name=output_name,
        )


if __name__ == "__main__":
    main()
