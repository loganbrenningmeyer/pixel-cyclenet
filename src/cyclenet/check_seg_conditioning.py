import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cyclenet.data.dataset import (
    CycleDomainSegDataset,
    SourceSegDataset,
    TranslateSegDataset,
    load_label_mask,
)
from cyclenet.data.transforms import load_cyclenet_transforms
from cyclenet.diffusion.losses import build_seg_condition


DEFAULT_PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 50],
    ],
    dtype=np.uint8,
)


def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)


def require_num_seg_classes(config: DictConfig) -> int:
    num_classes = config.model.get("num_seg_classes", None)
    if num_classes is None:
        num_classes = 8
        print(
            "[warning] Missing `model.num_seg_classes` in config. "
            "Defaulting to 8 for checker consistency with dataset.py."
        )
    return int(num_classes)


def denorm_image(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) / 2.0).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def seg_to_color(seg: torch.Tensor, palette: np.ndarray = DEFAULT_PALETTE) -> np.ndarray:
    seg_cpu = seg.detach().cpu()
    seg_idx = seg_cpu.argmax(dim=0).numpy().astype(np.int64)
    valid = (seg_cpu.sum(dim=0) > 0).numpy()

    if valid.any() and palette.shape[0] < int(seg_idx[valid].max()) + 1:
        raise ValueError(
            f"Palette has {palette.shape[0]} colors, but found class id {int(seg_idx[valid].max())}."
        )

    color = np.zeros((seg_idx.shape[0], seg_idx.shape[1], 3), dtype=np.uint8)
    color[valid] = palette[seg_idx[valid]]
    return color


def overlay_mask(rgb: np.ndarray, seg_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = (1.0 - alpha) * rgb.astype(np.float32) + alpha * seg_rgb.astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def build_label_path_from_rgb(
    rgb_path: Path,
    rgb_parent_dir: str,
    label_parent_dir: str,
) -> Path:
    if rgb_path.parent.name != rgb_parent_dir:
        raise ValueError(
            f"Expected RGB file under parent dir `{rgb_parent_dir}`, got: {rgb_path}"
        )
    return rgb_path.parent.parent / label_parent_dir / rgb_path.name


def collect_rgb_paths(
    data_dir: str,
    rgb_parent_dir: str,
    file_exts: set[str],
) -> list[Path]:
    paths = []
    for path in sorted(Path(data_dir).rglob("*")):
        if path.suffix.lower() not in file_exts:
            continue
        if path.parent.name != rgb_parent_dir:
            continue
        paths.append(path)
    return paths


def inspect_raw_masks(
    rgb_paths: list[Path],
    rgb_parent_dir: str,
    label_parent_dir: str,
    domain_name: str,
    max_inspect: int = 16,
) -> dict[str, object]:
    unique_vals = set()
    missing = []

    for rgb_path in rgb_paths[:max_inspect]:
        label_path = build_label_path_from_rgb(rgb_path, rgb_parent_dir, label_parent_dir)
        if not label_path.exists():
            missing.append((rgb_path, label_path))
            continue

        mask = load_label_mask(label_path)
        vals = np.unique(mask)
        unique_vals.update(int(v) for v in vals.tolist())

    print(f"[{domain_name}] raw RGB files found: {len(rgb_paths)}")
    print(f"[{domain_name}] raw label unique values over first {min(len(rgb_paths), max_inspect)} samples: {sorted(unique_vals)}")

    if missing:
        print(f"[{domain_name}] missing labels: {len(missing)}")
        first_rgb, first_lbl = missing[0]
        print(f"[{domain_name}] first missing label example:")
        print(f"  rgb   : {first_rgb}")
        print(f"  label : {first_lbl}")

    return {
        "unique_vals": unique_vals,
        "missing": missing,
    }


def validate_sample(
    img: torch.Tensor,
    seg: torch.Tensor,
    cond: torch.Tensor,
    num_classes: int,
    domain_name: str,
    sample_idx: int,
) -> None:
    assert img.ndim == 3, f"{domain_name}[{sample_idx}] image should be 3D CHW, got {tuple(img.shape)}"
    assert seg.ndim == 3, f"{domain_name}[{sample_idx}] seg should be 3D CHW, got {tuple(seg.shape)}"
    assert cond.ndim == 3, f"{domain_name}[{sample_idx}] cond should be 3D CHW, got {tuple(cond.shape)}"

    assert img.shape[0] == 3, f"{domain_name}[{sample_idx}] expected RGB channels=3, got {img.shape[0]}"
    assert seg.shape[0] == num_classes, (
        f"{domain_name}[{sample_idx}] expected seg channels={num_classes}, got {seg.shape[0]}"
    )
    assert cond.shape[0] == 3 + num_classes, (
        f"{domain_name}[{sample_idx}] expected cond channels={3 + num_classes}, got {cond.shape[0]}"
    )

    assert img.dtype == torch.float32, f"{domain_name}[{sample_idx}] image dtype should be float32, got {img.dtype}"
    assert seg.dtype == torch.float32, f"{domain_name}[{sample_idx}] seg dtype should be float32, got {seg.dtype}"
    assert cond.dtype == torch.float32, f"{domain_name}[{sample_idx}] cond dtype should be float32, got {cond.dtype}"

    img_min = float(img.min())
    img_max = float(img.max())
    seg_min = float(seg.min())
    seg_max = float(seg.max())
    cond_rgb_min = float(cond[:3].min())
    cond_rgb_max = float(cond[:3].max())
    cond_seg_min = float(cond[3:].min())
    cond_seg_max = float(cond[3:].max())

    assert img_min >= -1.001 and img_max <= 1.001, (
        f"{domain_name}[{sample_idx}] image range should be ~[-1,1], got [{img_min:.4f}, {img_max:.4f}]"
    )
    assert cond_rgb_min >= -1e-6 and cond_rgb_max <= 1.001, (
        f"{domain_name}[{sample_idx}] condition RGB range should be ~[0,1], got [{cond_rgb_min:.4f}, {cond_rgb_max:.4f}]"
    )
    assert cond_seg_min >= -1e-6 and cond_seg_max <= 1.001, (
        f"{domain_name}[{sample_idx}] condition seg range should be ~[0,1], got [{cond_seg_min:.4f}, {cond_seg_max:.4f}]"
    )

    per_pixel_sum = seg.sum(dim=0)
    valid_pixel = per_pixel_sum > 0

    max_valid_dev = 0.0
    if valid_pixel.any():
        max_valid_dev = float((per_pixel_sum[valid_pixel] - 1.0).abs().max())

    invalid_pixel = ~valid_pixel
    max_invalid_abs = 0.0
    if invalid_pixel.any():
        max_invalid_abs = float(per_pixel_sum[invalid_pixel].abs().max())

    unique_seg_values = torch.unique(seg.cpu())
    unique_seg_values_list = [float(v) for v in unique_seg_values.tolist()]

    if max_valid_dev > 1e-4:
        raise AssertionError(
            f"{domain_name}[{sample_idx}] valid segmentation pixels are not one-hot after transform; "
            f"max per-pixel channel-sum deviation is {max_valid_dev:.6f}"
        )

    if max_invalid_abs > 1e-4:
        raise AssertionError(
            f"{domain_name}[{sample_idx}] ignored pixels are expected to be all-zero across channels; "
            f"max ignored-pixel channel sum is {max_invalid_abs:.6f}"
        )

    invalid_seg_values = [v for v in unique_seg_values_list if v not in (0.0, 1.0)]
    if invalid_seg_values:
        raise AssertionError(
            f"{domain_name}[{sample_idx}] segmentation contains non-binary values: {invalid_seg_values[:10]}"
        )

    class_map = seg.argmax(dim=0).cpu()
    classes_present = torch.unique(class_map[valid_pixel.cpu()]).tolist() if valid_pixel.any() else []
    ignored_fraction = float(invalid_pixel.float().mean())

    print(f"[{domain_name}] sample {sample_idx}")
    print(f"  image shape/dtype : {tuple(img.shape)} / {img.dtype}")
    print(f"  seg shape/dtype   : {tuple(seg.shape)} / {seg.dtype}")
    print(f"  cond shape/dtype  : {tuple(cond.shape)} / {cond.dtype}")
    print(f"  image range       : [{img_min:.4f}, {img_max:.4f}]")
    print(f"  cond rgb range    : [{cond_rgb_min:.4f}, {cond_rgb_max:.4f}]")
    print(f"  cond seg range    : [{cond_seg_min:.4f}, {cond_seg_max:.4f}]")
    print(f"  classes present   : {classes_present}")
    print(f"  ignored fraction  : {ignored_fraction:.4f}")


def save_debug_visual(
    out_dir: Path,
    domain_name: str,
    sample_idx: int,
    img: torch.Tensor,
    seg: torch.Tensor,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = denorm_image(img)
    seg_rgb = seg_to_color(seg)
    overlay = overlay_mask(rgb, seg_rgb)
    panel = np.concatenate([rgb, seg_rgb, overlay], axis=1)

    out_path = out_dir / f"{domain_name.lower()}_{sample_idx:02d}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def inspect_dataset(
    dataset,
    domain_name: str,
    num_classes: int,
    num_samples: int,
    save_debug_dir: Path | None,
) -> None:
    print(f"\nInspecting dataset: {domain_name}")
    print(f"  num samples: {len(dataset)}")

    inspect_n = min(num_samples, len(dataset))
    for i in range(inspect_n):
        img, seg, src_idx, tgt_idx = dataset[i]
        cond = build_seg_condition(img.unsqueeze(0), seg.unsqueeze(0)).squeeze(0)

        validate_sample(img, seg, cond, num_classes, domain_name, i)

        print(f"  src_idx/tgt_idx   : {int(src_idx)} -> {int(tgt_idx)}")

        if save_debug_dir is not None:
            save_debug_visual(save_debug_dir, domain_name, i, img, seg)


def inspect_source_dataset(dataset, num_classes: int, num_samples: int, save_debug_dir: Path | None) -> None:
    print("\nInspecting source sampling dataset")
    print(f"  num samples: {len(dataset)}")

    inspect_n = min(num_samples, len(dataset))
    for i in range(inspect_n):
        img, seg = dataset[i]
        cond = build_seg_condition(img.unsqueeze(0), seg.unsqueeze(0)).squeeze(0)
        validate_sample(img, seg, cond, num_classes, "source", i)

        if save_debug_dir is not None:
            save_debug_visual(save_debug_dir, "source", i, img, seg)


def inspect_translate_dataset(dataset, num_classes: int, num_samples: int, save_debug_dir: Path | None) -> None:
    print("\nInspecting translation dataset")
    print(f"  num samples: {len(dataset)}")

    inspect_n = min(num_samples, len(dataset))
    for i in range(inspect_n):
        img, seg, filepath = dataset[i]
        cond = build_seg_condition(img.unsqueeze(0), seg.unsqueeze(0)).squeeze(0)
        validate_sample(img, seg, cond, num_classes, "translate", i)
        print(f"  filepath          : {filepath}")

        if save_debug_dir is not None:
            save_debug_visual(save_debug_dir, "translate", i, img, seg)


def inspect_batch(dataset, batch_size: int, num_classes: int, name: str) -> None:
    print(f"\nInspecting DataLoader batch: {name}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    batch = next(iter(loader))

    if len(batch) == 4:
        imgs, segs, src_idx, tgt_idx = batch
        cond = build_seg_condition(imgs, segs)
        print(f"  imgs shape/dtype  : {tuple(imgs.shape)} / {imgs.dtype}")
        print(f"  segs shape/dtype  : {tuple(segs.shape)} / {segs.dtype}")
        print(f"  cond shape/dtype  : {tuple(cond.shape)} / {cond.dtype}")
        print(f"  src_idx shape     : {tuple(src_idx.shape)}")
        print(f"  tgt_idx shape     : {tuple(tgt_idx.shape)}")

        assert imgs.shape[1] == 3
        assert segs.shape[1] == num_classes
        assert cond.shape[1] == 3 + num_classes

    elif len(batch) == 2:
        imgs, segs = batch
        cond = build_seg_condition(imgs, segs)
        print(f"  imgs shape/dtype  : {tuple(imgs.shape)} / {imgs.dtype}")
        print(f"  segs shape/dtype  : {tuple(segs.shape)} / {segs.dtype}")
        print(f"  cond shape/dtype  : {tuple(cond.shape)} / {cond.dtype}")

        assert imgs.shape[1] == 3
        assert segs.shape[1] == num_classes
        assert cond.shape[1] == 3 + num_classes

    elif len(batch) == 3:
        imgs, segs, filepaths = batch
        cond = build_seg_condition(imgs, segs)
        print(f"  imgs shape/dtype  : {tuple(imgs.shape)} / {imgs.dtype}")
        print(f"  segs shape/dtype  : {tuple(segs.shape)} / {segs.dtype}")
        print(f"  cond shape/dtype  : {tuple(cond.shape)} / {cond.dtype}")
        print(f"  num filepaths     : {len(filepaths)}")

        assert imgs.shape[1] == 3
        assert segs.shape[1] == num_classes
        assert cond.shape[1] == 3 + num_classes

    else:
        raise ValueError(f"Unexpected batch structure with length {len(batch)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-debug-dir", type=str, default=None)
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip translation-dataset inspection.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    num_classes = require_num_seg_classes(config)

    save_debug_dir = Path(args.save_debug_dir) if args.save_debug_dir is not None else None

    file_exts = {".jpg", ".png", ".tif", ".tiff"}
    rgb_parent_dir = str(config.data.rgb_parent_dir)
    label_parent_dir = str(config.data.label_parent_dir)

    print("Checking raw masks before dataset construction")
    src_rgb_paths = collect_rgb_paths(config.data.src_dir, rgb_parent_dir, file_exts)
    tgt_rgb_paths = collect_rgb_paths(config.data.tgt_dir, rgb_parent_dir, file_exts)

    src_stats = inspect_raw_masks(
        src_rgb_paths,
        rgb_parent_dir=rgb_parent_dir,
        label_parent_dir=label_parent_dir,
        domain_name="sim",
    )
    tgt_stats = inspect_raw_masks(
        tgt_rgb_paths,
        rgb_parent_dir=rgb_parent_dir,
        label_parent_dir=label_parent_dir,
        domain_name="real",
    )

    all_seen_vals = sorted(src_stats["unique_vals"] | tgt_stats["unique_vals"])
    print(f"[global] raw label values seen: {all_seen_vals}")
    allowed_vals = set(range(0, num_classes + 1))
    bad_vals = [v for v in all_seen_vals if v not in allowed_vals]
    if bad_vals:
        raise ValueError(
            f"Found unexpected raw label values {bad_vals}. "
            f"Expected raw labels in [0, {num_classes}] with 0 reserved for ignore."
        )

    transforms = load_cyclenet_transforms(config.data.transform_id, config.data.image_size)

    sim_ds = CycleDomainSegDataset(
        data_dir=config.data.src_dir,
        domain_idx=0,
        transforms=transforms,
        num_classes=num_classes,
        rgb_parent_dir=rgb_parent_dir,
        label_parent_dir=label_parent_dir,
    )
    real_ds = CycleDomainSegDataset(
        data_dir=config.data.tgt_dir,
        domain_idx=1,
        transforms=transforms,
        num_classes=num_classes,
        rgb_parent_dir=rgb_parent_dir,
        label_parent_dir=label_parent_dir,
    )

    inspect_dataset(sim_ds, "sim", num_classes, args.num_samples, save_debug_dir)
    inspect_dataset(real_ds, "real", num_classes, args.num_samples, save_debug_dir)

    inspect_batch(sim_ds, args.batch_size, num_classes, "CycleDomainSegDataset(sim)")
    inspect_batch(real_ds, args.batch_size, num_classes, "CycleDomainSegDataset(real)")

    source_ds = SourceSegDataset(
        src_dir=config.data.src_dir,
        image_size=config.data.image_size,
        num_classes=num_classes,
        rgb_parent_dir=rgb_parent_dir,
        label_parent_dir=label_parent_dir,
    )
    inspect_source_dataset(source_ds, num_classes, args.num_samples, save_debug_dir)
    inspect_batch(source_ds, args.batch_size, num_classes, "SourceSegDataset")

    if not args.skip_translate:
        translate_ds = TranslateSegDataset(
            src_dir=config.data.src_dir,
            image_size=config.data.image_size,
            num_classes=num_classes,
            rgb_parent_dir=rgb_parent_dir,
            label_parent_dir=label_parent_dir,
        )
        inspect_translate_dataset(translate_ds, num_classes, args.num_samples, save_debug_dir)
        inspect_batch(translate_ds, args.batch_size, num_classes, "TranslateSegDataset")

    print("\nAll segmentation-conditioning checks passed.")


if __name__ == "__main__":
    main()
