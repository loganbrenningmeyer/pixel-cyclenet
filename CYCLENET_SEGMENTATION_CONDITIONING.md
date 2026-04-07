# CycleNet Segmentation Conditioning Implementation Guide

This document explains how to extend the current CycleNet implementation so ControlNet is conditioned on:

- source RGB image
- source land-cover segmentation map as an 8-channel one-hot tensor

The goal is to improve semantic/label consistency during translation.

## Recommendation

For this repository, the most practical first implementation is:

- keep the existing RGB condition
- concatenate an 8-channel segmentation condition
- pass `c_img` with shape `(B, 11, H, W)` into ControlNet

This is the lowest-risk change because the current `ControlNet` already accepts a generic `in_ch` and only uses the condition through its own conditioning stem.

The intended condition becomes:

```python
c_img = torch.cat([rgb_cond, seg_cond], dim=1)  # (B, 3 + 8, H, W)
```

Where:

- `rgb_cond` is RGB normalized to `[0, 1]`
- `seg_cond` is an 8-channel one-hot tensor, or optionally U-Net softmax probabilities if that is what will be available at inference

## Important Design Decision

The current loss does not only use the source RGB image as condition. It also reuses the predicted target image `y_0_cond` as the conditioning image in the cycle/invariance terms.

Today the loss uses:

```python
x_0_ctrl = ((x_0 + 1.0) / 2.0).clamp(0.0, 1.0)
y_0_cond = ((y_0.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
```

If segmentation is added, the cleanest approach is:

- keep RGB changing across source/target-style passes
- keep the segmentation map fixed

That means:

```python
c_x = cat([x_0_ctrl, seg_0], dim=1)
c_y = cat([y_0_cond, seg_0], dim=1)
```

The segmentation should stay invariant across the cycle because the translation should change style/domain appearance, not land-cover semantics.

## Files That Need To Change

These are the main files affected:

- `src/cyclenet/models/controlnet.py`
- `src/cyclenet/train_cyclenet.py`
- `src/cyclenet/translate_cyclenet.py`
- `src/cyclenet/training/cyclenet_trainer.py`
- `src/cyclenet/diffusion/losses.py`
- `src/cyclenet/data/dataset.py`
- `src/cyclenet/data/transforms.py`
- `src/cyclenet/data/__init__.py`
- `configs/cyclenet/train_cyclenet.yaml`
- `configs/cyclenet/translate_cyclenet.yaml`

## 1. ControlNet Input Channels

### Current state

`ControlNet` already supports arbitrary `in_ch`:

```python
self.c_stem = nn.Sequential(
    nn.Conv2d(in_ch, self.base_ch, kernel_size=3, stride=1, padding=1),
    nn.GroupNorm(32, self.base_ch),
    nn.SiLU(),
)
```

So the model architecture does not need a structural rewrite. The main change is instantiating it with `in_ch=11` instead of `3`.

### Change in `src/cyclenet/train_cyclenet.py`

```python
num_seg_classes = config.model.num_seg_classes
control_in_ch = 3 + num_seg_classes

control = ControlNet(backbone, in_ch=control_in_ch).to(device)
```

### Change in `src/cyclenet/translate_cyclenet.py`

```python
num_seg_classes = cyclenet_config.model.num_seg_classes
control_in_ch = 3 + num_seg_classes

control = ControlNet(backbone, in_ch=control_in_ch).to(device)
```

## 2. Add Conditioning Config

Add explicit config fields so the model and data pipeline know segmentation is part of the condition.

### `configs/cyclenet/train_cyclenet.yaml`

```yaml
model:
  recon_weight: 1.0
  cycle_weight: 0.005
  consis_weight: 0.1
  invar_weight: 0.1

  control_condition: rgb_seg
  num_seg_classes: 8

data:
  src_dir: /path/to/src/rgb
  tgt_dir: /path/to/tgt/rgb
  src_label_dir: /path/to/src/labels
  tgt_label_dir: /path/to/tgt/labels
  image_size: 224
  transform_id: 1
```

### `configs/cyclenet/translate_cyclenet.yaml`

```yaml
data:
  src_dir: /path/to/src/rgb
  src_label_dir: /path/to/src/labels
  src_idx: 0
  out_dir: /path/to/output
  image_size: 224
```

If you do not have labels for one of the domains, see the section `If Labels Only Exist For One Domain`.

## 3. Dataset Changes

The existing datasets only return RGB images. You need paired RGB + label loading.

### Requirements

- geometry transforms must be applied identically to RGB and label
- labels must use nearest-neighbor interpolation
- labels should be returned as integer class IDs before conversion to one-hot

### Suggested helpers in `src/cyclenet/data/dataset.py`

```python
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset


def load_rgb(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_label_mask(path: Path) -> torch.Tensor:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read label mask: {path}")

    if mask.ndim == 3:
        mask = mask[..., 0]

    return mask


def to_one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    # mask: (H, W) long -> one_hot: (C, H, W) float
    mask = mask.long()
    one_hot = F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()
    return one_hot
```

### Suggested training dataset

This version assumes RGB and labels share the same relative filename under separate roots.

```python
class CycleDomainSegDataset(Dataset):
    def __init__(
        self,
        rgb_dir: str,
        label_dir: str,
        domain_idx: int,
        transforms,
        num_classes: int,
        file_exts: set[str] = {".jpg", ".png", ".tif", ".tiff"},
    ):
        self.samples = []
        self.domain_idx = int(domain_idx)
        self.transforms = transforms
        self.num_classes = num_classes
        self.rgb_dir = Path(rgb_dir)
        self.label_dir = Path(label_dir)

        for path in sorted(self.rgb_dir.rglob("*")):
            if path.suffix.lower() not in file_exts:
                continue

            rel_path = path.relative_to(self.rgb_dir)
            label_path = self.label_dir / rel_path

            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {path}: expected {label_path}")

            self.samples.append((path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rgb_path, label_path = self.samples[idx]

        image = load_rgb(rgb_path)
        mask = load_label_mask(label_path)

        transformed = self.transforms(image=image, mask=mask)

        image = transformed["image"]
        mask = transformed["mask"].long()
        seg = to_one_hot(mask, self.num_classes)

        src_idx = torch.tensor(self.domain_idx, dtype=torch.long)
        tgt_idx = torch.tensor(1 - self.domain_idx, dtype=torch.long)

        return image, seg, src_idx, tgt_idx
```

### Suggested source/translation datasets

These are needed because sample generation during training and inference both need labels.

```python
class SourceSegDataset(Dataset):
    def __init__(self, src_dir: str, src_label_dir: str, image_size: int, num_classes: int):
        self.samples = []
        self.src_dir = Path(src_dir)
        self.src_label_dir = Path(src_label_dir)
        self.transforms = load_source_pair_transforms(image_size)
        self.num_classes = num_classes

        for path in sorted(self.src_dir.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".png", ".tif", ".tiff"}:
                continue

            rel_path = path.relative_to(self.src_dir)
            label_path = self.src_label_dir / rel_path
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {path}: expected {label_path}")

            self.samples.append((path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rgb_path, label_path = self.samples[idx]
        image = load_rgb(rgb_path)
        mask = load_label_mask(label_path)

        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].long()
        seg = to_one_hot(mask, self.num_classes)

        return image, seg


class TranslateSegDataset(Dataset):
    def __init__(self, src_dir: str, src_label_dir: str, image_size: int, num_classes: int):
        self.samples = []
        self.src_dir = Path(src_dir)
        self.src_label_dir = Path(src_label_dir)
        self.transforms = load_source_pair_transforms(image_size)
        self.num_classes = num_classes

        for path in sorted(self.src_dir.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".png", ".tif", ".tiff"}:
                continue

            rel_path = path.relative_to(self.src_dir)
            label_path = self.src_label_dir / rel_path
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {path}: expected {label_path}")

            self.samples.append((path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        rgb_path, label_path = self.samples[idx]
        image = load_rgb(rgb_path)
        mask = load_label_mask(label_path)

        transformed = self.transforms(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"].long()
        seg = to_one_hot(mask, self.num_classes)

        return image, seg, str(rgb_path)
```

### Export changes in `src/cyclenet/data/__init__.py`

```python
from .dataset import (
    UNetDataset,
    CycleNetDataset,
    SourceDataset,
    CycleDomainSegDataset,
    SourceSegDataset,
    TranslateSegDataset,
)

__all__ = [
    "UNetDataset",
    "CycleNetDataset",
    "SourceDataset",
    "CycleDomainSegDataset",
    "SourceSegDataset",
    "TranslateSegDataset",
]
```

## 4. Transform Changes

Image-only transforms are not enough anymore. You need pair transforms that apply the same spatial augmentation to image and mask.

### Key rules

- RGB resize: bilinear
- segmentation resize: nearest-neighbor
- photometric augmentations must apply to image only
- segmentation must not be normalized like RGB

### Suggested implementation in `src/cyclenet/data/transforms.py`

```python
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_cyclenet_pair_transforms(transform_id: int = 0, image_size: int = 224) -> A.Compose:
    mean = tuple([0.5] * 3)
    std = tuple([0.5] * 3)

    if transform_id == 0:
        return A.Compose(
            [
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    elif transform_id == 1:
        return A.Compose(
            [
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    elif transform_id == 3:
        return A.Compose(
            [
                A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    else:
        raise ValueError("Unsupported transform_id")


def load_source_pair_transforms(image_size: int = 224) -> A.Compose:
    mean = tuple([0.5] * 3)
    std = tuple([0.5] * 3)

    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
```

## 5. Build A Shared Conditioning Tensor

It is worth centralizing control-condition construction so training and inference use the exact same layout.

### Suggested helper

Add a helper in either `src/cyclenet/diffusion/losses.py` or a small new utility module:

```python
def build_control_condition(rgb_pm1: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """
    rgb_pm1: (B, 3, H, W) in [-1, 1]
    seg:     (B, Cseg, H, W) in {0, 1} or probabilities
    """
    rgb_01 = ((rgb_pm1 + 1.0) / 2.0).clamp(0.0, 1.0)
    return torch.cat([rgb_01, seg.float()], dim=1)
```

This avoids duplicating the RGB normalization logic in multiple places.

## 6. Loss Changes

This is the most important logic change.

The existing `cyclenet_loss` signature is:

```python
def cyclenet_loss(model, x_0, t, src_idx, tgt_idx, sched):
```

It should become:

```python
def cyclenet_loss(model, x_0, seg_0, t, src_idx, tgt_idx, sched):
```

### Suggested update in `src/cyclenet/diffusion/losses.py`

```python
def build_control_condition(rgb_pm1: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    rgb_01 = ((rgb_pm1 + 1.0) / 2.0).clamp(0.0, 1.0)
    return torch.cat([rgb_01, seg.float()], dim=1)


def cyclenet_loss(
    model: CycleNet,
    x_0: torch.Tensor,
    seg_0: torch.Tensor,
    t: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    sched: DiffusionSchedule,
) -> dict[str, torch.Tensor]:
    eps_x = torch.randn_like(x_0)
    x_t = q_sample(x_0, t, eps_x, sched)

    c_x = build_control_condition(x_0, seg_0)

    eps_xt_x2x_x0 = model.forward(
        x_t=x_t,
        t=t,
        from_idx=src_idx,
        to_idx=src_idx,
        c_img=c_x,
    )
    recon_loss = F.mse_loss(eps_xt_x2x_x0, eps_x)

    eps_xt_x2y_x0 = model.forward(
        x_t=x_t,
        t=t,
        from_idx=src_idx,
        to_idx=tgt_idx,
        c_img=c_x,
        no_unet_grad=True,
    )

    y_0 = x0_from_eps(x_t, t, eps_xt_x2y_x0, sched)
    c_y = build_control_condition(y_0.detach(), seg_0)

    eps_y = torch.randn_like(y_0)
    y_t = q_sample(y_0.detach(), t, eps_y, sched)
    y_t_c = q_sample(y_0, t, eps_y, sched)

    eps_yt_y2x_y0 = model.forward(
        x_t=y_t_c,
        t=t,
        from_idx=tgt_idx,
        to_idx=src_idx,
        c_img=c_y,
    )
    cycle_loss = F.mse_loss((eps_xt_x2y_x0.detach() + eps_yt_y2x_y0), (eps_x + eps_y))

    eps_yt_x2x_x0 = model.forward(
        x_t=y_t,
        t=t,
        from_idx=src_idx,
        to_idx=src_idx,
        c_img=c_x,
    )
    consis_loss = F.mse_loss((eps_xt_x2y_x0.detach() + eps_yt_x2x_x0), (eps_x + eps_y))

    eps_xt_y2y_y0 = model.forward(
        x_t=x_t,
        t=t,
        from_idx=tgt_idx,
        to_idx=tgt_idx,
        c_img=c_y,
    )
    invar_loss = F.mse_loss(eps_xt_x2y_x0, eps_xt_y2y_y0.detach())

    return {
        "recon": recon_loss,
        "cycle": cycle_loss,
        "consis": consis_loss,
        "invar": invar_loss,
    }
```

### Why reuse `seg_0` for both `c_x` and `c_y`

Because the semantic layout should remain the same under translation.

If the source image contains:

- road in pixel region A
- building in pixel region B

Then the translated image should preserve those class assignments even if style/appearance changes.

## 7. Trainer Changes

The trainer currently expects batches of:

```python
x_0, src_idx, tgt_idx
```

It must change to:

```python
x_0, seg_0, src_idx, tgt_idx
```

### Update `train_step` in `src/cyclenet/training/cyclenet_trainer.py`

```python
def train_step(
    self,
    x_0: torch.Tensor,
    seg_0: torch.Tensor,
    src_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    self.optimizer.zero_grad()

    with autocast(device_type="cuda"):
        B = x_0.shape[0]
        t = torch.randint(0, self.sched.T, (B,), device=self.device)

        loss_dict = cyclenet_loss(
            model=self.model,
            x_0=x_0,
            seg_0=seg_0,
            t=t,
            src_idx=src_idx,
            tgt_idx=tgt_idx,
            sched=self.sched,
        )

        loss = (
            self.recon_weight * loss_dict["recon"]
            + self.cycle_weight * loss_dict["cycle"]
            + self.consis_weight * loss_dict["consis"]
            + self.invar_weight * loss_dict["invar"]
        )
```

### Update the training loop

```python
for x_0, seg_0, src_idx, tgt_idx in tqdm(self.dataloader, desc=f"Epoch {epoch}, Step {step}", unit="Batch"):
    if step > steps:
        break

    x_0 = x_0.to(self.device, non_blocking=True)
    seg_0 = seg_0.to(self.device, non_blocking=True)
    src_idx = src_idx.to(self.device, non_blocking=True)
    tgt_idx = tgt_idx.to(self.device, non_blocking=True)

    loss_dict, loss = self.train_step(x_0, seg_0, src_idx, tgt_idx)
```

### Update sample generation during training

Current sample generation only loads RGB source samples. That is not enough anymore.

Use `SourceSegDataset` and update:

```python
x_src, seg_src = self._next_sample_batch()
c_img = build_control_condition(x_src, seg_src)
```

Then pass:

```python
c_img=c_img
```

into both DDPM and DDIM calls.

### Update `_next_sample_batch`

```python
def _next_sample_batch(self):
    if not self.is_main or self.sample_iter is None:
        return None

    try:
        x_src, seg_src = next(self.sample_iter)
    except StopIteration:
        self.sample_iter = iter(self.sample_loader)
        x_src, seg_src = next(self.sample_iter)

    return (
        x_src.to(self.device, non_blocking=True),
        seg_src.to(self.device, non_blocking=True),
    )
```

## 8. Training Script Changes

### Update imports in `src/cyclenet/train_cyclenet.py`

```python
from cyclenet.data import CycleDomainSegDataset, SourceSegDataset, DomainSampler, load_cyclenet_pair_transforms
```

### Update dataset construction

```python
num_seg_classes = config.model.num_seg_classes
transforms = load_cyclenet_pair_transforms(config.data.transform_id, config.data.image_size)

real_ds = CycleDomainSegDataset(
    rgb_dir=config.data.tgt_dir,
    label_dir=config.data.tgt_label_dir,
    domain_idx=1,
    transforms=transforms,
    num_classes=num_seg_classes,
)

sim_ds = CycleDomainSegDataset(
    rgb_dir=config.data.src_dir,
    label_dir=config.data.src_label_dir,
    domain_idx=0,
    transforms=transforms,
    num_classes=num_seg_classes,
)
```

### Update sample dataset construction

```python
sample_dataset = SourceSegDataset(
    src_dir=config.data.src_dir,
    src_label_dir=config.data.src_label_dir,
    image_size=config.data.image_size,
    num_classes=num_seg_classes,
)
```

### Update ControlNet initialization

```python
control = ControlNet(backbone, in_ch=3 + num_seg_classes).to(device)
```

## 9. Translation Script Changes

The translation pipeline must receive segmentation conditions too. Otherwise training and inference will not match.

### Update imports in `src/cyclenet/translate_cyclenet.py`

```python
from cyclenet.data import TranslateSegDataset
from cyclenet.diffusion.losses import build_control_condition
```

### Update dataset construction

```python
num_seg_classes = cyclenet_config.model.num_seg_classes

dataset = TranslateSegDataset(
    src_dir=translate_config.data.src_dir,
    src_label_dir=translate_config.data.src_label_dir,
    image_size=translate_config.data.image_size,
    num_classes=num_seg_classes,
)
```

### Update loop inputs

```python
for x_src, seg_src, filepaths in loader_iter:
    B = x_src.shape[0]

    src_idx = torch.full((B,), fill_value=src_domain_idx, device=device, dtype=torch.long)
    tgt_idx = torch.full((B,), fill_value=tgt_domain_idx, device=device, dtype=torch.long)

    x_src = x_src.to(device, non_blocking=True)
    seg_src = seg_src.to(device, non_blocking=True)
    c_img = build_control_condition(x_src, seg_src)
```

### Update sampling calls

```python
samples, _ = cyclenet_ddim_loop(
    model=model,
    x_src=x_src,
    src_idx=src_idx,
    tgt_idx=tgt_idx,
    c_img=c_img,
    sched=sched,
    w=cfg_weight,
    strength=noise_strength,
    num_steps=num_steps,
    eta=eta,
)
```

## 10. Optional: Use U-Net Softmax Instead Of Hard One-Hot

If your deployed system will use the trained U-Net to generate segmentation conditions at inference, it can be better to train ControlNet on soft segmentation probabilities instead of perfect one-hot labels.

That looks like:

```python
with torch.no_grad():
    logits = seg_model(image)
    seg_prob = torch.softmax(logits, dim=1)

c_img = torch.cat([rgb_01, seg_prob], dim=1)
```

This reduces train/inference mismatch if ground-truth labels are not available at inference.

Practical recommendation:

- if inference has GT segmentation: use one-hot in training and inference
- if inference uses predicted U-Net segmentation: mix GT one-hot and predicted softmax during training, or train directly with predicted softmax

## 11. Checkpoint Compatibility

Changing ControlNet input channels from `3` to `11` changes the shape of `control.c_stem.0.weight`.

That means:

- existing CycleNet checkpoints will not load strictly into the new architecture
- your pretrained UNet backbone still loads fine
- CycleNet/ControlNet training must restart from a fresh ControlNet initialization

The backbone and domain embedding can still be loaded exactly as before.

## 12. Data Assumptions You Must Make Explicit

Before implementing, define these clearly:

1. How RGB files map to label files.
2. Whether both domains have labels.
3. Whether label masks are stored as:
   - single-channel integer IDs in `[0, 7]`
   - color-coded PNGs
4. Whether inference uses:
   - ground-truth labels
   - U-Net predictions

If the masks are color-coded, add a conversion step from RGB palette to class IDs before one-hot encoding.

## 13. If Labels Only Exist For One Domain

This is the main caveat.

Your current CycleNet training is symmetric across both domains. Each sample can act as `x_0`, and the loss expects a valid condition for whichever domain the sample comes from.

If only source-domain images have labels:

- the straightforward symmetric loss no longer has all required inputs
- you need either:
  - pseudo-labels for the unlabeled domain
  - or an asymmetric training rule that only uses segmentation-conditioned ControlNet on the labeled side

Best practice if only one domain is labeled:

- generate pseudo-labels for the other domain with the U-Net
- store them and train the same pipeline on both domains

That keeps the implementation simple and preserves the current loss design.

## 14. Minimal Implementation Path

If you want the smallest coherent change set, implement in this order:

1. Add config fields for `num_seg_classes`, `src_label_dir`, and `tgt_label_dir`.
2. Add paired RGB+mask datasets.
3. Add paired transforms with nearest-neighbor mask resize.
4. Change `ControlNet` init from `in_ch=3` to `in_ch=11`.
5. Add `build_control_condition`.
6. Update `cyclenet_loss` to accept `seg_0` and build `c_x` and `c_y`.
7. Update trainer batches and sample generation to carry segmentation.
8. Update translation to require source labels.

## 15. Final Recommendation

For this repository, the best first version is:

- `c_img = concat(RGB[0,1], segmentation_one_hot)`
- fixed segmentation map across cycle/invariance passes
- paired RGB+mask transforms
- labels for both domains, ideally GT or pseudo-labels if GT is missing

That gives you the strongest semantic prior with the smallest architectural change.

If you later want a cleaner architecture, the next step would be:

- separate RGB stem
- separate segmentation stem
- fuse them before the zero-conv injection path

But that is optional. Concatenation is the right first implementation here.
