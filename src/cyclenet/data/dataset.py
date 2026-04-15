import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from albumentations import Compose

from .transforms import load_source_transforms


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


def to_one_hot_with_ignore(mask: torch.Tensor, num_classes: int = 8, ignore_value: int = 0) -> torch.Tensor:
    """
    Raw mask values:
      0 = ignore
      1..8 = valid semantic classes

    Output:
      seg: (8, H, W) float
      - ignore pixels become all zeros
      - class 1 maps to channel 0
      - class 8 maps to channel 7
    """
    mask = mask.long()

    h, w = mask.shape
    seg = torch.zeros((num_classes, h, w), dtype=torch.float32)

    valid = mask != ignore_value
    if valid.any():
        class_idx = mask[valid] - 1  # 1..8 -> 0..7

        if class_idx.min() < 0 or class_idx.max() >= num_classes:
            bad_vals = torch.unique(mask[(mask != ignore_value) & ((mask < 1) | (mask > num_classes))])
            raise ValueError(f"Found out-of-range label values: {bad_vals.tolist()}")

        seg[class_idx, valid] = 1.0

    return seg


class DomainDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        domain_idx: int, 
        transforms: Compose, 
        file_exts: set[str] = {".jpg", ".png", ".tif", ".tiff"},
        rgb_parent_dirs: set[str] | None = None,
    ):
        # ----------
        # Store domain paths with domain index
        # ----------
        self.samples = []

        for path in sorted(Path(data_dir).rglob("*")):
            # -- Check file extension
            if path.suffix.lower() not in file_exts:
                continue

            # -- Check allowed parent directory
            if rgb_parent_dirs is not None:
                if path.parent.name not in rgb_parent_dirs:
                    continue
            
            # -- Append sample if all checks pass
            self.samples.append(path)
            
        self.domain_idx = domain_idx
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img = cv2.imread(self.samples[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]
        return img, torch.tensor(self.domain_idx, dtype=torch.long)
    

class CycleDomainDataset(Dataset):
    """
    Like DomainDataset, but returns (img, src_idx, tgt_idx) for CycleNetTrainer.
    """
    def __init__(
        self, 
        data_dir: str, 
        domain_idx: int, 
        transforms: Compose, 
        file_exts: set[str] = {".jpg", ".png", ".tif", ".tiff"},
        rgb_parent_dirs: set[str] | None = None,
    ):
        self.samples = []
        for path in sorted(Path(data_dir).rglob("*")):
            # -- Check file extension
            if path.suffix.lower() not in file_exts:
                continue

            # -- Check allowed parent directory
            if rgb_parent_dirs is not None:
                if path.parent.name not in rgb_parent_dirs:
                    continue

            self.samples.append(path)

        self.domain_idx = int(domain_idx)
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.samples[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]

        src_idx = torch.tensor(self.domain_idx, dtype=torch.long)
        tgt_idx = torch.tensor(1 - self.domain_idx, dtype=torch.long)

        return img, src_idx, tgt_idx
    

class CycleDomainSegDataset(Dataset):
    """
    Returns (img, seg, src_idx, tgt_idx) for CycleNetTrainer, including images'
    corresponding segmentation masks
    """
    def __init__(
        self,
        data_dir: str,
        domain_idx: int,
        transforms: Compose,
        num_classes: int,
        file_exts: set[str] = {".jpg", ".png", ".tif", ".tiff"},
        rgb_parent_dirs: set[str] = {"opt", "pre_opt"},
        label_parent_dir: str = "gt_ss_mask",
    ):
        self.samples = []
        self.domain_idx = int(domain_idx)
        self.transforms = transforms
        self.num_classes = num_classes
        self.rgb_parent_dirs = rgb_parent_dirs
        self.label_parent_dir = label_parent_dir

        for path in sorted(Path(data_dir).rglob("*")):
            # -- Check file extension
            if path.suffix.lower() not in file_exts:
                continue

            # -------------------------
            # Derive label path from RGB filenames
            # -------------------------
            if path.parent.name not in rgb_parent_dirs:
                continue

            label_path = path.parent.parent / self.label_parent_dir / path.name
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {path}: expected {label_path}")

            self.samples.append((path, label_path))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        rgb_path, label_path = self.samples[idx]

        img = load_rgb(rgb_path)
        mask = load_label_mask(label_path)

        transformed = self.transforms(image=img, mask=mask)

        img = transformed["image"]
        mask = transformed["mask"].long()
        seg = to_one_hot_with_ignore(mask, self.num_classes, ignore_value=0)

        src_idx = torch.tensor(self.domain_idx, dtype=torch.long)
        tgt_idx = torch.tensor(1 - self.domain_idx, dtype=torch.long)

        return img, seg, src_idx, tgt_idx


class SourceDataset(Dataset):
    def __init__(
        self, 
        src_dir: str, 
        image_size: int = 224,
        file_exts: set[str] = {".jpg", ".png", ".tif", ".tiff"},
        rgb_parent_dirs: set[str] | None = None,
    ):
        # -------------------------
        # Store all images in src_dir
        # -------------------------
        self.samples = []

        for path in sorted(Path(src_dir).rglob("*")):
            # -- Check file extension
            if path.suffix.lower() not in file_exts:
                continue

            # -- Check allowed parent directory
            if rgb_parent_dirs is not None:
                if path.parent.name not in rgb_parent_dirs:
                    continue

            self.samples.append(path)

        # -------------------------
        # Define transforms
        # -------------------------
        self.transforms = load_source_transforms(image_size)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        filepath = self.samples[idx]
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]

        return img
    

class SourceSegDataset(Dataset):
    """
    
    """
    def __init__(
        self,
        src_dir: str,
        image_size: int,
        num_classes: int,
        file_exts: set[str] = {".jpg", ".png", ".tif", ".tiff"},
        rgb_parent_dirs: set[str] = {"opt", "pre_opt"},
        label_parent_dir: str = "gt_ss_mask",
    ):
        self.samples = []
        self.src_dir = Path(src_dir)
        self.rgb_parent_dirs = rgb_parent_dirs
        self.label_parent_dir = label_parent_dir
        self.transforms = load_source_transforms(image_size)
        self.num_classes = num_classes

        for path in sorted(self.src_dir.rglob("*")):
            # -- Check file extension
            if path.suffix.lower() not in file_exts:
                continue

            # -------------------------
            # Derive label path from RGB filenames
            # -------------------------
            if path.parent.name not in rgb_parent_dirs:
                continue

            label_path = path.parent.parent / self.label_parent_dir / path.name
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {path}: expected {label_path}")
            
            self.samples.append((path, label_path))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        rgb_path, label_path = self.samples[idx]

        img = load_rgb(rgb_path)
        mask = load_label_mask(label_path)

        transformed = self.transforms(image=img, mask=mask)

        img = transformed["image"]
        mask = transformed["mask"].long()
        seg = to_one_hot_with_ignore(mask, self.num_classes, ignore_value=0)

        return img, seg
    

class TranslateDataset(Dataset):
    def __init__(
        self, 
        src_dir: str, 
        image_size: int = 224,
        file_exts: set[str] = {".jpg", ".png", ".tif", ".tiff"},
        rgb_parent_dirs: set[str] = {"opt"},
    ):
        # -------------------------
        # Store all images in src_dir
        # -------------------------
        self.samples = []

        for path in sorted(Path(src_dir).rglob("*")):
            # -- Check file extension
            if path.suffix.lower() not in file_exts:
                continue

            # -- Check allowed parent directory
            if rgb_parent_dirs is not None:
                if path.parent.name not in rgb_parent_dirs:
                    continue

            self.samples.append(path)

        self.samples = sorted(self.samples)

        # -------------------------
        # Define transforms
        # -------------------------
        self.transforms = load_source_transforms(image_size)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        filepath = self.samples[idx]
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]

        return img, str(filepath)
    

class TranslateSegDataset(Dataset):
    """
    
    """
    def __init__(
        self,
        src_dir: str,
        image_size: int,
        num_classes: int,
        file_exts: set[str] = {".jpg", ".png", ".tif", ".tiff"},
        rgb_parent_dirs: set[str] = {"opt"},
        label_parent_dir: str = "gt_ss_mask",
    ):
        self.samples = []
        self.src_dir = Path(src_dir)
        self.rgb_parent_dirs = rgb_parent_dirs
        self.label_parent_dir = label_parent_dir
        self.transforms = load_source_transforms(image_size)
        self.num_classes = num_classes

        for path in sorted(self.src_dir.rglob("*")):
            # -- Check file extension
            if path.suffix.lower() not in file_exts:
                continue

            # -------------------------
            # Derive label path from RGB filenames
            # -------------------------
            if path.parent.name not in rgb_parent_dirs:
                continue

            label_path = path.parent.parent / self.label_parent_dir / path.name
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {path}: expected {label_path}")
            
            self.samples.append((path, label_path))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        rgb_path, label_path = self.samples[idx]

        img = load_rgb(rgb_path)
        mask = load_label_mask(label_path)

        transformed = self.transforms(image=img, mask=mask)

        img = transformed["image"]
        mask = transformed["mask"].long()
        seg = to_one_hot_with_ignore(mask, self.num_classes, ignore_value=0)

        return img, seg, str(rgb_path)
