import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from albumentations import Compose

from .transforms import load_unet_transforms, load_cyclenet_transforms, load_source_transforms


class DomainDataset(Dataset):
    def __init__(self, data_dir: str, domain_idx: int, transforms: Compose):
        # ----------
        # Store domain paths with domain index
        # ----------
        self.samples = []

        for path in Path(data_dir).rglob("*"):
            if path.suffix.lower() in {".jpg", ".png"}:
                self.samples.append(path)
            
        self.domain_idx = domain_idx
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_np = np.array(Image.open(self.samples[idx]).convert("RGB"))
        img = self.transforms(image=img_np)["image"]
        return img, torch.tensor(self.domain_idx, dtype=torch.long)
    

class CycleDomainDataset(Dataset):
    """
    Like DomainDataset, but returns (img, src_idx, tgt_idx) for CycleNetTrainer.
    """
    def __init__(self, data_dir: str, domain_idx: int, transforms: Compose):
        self.samples = []
        for path in Path(data_dir).rglob("*"):
            if path.suffix.lower() in {".jpg", ".png"}:
                self.samples.append(path)

        self.domain_idx = int(domain_idx)
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_np = np.array(Image.open(self.samples[idx]).convert("RGB"))
        img = self.transforms(image=img_np)["image"]

        src_idx = torch.tensor(self.domain_idx, dtype=torch.long)
        tgt_idx = torch.tensor(1 - self.domain_idx, dtype=torch.long)

        return img, src_idx, tgt_idx


class UNetDataset(Dataset):
    def __init__(self, src_dir: str, tgt_dir: str, transform_id: int = 0, image_size: int = 224):
        # -------------------------
        # Store domain src/tgt paths with domain indices
        # -------------------------
        self.samples = []

        # -- Source: 0
        for path in Path(src_dir).rglob("*"):
            if path.suffix.lower() in {".jpg", ".png"}:
                self.samples.append((path, 0))

        # -- Target: 1
        for path in Path(tgt_dir).rglob("*"):
            if path.suffix.lower() in {".jpg", ".png"}:
                self.samples.append((path, 1))

        # -------------------------
        # Define transforms
        # -------------------------
        self.transforms = load_unet_transforms(transform_id, image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, d_idx = self.samples[idx]
        img_np = np.array(Image.open(filepath).convert("RGB"))
        img = self.transforms(image=img_np)["image"]

        return img, torch.tensor(d_idx, dtype=torch.long)


class CycleNetDataset(Dataset):
    def __init__(self, src_dir: str, tgt_dir: str, transform_id: int = 0, image_size: int = 224):
        # -------------------------
        # Store domain src/tgt paths with domain indices
        # -------------------------
        self.samples = []

        # -- Source: 0
        for path in Path(src_dir).rglob("*"):
            if path.suffix.lower() in {".jpg", ".png"}:
                self.samples.append((path, 0))

        # -- Target: 1
        for path in Path(tgt_dir).rglob("*"):
            if path.suffix in {".jpg", ".png"}:
                self.samples.append((path, 1))

        # -------------------------
        # Define transforms
        # -------------------------
        self.transforms = load_cyclenet_transforms(transform_id, image_size)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        # -- Load image / apply transforms
        filepath, src_idx = self.samples[idx]
        img_np = np.array(Image.open(filepath).convert("RGB"))
        img = self.transforms(image=img_np)["image"]

        # -- Invert src_idx for tgt_idx
        src_idx = torch.tensor(src_idx, dtype=torch.long)
        tgt_idx = 1 - src_idx

        return img, src_idx, tgt_idx


class SourceDataset(Dataset):
    def __init__(self, src_dir: str, image_size: int = 224):
        # -------------------------
        # Store all images in src_dir
        # -------------------------
        self.samples = []

        for path in Path(src_dir).rglob("*"):
            if path.suffix.lower() in {".jpg", ".png"}:
                self.samples.append(path)

        # -------------------------
        # Define transforms
        # -------------------------
        self.transforms = load_source_transforms(image_size)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        filepath = self.samples[idx]
        img_np = np.array(Image.open(filepath).convert("RGB"))
        img = self.transforms(image=img_np)["image"]

        return img