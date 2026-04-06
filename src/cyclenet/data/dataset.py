import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from albumentations import Compose

from .transforms import load_unet_transforms, load_cyclenet_transforms, load_source_transforms


class DomainDataset(Dataset):
    def __init__(
        self, 
        data_dir: str, 
        domain_idx: int, 
        transforms: Compose, 
        file_exts: set[str] = {".jpg", ".png", ".tif", ".tiff"},
        parent_dirs: set[str] | None = None,
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
            if parent_dirs is not None:
                if path.parent.name not in parent_dirs:
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
        parent_dirs: set[str] | None = None,
    ):
        self.samples = []
        for path in sorted(Path(data_dir).rglob("*")):
            # -- Check file extension
            if path.suffix.lower() in file_exts:
                self.samples.append(path)

            # -- Check allowed parent directory
            if parent_dirs is not None:
                if path.parent.name not in parent_dirs:
                    continue

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


class UNetDataset(Dataset):
    def __init__(self, src_dir: str, tgt_dir: str, transform_id: int = 0, image_size: int = 224):
        # -------------------------
        # Store domain src/tgt paths with domain indices
        # -------------------------
        self.samples = []

        # -- Source: 0
        for path in sorted(Path(src_dir).rglob("*")):
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
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]

        return img, torch.tensor(d_idx, dtype=torch.long)


class CycleNetDataset(Dataset):
    def __init__(self, src_dir: str, tgt_dir: str, transform_id: int = 0, image_size: int = 224):
        # -------------------------
        # Store domain src/tgt paths with domain indices
        # -------------------------
        self.samples = []

        # -- Source: 0
        for path in sorted(Path(src_dir).rglob("*")):
            if path.suffix.lower() in {".jpg", ".png"}:
                self.samples.append((path, 0))

        # -- Target: 1
        for path in sorted(Path(tgt_dir).rglob("*")):
            if path.suffix.lower() in {".jpg", ".png"}:
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
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]

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

        for path in sorted(Path(src_dir).rglob("*")):
            if path.suffix.lower() in {".jpg", ".png"}:
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
    

class TranslateDataset(Dataset):
    def __init__(self, src_dir: str, image_size: int = 224):
        # -------------------------
        # Store all images in src_dir
        # -------------------------
        self.samples = []

        for path in sorted(Path(src_dir).rglob("*")):
            if path.suffix.lower() in {".jpg", ".png"}:
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