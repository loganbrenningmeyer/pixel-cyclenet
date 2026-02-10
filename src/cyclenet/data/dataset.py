import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class UNetDataset(Dataset):
    def __init__(self, src_dir: str, tgt_dir: str, image_size: int = 224):
        # -------------------------
        # Store domain src/tgt paths with domain indices
        # -------------------------
        self.samples = []

        # -- Source: 0
        for path in Path(src_dir).rglob("*"):
            if path.suffix in {".jpg", ".png"}:
                self.samples.append((path, 0))

        # -- Target: 1
        for path in Path(tgt_dir).rglob("*"):
            if path.suffix in {".jpg", ".png"}:
                self.samples.append((path, 1))

        # -------------------------
        # Define transforms
        # -------------------------
        self.transforms = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, d_idx = self.samples[idx]
        img = Image.open(filepath).convert("RGB")
        img = self.transforms(img)
        return img, torch.tensor(d_idx, dtype=torch.long)


class CycleNetDataset(Dataset):
    def __init__(self, src_dir: str, tgt_dir: str, image_size: int = 224):
        # -------------------------
        # Store domain src/tgt paths with domain indices
        # -------------------------
        self.samples = []

        # -- Source: 0
        for path in Path(src_dir).rglob("*"):
            if path.suffix in {".jpg", ".png"}:
                self.samples.append((path, 0))

        # -- Target: 1
        for path in Path(tgt_dir).rglob("*"):
            if path.suffix in {".jpg", ".png"}:
                self.samples.append((path, 1))

        # -------------------------
        # Define transforms
        # -------------------------
        self.transforms = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        # -- Load image / apply transforms
        filepath, src_idx = self.samples[idx]
        img = Image.open(filepath).convert("RGB")
        img = self.transforms(img)

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
            if path.suffix in {".jpg", ".png"}:
                self.samples.append(path)

        # -------------------------
        # Define transforms
        # -------------------------
        self.transforms = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        filepath = self.samples[idx]
        img = Image.open(filepath).convert("RGB")
        img = self.transforms(img)

        return img