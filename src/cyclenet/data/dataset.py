from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class UNetDataset(Dataset):
    def __init__(self, x_dir: str, y_dir: str, x_label: str, y_label: str):
        # -------------------------
        # Store domain x/y paths with labels
        # -------------------------
        self.samples = []

        for path in Path(x_dir).rglob("*"):
            if path.suffix in {".jpg", ".png"}:
                self.samples.append((path, x_label))

        for path in Path(y_dir).rglob("*"):
            if path.suffix in {".jpg", ".png"}:
                self.samples.append((path, y_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, label = self.samples[idx]
        img = Image.open(filepath).convert("RGB")
        return img, label


class CycleNetDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass