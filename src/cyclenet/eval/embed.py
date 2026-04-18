from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.models.feature_extraction import create_feature_extractor

from cyclenet.models import DeepLabV3, DEEPLAB_TRANSFORMS


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _save_embeddings(feats: np.ndarray, save_path: str | Path | None) -> None:
    if save_path is None:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, feats)


class CLIPEmbedder:
    def __init__(
        self,
        device: str | torch.device | None = None,
        clip_path: str = "openai/clip-vit-base-patch32",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(clip_path).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(clip_path)

    def embed(
        self,
        img_paths: list[str],
        batch_size: int = 64,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        feats_all = []

        with torch.inference_mode():
            for batch_paths in _batched(img_paths, batch_size):
                images = []
                for path in batch_paths:
                    with Image.open(path) as img:
                        images.append(img.convert("RGB"))

                inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
                feats = self.model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                feats_all.append(feats.cpu().numpy())

        feats_np = np.concatenate(feats_all, axis=0)
        _save_embeddings(feats_np, save_path)
        return feats_np


class InceptionEmbedder:
    def __init__(
        self,
        device: str | torch.device | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # -- Load base InceptionV3 model
        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.base_model = inception_v3(weights=weights).to(self.device)
        self.base_model.eval()
        # -- Create avgpool feature extractor
        self.model = create_feature_extractor(
            self.base_model,
            return_nodes={"avgpool": "feat"},
        ).to(self.device)
        # -- IMAGENET1K_V1 weights transforms
        self.transforms = weights.transforms()

    def embed(
        self,
        img_paths: list[str],
        batch_size: int = 64,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        feats_all = []

        with torch.inference_mode():
            for batch_paths in _batched(img_paths, batch_size):
                images = []
                for path in batch_paths:
                    with Image.open(path) as img:
                        images.append(self.transforms(img.convert("RGB")))

                x = torch.stack(images, dim=0).to(self.device)
                feats = self.model(x)["feat"]  # [B, 2048, 1, 1]
                feats = feats.flatten(1)       # [B, 2048]
                feats_all.append(feats.cpu().numpy())

        feats_np = np.concatenate(feats_all, axis=0)
        _save_embeddings(feats_np, save_path)
        return feats_np


class DeepLabEmbedder:
    def __init__(
        self, 
        ckpt_path: str | Path, 
        num_classes: int = 8, 
        feature_layer: str = "prelogits",
        device: str | torch.device | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepLabV3(num_classes).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.feature_layer = feature_layer
        self.transforms = DEEPLAB_TRANSFORMS

    def embed(
        self,
        img_paths: list[str],
        batch_size: int = 64,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        feats_all = []

        with torch.inference_mode():
            for batch_paths in _batched(img_paths, batch_size):
                images = []
                for path in batch_paths:
                    with Image.open(path) as img:
                        img_np = np.array(img.convert("RGB"))
                        images.append(self.transforms(image=img_np)["image"])

                x = torch.stack(images, dim=0).to(self.device)
                feats = self.model.extract_features(x, layer=self.feature_layer)    # [B, C, H, W]
                feats = feats.mean(dim=(2, 3))      # global average pooling: [B, C]
                feats_all.append(feats.cpu().numpy())

        feats_np = np.concatenate(feats_all, axis=0)
        _save_embeddings(feats_np, save_path)
        return feats_np