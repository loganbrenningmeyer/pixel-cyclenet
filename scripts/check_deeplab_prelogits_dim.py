#!/usr/bin/env python3
from __future__ import annotations

import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


def extract_prelogits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    features = model.backbone(x)
    feat = features["out"]
    for mod in list(model.classifier.children())[:-1]:
        feat = mod(feat)
    return feat


def main() -> None:
    # Number of semantic classes for the final projection layer.
    num_classes = 8
    # Synthetic batch shape used for the forward pass.
    input_shape = (1, 3, 256, 256)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
        aux_loss=True,
    ).to(device)
    model.eval()

    x = torch.randn(*input_shape, device=device)

    with torch.inference_mode():
        prelogits = extract_prelogits(model=model, x=x)
        pooled = prelogits.mean(dim=(2, 3))

    print(f"input shape: {tuple(x.shape)}")
    print(f"prelogits feature map shape: {tuple(prelogits.shape)}")
    print(f"globally pooled shape: {tuple(pooled.shape)}")
    print(f"embedding dimension: {pooled.shape[1]}")

    expected_dim = 256
    if pooled.shape[1] != expected_dim:
        raise RuntimeError(f"Expected pooled prelogits dimension {expected_dim}, got {pooled.shape[1]}")

    print("Dimension check passed.")


if __name__ == "__main__":
    main()
