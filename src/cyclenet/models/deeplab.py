import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEEPLAB_TRANSFORMS = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


class DeepLabV3(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()

        self.num_classes = num_classes

        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=num_classes,
            aux_loss=True,
        )

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return {
            "logits": out["out"],
            "aux_logits": out["aux"],
        }
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Returns integer segmentation map (B, H, W) for the batch of images
        """
        logits = self.model(x)["out"]   # (B, C, H, W)
        preds = logits.argmax(dim=1)    # (B, H, W)
        return preds
    
    def extract_features(self, x: torch.Tensor, layer: str = "classifier") -> torch.Tensor:
        features = self.model.backbone(x)

        if layer == "backbone":
            feat = features["out"]                  # [B, 2048, h, w]
        elif layer == "aux_backbone":
            feat = features["aux"]                  # [B, 1024, h, w]
        elif layer == "classifier":
            feat = self.model.classifier[0](features["out"])  # ASPP output, [B, 256, h, w]
        elif layer == "prelogits":
            # -- Run the existing DeepLab head up to, but not including, the
            # -- final 1x1 class projection conv so checkpoint keys stay unchanged.
            feat = features["out"]
            for mod in list(self.model.classifier.children())[:-1]:
                feat = mod(feat)
        else:
            raise ValueError(f"Unknown layer: {layer}")

        return feat
