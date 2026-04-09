import numpy as np
import torch
import umap
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()


def get_clip_embeddings(img_paths: list[str]) -> np.ndarray:
    # -------------------------
    # Pass image batch into CLIP
    # -------------------------
    images = [Image.open(path).convert("RGB") for path in img_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        feats = model.get_image_features(**inputs)      # (B,512)

    # -------------------------
    # Normalize features
    # -------------------------
    feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu().numpy()


def umap_embed(
    feats: np.ndarray, n_components: int = 2, random_state: int = 42
) -> np.ndarray:
    """
    Given numpy array of features (N, d) gets UMAP embedding and
    returns shape (N, n_components)
    -- ( Recommended ): 2-5k images per domain
    """
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    emb_2d = reducer.fit_transform(feats)

    return emb_2d  # (N, n_components)


def clip_umap(
    img_paths: list[str], n_components: int = 2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gets CLIP embeddings of images then embeds using UMAP
    """
    feats = get_clip_embeddings(img_paths)
    emb_2d = umap_embed(feats, n_components, random_state)

    return emb_2d, feats
