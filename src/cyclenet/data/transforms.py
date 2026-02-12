import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_unet_transforms(transform_id: int = 0, image_size: int = 224) -> A.Compose:
    """
    
    """
    std = tuple([0.5] * 3)
    mean = tuple([0.5] * 3)

    # -- Pad by 4% image size all sides
    pad = int(0.04 * image_size)

    # -- [0]: No augmentations
    if transform_id == 0:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    # -- [1]: Mild augmentations (crop / flip)
    elif transform_id == 1:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),

            # -- Reflection padding + random crop / horizontal flip
            A.PadIfNeeded(
                min_height=image_size + 2 * pad,
                min_width=image_size + 2 * pad,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            A.RandomCrop(
                height=image_size,
                width=image_size,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),

            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    # -- [2]: Stronger augmentations (crop / flip, photometric, sensor)
    elif transform_id == 2:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),

            # -- Reflection padding + random crop / horizontal flip
            A.PadIfNeeded(
                min_height=image_size + 2 * pad,
                min_width=image_size + 2 * pad,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            ),
            A.RandomCrop(
                height=image_size,
                width=image_size,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),

            # -- Photometric augs
            A.RandomBrightnessContrast(
                brightness_limit=0.06,
                contrast_limit=0.06,
                p=0.35,
            ),
            A.RandomGamma(
                gamma_limit=(95, 105),
                p=0.35,
            ),

            # -- Sensor artifacts
            A.OneOf([
                A.GaussianBlur(
                    blur_limit=(3, 3),
                    sigma_limit=(0.1, 1.0),
                    p=1.0,
                ),
                A.GaussNoise(
                    std_range=(0.01, 0.03),
                    mean_range=(0.0, 0.0),
                    per_channel=True,
                    noise_scale_factor=1.0,
                    p=1.0,
                ),
            ], p=0.1),

            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    else:
        raise ValueError("transform_id must be 0, 1, or 2.")
    

def load_cyclenet_transforms(transform_id: int = 0, image_size: int = 224) -> A.Compose:
    """
    
    """
    std = tuple([0.5] * 3)
    mean = tuple([0.5] * 3)

    # -- [0] No augmentations
    if transform_id == 0:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    # -- [1] Mild augmentations (flip only)
    elif transform_id == 1:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    # -- [2] Stronger augmentations (crop / flip, mild photometric augs)
    elif transform_id == 2:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),

            # -- Only flip
            A.HorizontalFlip(p=0.5),

            # -- Photometric augs
            A.RandomBrightnessContrast(
                brightness_limit=0.06,
                contrast_limit=0.06,
                p=0.4,
            ),
            A.RandomGamma(
                gamma_limit=(95, 105),
                p=0.4,
            ),

            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    else:
        raise ValueError("transform_id must be 0, 1, or 2.")
    

def load_source_transforms(image_size: int = 224) -> A.Compose:
    """
    
    """
    mean = tuple([0.5] * 3)
    std = tuple([0.5] * 3)

    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
