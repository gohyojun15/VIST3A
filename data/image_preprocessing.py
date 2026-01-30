import torch
import torch.nn.functional as F
import torchvision
from PIL import Image


def load_image(image_path: str, transform: torchvision.transforms.Compose):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = transform(img)
        return img


def resize_shorter_crop_square_batch(
    images: torch.Tensor,
    target_size: int = 448,
):
    """
    Step‑by‑step:
    1)  Isotropically scale the tensor so that *min(H, W) = target_size*.
    2)  Centre‑crop the *other* dimension to `target_size`, producing
        a square `(T,C,target_size,target_size)`.
    """
    T, C, H0, W0 = images.shape
    # ----------------------------------------------------------------------
    # 1. Isotropic resize – make *shorter* side == target_size -------------
    # ----------------------------------------------------------------------
    scale = target_size / min(H0, W0)
    new_h = round(H0 * scale)
    new_w = round(W0 * scale)
    images = F.interpolate(
        images, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    # ----------------------------------------------------------------------
    # 2. Centre-crop the *other* dimension to target_size -------------------
    # ----------------------------------------------------------------------
    # vertical crop (landscape or square cases where new_h >= target_size)
    if new_h > target_size:
        y0 = (new_h - target_size) // 2
        images = images[:, :, y0 : y0 + target_size, :]
        new_h = target_size

    # horizontal crop (portrait or square cases where new_w >= target_size)
    if new_w > target_size:
        x0 = (new_w - target_size) // 2
        images = images[:, :, :, x0 : x0 + target_size]
        new_w = target_size

    # At this point (new_h, new_w) == (target_size, target_size)
    assert new_h == target_size and new_w == target_size, "logic error"
    return images
