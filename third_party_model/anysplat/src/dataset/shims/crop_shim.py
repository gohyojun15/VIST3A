import random
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import torchvision.transforms.functional as F
import cv2

from ..types import AnyExample, AnyViews


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")

def rescale_depth(
    depth: Float[Tensor, "1 h w"],
    shape: tuple[int, int],
) -> Float[Tensor, "1 h_out w_out"]:
    h, w = shape
    depth_new = depth.detach().cpu().numpy()
    depth_new = cv2.resize(depth_new, (w,h), interpolation=cv2.INTER_NEAREST)
    depth_new = torch.from_numpy(depth_new).to(depth.device)
    return depth_new
    
def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depths: Float[Tensor, "*#batch 1 h w"] | None = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    Float[Tensor, "*#batch 1 h_out w_out"] | None,  # updated depths
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    if depths is not None:
        depths = depths[..., row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    
    if depths is not None:
        return images, intrinsics, depths
    else:
        return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    intr_aug: bool = False,
    scale_range: tuple[float, float] = (0.77, 1.0),
    depths: Float[Tensor, "*#batch 1 h w"] | None = None,
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    Float[Tensor, "*#batch 1 h_out w_out"] | None,  # updated depths
]:
    if type(images) == list:
        images_new = []
        intrinsics_new = []
        for i in range(len(images)):
            image = images[i]
            intrinsic = intrinsics[i]
            
            *_, h_in, w_in = image.shape
            h_out, w_out = shape

            scale_factor = max(h_out / h_in, w_out / w_in)
            h_scaled = round(h_in * scale_factor)
            w_scaled = round(w_in * scale_factor)
            image = F.resize(image, (h_scaled, w_scaled))
            image = F.center_crop(image, (h_out, w_out))
            images_new.append(image)
            
            intrinsic_new = intrinsic.clone()
            intrinsic_new[..., 0, 0] *= w_scaled / w_in  # fx
            intrinsic_new[..., 1, 1] *= h_scaled / h_in  # fy
            intrinsics_new.append(intrinsic_new)
        
        if depths is not None:
            depths_new = []
            for i in range(len(depths)):
                depth = depths[i]
                depth = rescale_depth(depth, (h_out, w_out))
                depth = F.center_crop(depth, (h_out, w_out))
                depths_new.append(depth)
            return torch.stack(images_new), torch.stack(intrinsics_new), torch.stack(depths_new)
        else:
            return torch.stack(images_new), torch.stack(intrinsics_new)
    
    else:
        # we only support intr_aug for clean datasets
        *_, h_in, w_in = images.shape
        h_out, w_out = shape
        # assert h_out <= h_in and w_out <= w_in # to avoid the case that the image is too small, like co3d
        
        if intr_aug:
            scale = random.uniform(*scale_range)
            h_scale = round(h_out * scale)
            w_scale = round(w_out * scale)
        else:
            h_scale = h_out
            w_scale = w_out

        scale_factor = max(h_scale / h_in, w_scale / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)
        assert h_scaled == h_scale or w_scaled == w_scale

        # Reshape the images to the correct size. Assume we don't have to worry about
        # changing the intrinsics based on how the images are rounded.
        *batch, c, h, w = images.shape
        images = images.reshape(-1, c, h, w)
        images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
        images = images.reshape(*batch, c, h_scaled, w_scaled)

        if depths is not None:
            if type(depths) == list:
                depths_new = []
                for i in range(len(depths)):
                    depth = depths[i]
                    depth = rescale_depth(depth, (h_scaled, w_scaled)) 
                    depths_new.append(depth)
                depths = torch.stack(depths_new)
            else:
                depths = depths.reshape(-1, h, w)
                depths = torch.stack([rescale_depth(depth, (h_scaled, w_scaled)) for depth in depths])
                depths = depths.reshape(*batch, h_scaled, w_scaled)
            
            images, intrinsics, depths = center_crop(images, intrinsics, (h_scale, w_scale), depths)

            if intr_aug:
                images = F.resize(images, size=(h_out, w_out), interpolation=F.InterpolationMode.BILINEAR)
                depths = F.resize(depths, size=(h_out, w_out), interpolation=F.InterpolationMode.NEAREST)
                
            return images, intrinsics, depths
        else:
            images, intrinsics = center_crop(images, intrinsics, (h_scale, w_scale))

            if intr_aug:
                images = F.resize(images, size=(h_out, w_out))

            return images, intrinsics


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int], intr_aug: bool = False) -> AnyViews:
    if "depth" in views.keys():
        images, intrinsics, depths = rescale_and_crop(views["image"], views["intrinsics"], shape, depths=views["depth"], intr_aug=intr_aug)
        return {
            **views,
            "image": images,
            "intrinsics": intrinsics,
            "depth": depths,
        }
    else:
        images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape, intr_aug)
        return {
            **views,
            "image": images,
            "intrinsics": intrinsics,
        }
        

def apply_crop_shim(example: AnyExample, shape: tuple[int, int], intr_aug: bool = False) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape, intr_aug),
        "target": apply_crop_shim_to_views(example["target"], shape, intr_aug),
    }
