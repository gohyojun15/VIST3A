import copy
import random
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from ..types import AnyExample, AnyViews


def reflect_extrinsics(
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


def reflect_views(views: AnyViews) -> AnyViews:
    if "depth" in views.keys():
        return {
            **views,
            "image": views["image"].flip(-1),
            "extrinsics": reflect_extrinsics(views["extrinsics"]),
            "depth": views["depth"].flip(-1),
        }
    else:
        return {
            **views,
            "image": views["image"].flip(-1),
            "extrinsics": reflect_extrinsics(views["extrinsics"]),
        }


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example
    
    return {
        **example,
        "context": reflect_views(example["context"]),
        "target": reflect_views(example["target"]),
    }

def rotate_90_degrees(
    image: torch.Tensor, depth_map: torch.Tensor | None, extri_opencv: torch.Tensor, intri_opencv: torch.Tensor, clockwise=True
):
    """
    Rotates the input image, depth map, and camera parameters by 90 degrees.

    Applies one of two 90-degree rotations:
    - Clockwise 
    - Counterclockwise (if clockwise=False)

    The extrinsic and intrinsic matrices are adjusted accordingly to maintain
    correct camera geometry.

    Args:
        image (torch.Tensor):
            Input image tensor of shape (C, H, W).
        depth_map (torch.Tensor or None):
            Depth map tensor of shape (H, W), or None if not available.
        extri_opencv (torch.Tensor):
            Extrinsic matrix (3x4) in OpenCV convention.
        intri_opencv (torch.Tensor):
            Intrinsic matrix (3x3).
        clockwise (bool):
            If True, rotates the image 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        tuple:
            (
                rotated_image,
                rotated_depth_map,
                new_extri_opencv,
                new_intri_opencv
            )

            Where each is the updated version after the rotation.
    """
    image_height, image_width = image.shape[-2:]
    
    # Rotate the image and depth map
    rotated_image, rotated_depth_map = rotate_image_and_depth_rot90(image, depth_map, clockwise)
    # Adjust the intrinsic matrix
    new_intri_opencv = adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise)
    # Adjust the extrinsic matrix
    new_extri_opencv = adjust_extrinsic_matrix_rot90(extri_opencv, clockwise)

    return (
        rotated_image,
        rotated_depth_map,
        new_extri_opencv,
        new_intri_opencv,
    )


def rotate_image_and_depth_rot90(image: torch.Tensor, depth_map: torch.Tensor | None, clockwise: bool):
    """
    Rotates the given image and depth map by 90 degrees (clockwise or counterclockwise).

    Args:
        image (torch.Tensor):
            Input image tensor of shape (C, H, W).
        depth_map (torch.Tensor or None):
            Depth map tensor of shape (H, W), or None if not available.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        tuple:
            (rotated_image, rotated_depth_map)
    """
    rotated_depth_map = None
    if clockwise:
        rotated_image = torch.rot90(image, k=-1, dims=[-2, -1])
        if depth_map is not None:
            rotated_depth_map = torch.rot90(depth_map, k=-1, dims=[-2, -1])
    else:
        rotated_image = torch.rot90(image, k=1, dims=[-2, -1])
        if depth_map is not None:
            rotated_depth_map = torch.rot90(depth_map, k=1, dims=[-2, -1])
    return rotated_image, rotated_depth_map


def adjust_extrinsic_matrix_rot90(extri_opencv: torch.Tensor, clockwise: bool):
    """
    Adjusts the extrinsic matrix (3x4) for a 90-degree rotation of the image.

    The rotation is in the image plane. This modifies the camera orientation
    accordingly. The function applies either a clockwise or counterclockwise
    90-degree rotation.

    Args:
        extri_opencv (torch.Tensor):
            Extrinsic matrix (3x4) in OpenCV convention.
        clockwise (bool):
            If True, rotate extrinsic for a 90-degree clockwise image rotation;
            otherwise, counterclockwise.

    Returns:
        torch.Tensor:
            A new 3x4 extrinsic matrix after the rotation.
    """
    R = extri_opencv[:3, :3]
    t = extri_opencv[:3, 3]

    if clockwise:
        R_rotation = torch.tensor([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=extri_opencv.dtype, device=extri_opencv.device)
    else:
        R_rotation = torch.tensor([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ], dtype=extri_opencv.dtype, device=extri_opencv.device)

    new_R = torch.matmul(R_rotation, R)
    new_t = torch.matmul(R_rotation, t)
    new_extri_opencv = torch.cat((new_R, new_t.reshape(-1, 1)), dim=1)
    new_extri_opencv = torch.cat((new_extri_opencv, 
                                  torch.tensor([[0, 0, 0, 1]], 
                                dtype=extri_opencv.dtype, device=extri_opencv.device)), dim=0)
    return new_extri_opencv


def adjust_intrinsic_matrix_rot90(intri_opencv: torch.Tensor, image_width: int, image_height: int, clockwise: bool):
    """
    Adjusts the intrinsic matrix (3x3) for a 90-degree rotation of the image in the image plane.

    Args:
        intri_opencv (torch.Tensor):
            Intrinsic matrix (3x3).
        image_width (int):
            Original width of the image.
        image_height (int):
            Original height of the image.
        clockwise (bool):
            If True, rotate 90 degrees clockwise; else 90 degrees counterclockwise.

    Returns:
        torch.Tensor:
            A new 3x3 intrinsic matrix after the rotation.
    """
    intri_opencv = copy.deepcopy(intri_opencv)
    intri_opencv[0, :] *= image_width
    intri_opencv[1, :] *= image_height

    fx, fy, cx, cy = (
        intri_opencv[0, 0],
        intri_opencv[1, 1],
        intri_opencv[0, 2],
        intri_opencv[1, 2],
    )

    new_intri_opencv = torch.eye(3, dtype=intri_opencv.dtype, device=intri_opencv.device)
    if clockwise:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx
    
    new_intri_opencv[0, :] /= image_height
    new_intri_opencv[1, :] /= image_width

    return new_intri_opencv
