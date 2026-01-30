import io
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import skvideo
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


def save_video(tensor, save_path, fps=10):
    """
    Save a tensor of shape (N, C, H, W) as a video file using imageio.
    Args:
        tensor: Tensor of shape (N, C, H, W) in range [0, 1]
        save_path: Path to save the video file
        fps: Frames per second for the video
    """
    # Convert tensor to numpy array and adjust dimensions
    video = tensor.cpu().detach().numpy()  # (N, C, H, W)
    video = np.transpose(video, (0, 2, 3, 1))  # (N, H, W, C)

    # Scale to [0, 255] and convert to uint8
    video = (video * 255).astype(np.uint8)

    # Ensure the directory exists
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Use imageio to write video (handles codec compatibility automatically)
    import imageio

    writer = imageio.get_writer(save_path, fps=fps)

    for frame in video:
        writer.append_data(frame)

    writer.close()


def save_interpolated_video(
    pred_extrinsics,
    pred_intrinsics,
    b,
    h,
    w,
    gaussians,
    save_path,
    decoder_func,
    t=10,
    name=None,
    cov_ignore=False,
):
    # Interpolate between neighboring frames
    # t: Number of extra views to interpolate between each pair
    interpolated_extrinsics = []
    interpolated_intrinsics = []

    # For each pair of neighboring frame
    for i in range(pred_extrinsics.shape[1] - 1):
        # Add the current frame
        interpolated_extrinsics.append(pred_extrinsics[:, i : i + 1])
        interpolated_intrinsics.append(pred_intrinsics[:, i : i + 1])

        # Interpolate between current and next frame
        for j in range(1, t + 1):
            alpha = j / (t + 1)

            # Interpolate extrinsics
            start_extrinsic = pred_extrinsics[:, i]
            end_extrinsic = pred_extrinsics[:, i + 1]

            # Separate rotation and translation
            start_rot = start_extrinsic[:, :3, :3]
            end_rot = end_extrinsic[:, :3, :3]
            start_trans = start_extrinsic[:, :3, 3]
            end_trans = end_extrinsic[:, :3, 3]

            # Interpolate translation (linear)
            interp_trans = (1 - alpha) * start_trans + alpha * end_trans

            # Interpolate rotation (spherical)
            start_rot_flat = start_rot.reshape(b, 9)
            end_rot_flat = end_rot.reshape(b, 9)
            interp_rot_flat = (1 - alpha) * start_rot_flat + alpha * end_rot_flat
            interp_rot = interp_rot_flat.reshape(b, 3, 3)

            # Normalize rotation matrix to ensure it's orthogonal
            u, _, v = torch.svd(interp_rot)
            interp_rot = torch.bmm(u, v.transpose(1, 2))

            # Combine interpolated rotation and translation
            interp_extrinsic = (
                torch.eye(4, device=pred_extrinsics.device).unsqueeze(0).repeat(b, 1, 1)
            )
            interp_extrinsic[:, :3, :3] = interp_rot
            interp_extrinsic[:, :3, 3] = interp_trans

            # Interpolate intrinsics (linear)
            start_intrinsic = pred_intrinsics[:, i]
            end_intrinsic = pred_intrinsics[:, i + 1]
            interp_intrinsic = (1 - alpha) * start_intrinsic + alpha * end_intrinsic

            # Add interpolated frame
            interpolated_extrinsics.append(interp_extrinsic.unsqueeze(1))
            interpolated_intrinsics.append(interp_intrinsic.unsqueeze(1))

    # Concatenate all frames
    pred_all_extrinsic = torch.cat(interpolated_extrinsics, dim=1)
    pred_all_intrinsic = torch.cat(interpolated_intrinsics, dim=1)

    # Add the last frame
    interpolated_extrinsics.append(pred_all_extrinsic[:, -1:])
    interpolated_intrinsics.append(pred_all_intrinsic[:, -1:])

    # Update K to reflect the new number of frames
    num_frames = pred_all_extrinsic.shape[1]

    # Render interpolated views
    interpolated_output = decoder_func.forward(
        gaussians,
        pred_all_extrinsic,
        pred_all_intrinsic.float(),
        torch.ones(1, num_frames, device=pred_all_extrinsic.device) * 0.1,
        torch.ones(1, num_frames, device=pred_all_extrinsic.device) * 100,
        (h, w),
        cov_ignore=cov_ignore,
    )

    # Convert to video format
    video = interpolated_output.color[0].clip(min=0, max=1)
    depth = interpolated_output.depth[0]

    # Normalize depth for visualization
    # to avoid `quantile() input tensor is too large`
    num_views = pred_extrinsics.shape[1]
    depth_norm = (depth - depth[::num_views].quantile(0.01)) / (
        depth[::num_views].quantile(0.99) - depth[::num_views].quantile(0.01)
    )
    depth_norm = plt.cm.turbo(depth_norm.cpu().numpy())
    depth_colored = (
        torch.from_numpy(depth_norm[..., :3]).permute(0, 3, 1, 2).to(depth.device)
    )
    depth_colored = depth_colored.clip(min=0, max=1)

    # Save depth video
    if name is None:
        save_video(depth_colored, os.path.join(save_path, f"depth.mp4"), fps=20)
        # Save video

        save_video(video, os.path.join(save_path, f"gs.mp4"), fps=20)
    elif name == "optim":
        save_video(
            depth_colored, os.path.join(save_path, f"gs_optimized_depth.mp4"), fps=20
        )
        save_video(video, os.path.join(save_path, f"gs_optimized.mp4"), fps=20)

    return os.path.join(save_path, f"gs.mp4"), os.path.join(save_path, f"depth.mp4")
