# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch

def batchify_unproject_depth_map_to_point_map(
    depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (torch.Tensor): Batch of depth maps of shape (B, V, H, W, 1) or (B, V, H, W)
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices of shape (B, V, 3, 4)
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices of shape (B, V, 3, 3)
        
    Returns:
        torch.Tensor: Batch of 3D world coordinates of shape (S, H, W, 3)
    """

    # Handle both (S, H, W, 1) and (S, H, W) cases
    if depth_map.dim() == 5:
        depth_map = depth_map.squeeze(-1)  # (S, H, W)
        
    # Generate batched camera coordinates
    H, W = depth_map.shape[2:]
    batch_size, num_views = depth_map.shape[0], depth_map.shape[1]
    
    # Intrinsic parameters (S, 3, 3)
    intrinsics_cam, extrinsics_cam, depth_map = intrinsics_cam.flatten(0, 1), extrinsics_cam.flatten(0, 1), depth_map.flatten(0, 1)
    fu = intrinsics_cam[:, 0, 0]  # (S,)
    fv = intrinsics_cam[:, 1, 1]  # (S,)
    cu = intrinsics_cam[:, 0, 2]  # (S,)
    cv = intrinsics_cam[:, 1, 2]  # (S,)
    
    # Generate grid of pixel coordinates
    u = torch.arange(W, device=depth_map.device)[None, None, :].expand(batch_size * num_views, H, W)  # (S, H, W)
    v = torch.arange(H, device=depth_map.device)[None, :, None].expand(batch_size * num_views, H, W)  # (S, H, W)
    
    # Unproject to camera coordinates (S, H, W, 3)
    x_cam = (u - cu[:, None, None]) * depth_map / fu[:, None, None]
    y_cam = (v - cv[:, None, None]) * depth_map / fv[:, None, None]
    z_cam = depth_map
    
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1)  # (S, H, W, 3)
    
    # Transform to world coordinates
    cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # (S, 4, 4)

    # homo transformation
    homo_pts = torch.cat((cam_coords, torch.ones_like(cam_coords[..., :1])), dim=-1).flatten(1, 2)
    world_coords = torch.bmm(cam_to_world, homo_pts.transpose(1, 2)).transpose(1, 2)[:, :, :3].view(batch_size*num_views, H, W, 3)
    
    return world_coords.view(batch_size, num_views, H, W, 3)

def unproject_depth_map_to_point_map(
    depth_map: torch.Tensor, extrinsics_cam: torch.Tensor, intrinsics_cam: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (torch.Tensor): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices of shape (S, 3, 3)
        
    Returns:
        torch.Tensor: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = torch.stack(world_points_list, dim=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    eps=1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).
        extrinsic (torch.Tensor): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = torch.matmul(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: torch.Tensor, intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(W, device=depth_map.device), 
                         torch.arange(H, device=depth_map.device), 
                         indexing='xy')

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1).to(dtype=torch.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    R_transposed = R.transpose(1, 2)  # (N,3,3)
    top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
    inverted_matrix = torch.eye(4, 4, device=R.device)[None].repeat(len(R), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix
