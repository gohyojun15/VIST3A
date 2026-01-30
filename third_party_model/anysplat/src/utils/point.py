import torch
from torch import Tensor

def get_normal_map(depth_map: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W).
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Camera coordinates (H, W, 3)
    """
    B, H, W = depth_map.shape
    assert intrinsic.shape == (B, 3, 3), "Intrinsic matrix must be Bx3x3"
    assert (intrinsic[:, 0, 1] == 0).all() and (intrinsic[:, 1, 0] == 0).all(), "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu = intrinsic[:, 0, 0] * W  # (B,)
    fv = intrinsic[:, 1, 1] * H  # (B,)
    cu = intrinsic[:, 0, 2] * W  # (B,)
    cv = intrinsic[:, 1, 2] * H  # (B,)

    # Generate grid of pixel coordinates
    u = torch.arange(W, device=depth_map.device)[None, None, :].expand(B, H, W)
    v = torch.arange(H, device=depth_map.device)[None, :, None].expand(B, H, W)

    # Unproject to camera coordinates (B, H, W)
    x_cam = (u - cu[:, None, None]) * depth_map / fu[:, None, None]
    y_cam = (v - cv[:, None, None]) * depth_map / fv[:, None, None]
    z_cam = depth_map
    
    # Stack to form camera coordinates (B, H, W, 3)
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1).to(dtype=torch.float32)

    output = torch.zeros_like(cam_coords)
    # Calculate dx using batch dimension (B, H-2, W-2, 3)
    dx = cam_coords[:, 2:, 1:-1] - cam_coords[:, :-2, 1:-1]
    # Calculate dy using batch dimension (B, H-2, W-2, 3)
    dy = cam_coords[:, 1:-1, 2:] - cam_coords[:, 1:-1, :-2]
    # Cross product and normalization (B, H-2, W-2, 3)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    # Assign the computed normal map to the output tensor
    output[:, 1:-1, 1:-1, :] = normal_map

    return output