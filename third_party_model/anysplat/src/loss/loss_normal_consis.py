from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from third_party_model.anysplat.src.dataset.types import BatchedExample
from third_party_model.anysplat.src.model.decoder.decoder import DecoderOutput
from third_party_model.anysplat.src.model.types import Gaussians
from .loss import Loss
from typing import Generic, TypeVar
from dataclasses import fields
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from third_party_model.anysplat.src.loss.depth_anything.dpt import DepthAnything
from third_party_model.anysplat.src.misc.utils import vis_depth_map
import open3d as o3d

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


@dataclass
class LossNormalConsisCfg:
    normal_weight: float
    smooth_weight: float
    sigma_image: float | None
    use_second_derivative: bool
    detach: bool = False
    conf: bool = False
    not_use_valid_mask: bool = False


@dataclass
class LossNormalConsisCfgWrapper:
    normal_consis: LossNormalConsisCfg


class TVLoss(torch.nn.Module):
    """TV loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        Args:
            pred: [batch, H, W, 3]

        Returns:
            tv_loss: [batch]
        """
        h_diff = pred[..., :, :-1, :] - pred[..., :, 1:, :]
        w_diff = pred[..., :-1, :, :] - pred[..., 1:, :, :]
        return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))


class LossNormalConsis(Loss[LossNormalConsisCfg, LossNormalConsisCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Scale the depth between the near and far planes.
        conf_valid_mask = depth_dict["conf_valid_mask"].flatten(0, 1)
        valid_mask = batch["context"]["valid_mask"][:, batch["using_index"]].flatten(
            0, 1
        )
        if self.cfg.conf:
            valid_mask = valid_mask & conf_valid_mask
        if self.cfg.not_use_valid_mask:
            valid_mask = torch.ones_like(valid_mask, device=valid_mask.device)
        render_normal = self.get_normal_map(
            prediction.depth.flatten(0, 1), batch["context"]["intrinsics"].flatten(0, 1)
        )
        pred_normal = self.get_normal_map(
            depth_dict["depth"].flatten(0, 1).squeeze(-1),
            batch["context"]["intrinsics"].flatten(0, 1),
        )
        if self.cfg.detach:
            pred_normal = pred_normal.detach()
        alpha1_loss = (1 - (render_normal * pred_normal).sum(-1)).mean()
        alpha2_loss = F.l1_loss(render_normal, pred_normal, reduction="mean")
        normal_smooth_loss = TVLoss()(render_normal)
        normal_loss = (alpha1_loss + alpha2_loss) / 2
        return self.cfg.normal_weight * torch.nan_to_num(
            normal_loss, nan=0.0
        ) + self.cfg.smooth_weight * torch.nan_to_num(normal_smooth_loss, nan=0.0)

    def get_normal_map(
        self, depth_map: torch.Tensor, intrinsic: torch.Tensor
    ) -> torch.Tensor:
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
        assert (intrinsic[:, 0, 1] == 0).all() and (intrinsic[:, 1, 0] == 0).all(), (
            "Intrinsic matrix must have zero skew"
        )

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
