from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from third_party_model.anysplat.src.dataset.types import BatchedExample
from third_party_model.anysplat.src.model.decoder.decoder import DecoderOutput
from third_party_model.anysplat.src.model.types import Gaussians
from .loss import Loss
from typing import Generic, Literal, TypeVar
from dataclasses import fields
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from third_party_model.anysplat.src.loss.depth_anything.dpt import DepthAnything
from third_party_model.anysplat.src.misc.utils import vis_depth_map

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


@dataclass
class LossDepthGTCfg:
    weight: float
    type: Literal["l1", "mse", "silog", "gradient", "l1+gradient"] | None


@dataclass
class LossDepthGTCfgWrapper:
    depthgt: LossDepthGTCfg


class LossDepthGT(Loss[LossDepthGTCfg, LossDepthGTCfgWrapper]):
    def gradient_loss(self, gs_depth, target_depth, target_valid_mask):
        diff = gs_depth - target_depth

        grad_x_diff = diff[:, :, :, 1:] - diff[:, :, :, :-1]
        grad_y_diff = diff[:, :, 1:, :] - diff[:, :, :-1, :]

        mask_x = target_valid_mask[:, :, :, 1:] * target_valid_mask[:, :, :, :-1]
        mask_y = target_valid_mask[:, :, 1:, :] * target_valid_mask[:, :, :-1, :]

        grad_x_diff = grad_x_diff * mask_x
        grad_y_diff = grad_y_diff * mask_y

        grad_x_diff = grad_x_diff.clamp(min=-100, max=100)
        grad_y_diff = grad_y_diff.clamp(min=-100, max=100)

        loss_x = grad_x_diff.abs().sum()
        loss_y = grad_y_diff.abs().sum()
        num_valid = mask_x.sum() + mask_y.sum()

        if num_valid == 0:
            gradient_loss = 0
        else:
            gradient_loss = (loss_x + loss_y) / (num_valid + 1e-6)

        return gradient_loss

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Scale the depth between the near and far planes.

        # prediction: B, H, W, C
        # target: B, H, W, C
        # mask: B, H, W

        target_depth = batch["target"]["depth"]
        target_valid_mask = batch["target"]["valid_mask"]
        gs_depth = prediction.depth.clamp(1e-3)

        if self.cfg.type == "l1":
            depth_loss = torch.abs(
                target_depth[target_valid_mask] - gs_depth[target_valid_mask]
            ).mean()
        elif self.cfg.type == "mse":
            depth_loss = F.mse_loss(
                target_depth[target_valid_mask], gs_depth[target_valid_mask]
            )
        elif self.cfg.type == "silog":
            depth_loss = (
                torch.log(gs_depth[target_valid_mask]) ** 2
                + (gs_depth[target_valid_mask] - target_depth[target_valid_mask]) ** 2
                - 0.5
            )
            depth_loss = depth_loss.mean()
        elif self.cfg.type == "gradient":
            depth_loss = self.gradient_loss(gs_depth, target_depth, target_valid_mask)
        elif self.cfg.type == "l1+gradient":
            depth_loss_l1 = torch.abs(
                target_depth[target_valid_mask] - gs_depth[target_valid_mask]
            ).mean()
            depth_loss_gradient = self.gradient_loss(
                gs_depth, target_depth, target_valid_mask
            )
            depth_loss = depth_loss_l1 + depth_loss_gradient

        return self.cfg.weight * torch.nan_to_num(
            depth_loss, nan=0.0, posinf=0.0, neginf=0.0
        )
