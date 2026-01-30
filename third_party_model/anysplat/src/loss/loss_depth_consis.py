from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from third_party_model.anysplat.src.dataset.types import BatchedExample
from third_party_model.anysplat.src.model.decoder.decoder import DecoderOutput
from third_party_model.anysplat.src.model.types import Gaussians
from .loss import Loss
from typing import Generic, Literal, Optional, TypeVar
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
class LossDepthConsisCfg:
    weight: float
    sigma_image: float | None
    use_second_derivative: bool
    loss_type: Literal["MSE", "EdgeAwareLogL1", "PearsonDepth"] = "MSE"
    detach: bool = False
    conf: bool = False
    not_use_valid_mask: bool = False
    apply_after_step: int = 0


@dataclass
class LossDepthConsisCfgWrapper:
    depth_consis: LossDepthConsisCfg


class LogL1(torch.nn.Module):
    """Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.log(1 + torch.abs(pred - gt)).mean()
        else:
            return torch.log(1 + torch.abs(pred - gt))


class EdgeAwareLogL1(torch.nn.Module):
    """Gradient aware Log-L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation
        self.logl1 = LogL1(implementation="per-pixel")

    def forward(self, pred: Tensor, gt: Tensor, rgb: Tensor, mask: Optional[Tensor]):
        logl1 = self.logl1(pred, gt)

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )
        lambda_x = torch.exp(-grad_img_x)
        lambda_y = torch.exp(-grad_img_y)

        loss_x = lambda_x * logl1[..., :, :-1, :]
        loss_y = lambda_y * logl1[..., :-1, :, :]

        if self.implementation == "per-pixel":
            if mask is not None:
                loss_x[~mask[..., :, :-1, :]] = 0
                loss_y[~mask[..., :-1, :, :]] = 0
            return loss_x[..., :-1, :, :] + loss_y[..., :, :-1, :]

        if mask is not None:
            assert mask.shape[:2] == pred.shape[:2]
            loss_x = loss_x[mask[..., :, :-1, :]]
            loss_y = loss_y[mask[..., :-1, :, :]]

        if self.implementation == "scalar":
            return loss_x.mean() + loss_y.mean()


class LossDepthConsis(Loss[LossDepthConsisCfg, LossDepthConsisCfgWrapper]):
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
        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(
                0.0, dtype=torch.float32, device=prediction.depth.device
            )

        # Scale the depth between the near and far planes.
        # conf_valid_mask = depth_dict['conf_valid_mask']
        rendered_depth = prediction.depth
        gt_rgb = (batch["context"]["image"] + 1) / 2
        valid_mask = depth_dict["distill_infos"]["conf_mask"]

        if batch["context"]["valid_mask"].sum() > 0:
            valid_mask = batch["context"]["valid_mask"]
        # if self.cfg.conf:
        #     valid_mask = valid_mask & conf_valid_mask
        if self.cfg.not_use_valid_mask:
            valid_mask = torch.ones_like(valid_mask, device=valid_mask.device)
        pred_depth = depth_dict["depth"].squeeze(-1)
        if self.cfg.detach:
            pred_depth = pred_depth.detach()
        if self.cfg.loss_type == "MSE":
            depth_loss = F.mse_loss(rendered_depth, pred_depth, reduction="none")[
                valid_mask
            ].mean()
        elif self.cfg.loss_type == "EdgeAwareLogL1":
            rendered_depth = rendered_depth.flatten(0, 1).unsqueeze(-1)
            pred_depth = pred_depth.flatten(0, 1).unsqueeze(-1)
            gt_rgb = gt_rgb.flatten(0, 1).permute(0, 2, 3, 1)
            valid_mask = valid_mask.flatten(0, 1).unsqueeze(-1)
            depth_loss = EdgeAwareLogL1()(
                rendered_depth, pred_depth, gt_rgb, valid_mask
            )
        return self.cfg.weight * torch.nan_to_num(depth_loss, nan=0.0)
