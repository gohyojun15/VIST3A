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

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


@dataclass
class LossDepthCfg:
    weight: float
    sigma_image: float | None
    use_second_derivative: bool


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__(cfg)

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

        model_configs = {
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        }
        encoder = "vits"  # or 'vitb', 'vits'
        depth_anything = DepthAnything(model_configs[encoder])
        depth_anything.load_state_dict(
            torch.load(f"src/loss/depth_anything/depth_anything_{encoder}14.pth")
        )

        self.depth_anything = depth_anything
        for param in self.depth_anything.parameters():
            param.requires_grad = False

    def disp_rescale(self, disp: Float[Tensor, "B H W"]):
        disp = disp.flatten(1, 2)
        disp_median = torch.median(disp, dim=-1, keepdim=True)[0]  # (B, V, 1)
        disp_var = (disp - disp_median).abs().mean(dim=-1, keepdim=True)  # (B, V, 1)
        disp = (disp - disp_median) / (disp_var + 1e-6)
        return disp

    def smooth_l1_loss(self, pred, target, beta=1.0, reduction="none"):
        diff = pred - target
        abs_diff = torch.abs(diff)

        loss = torch.where(abs_diff < beta, 0.5 * diff**2 / beta, abs_diff - 0.5 * beta)

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(
                "Invalid reduction type. Choose from 'mean', 'sum', or 'none'."
            )

    def ctx_depth_loss(
        self,
        depth_map: Float[Tensor, "B V H W C"],
        depth_conf: Float[Tensor, "B V H W"],
        batch: BatchedExample,
        cxt_depth_weight: float = 0.01,
        alpha: float = 0.2,
    ):
        B, V, _, H, W = batch["context"]["image"].shape
        ctx_imgs = batch["context"]["image"].view(B * V, 3, H, W).float()
        da_output = self.depth_anything(ctx_imgs)
        da_output = self.disp_rescale(da_output)

        disp_context = 1.0 / depth_map.flatten(0, 1).squeeze(-1).clamp(
            1e-3
        )  # (B * V, H, W)
        context_output = self.disp_rescale(disp_context)

        depth_conf = depth_conf.flatten(0, 1).flatten(1, 2)  # (B * V)

        return (
            cxt_depth_weight
            * (
                self.smooth_l1_loss(
                    context_output * depth_conf,
                    da_output * depth_conf,
                    reduction="none",
                )
                - alpha * torch.log(depth_conf)
            ).mean()
        )

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Scale the depth between the near and far planes.
        target_imgs = batch["target"]["image"]
        B, V, _, H, W = target_imgs.shape
        target_imgs = target_imgs.view(B * V, 3, H, W)
        da_output = self.depth_anything(target_imgs.float())
        da_output = self.disp_rescale(da_output)

        disp_gs = 1.0 / prediction.depth.flatten(0, 1).clamp(1e-3).float()
        gs_output = self.disp_rescale(disp_gs)

        return self.cfg.weight * torch.nan_to_num(
            F.smooth_l1_loss(da_output, gs_output), nan=0.0
        )
