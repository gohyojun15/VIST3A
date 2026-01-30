from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor
import torch
import torch.nn.functional as F
from third_party_model.anysplat.src.dataset.types import BatchedExample
from third_party_model.anysplat.src.model.decoder.decoder import DecoderOutput
from third_party_model.anysplat.src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossOpacityCfg:
    weight: float
    type: Literal["exp", "mean", "exp+mean"] = "exp+mean"


@dataclass
class LossOpacityCfgWrapper:
    opacity: LossOpacityCfg


class LossOpacity(Loss[LossOpacityCfg, LossOpacityCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        alpha = prediction.alpha
        valid_mask = batch["context"]["valid_mask"].float()
        opacity_loss = F.mse_loss(alpha, valid_mask, reduction="none").mean()
        # if self.cfg.type == "exp":
        #     opacity_loss = torch.exp(-(gaussians.opacities - 0.5) ** 2 / 0.05).mean()
        # elif self.cfg.type == "mean":
        #     opacity_loss = gaussians.opacities.mean()
        # elif self.cfg.type == "exp+mean":
        #     opacity_loss = 0.5 * torch.exp(-(gaussians.opacities - 0.5) ** 2 / 0.05).mean() + gaussians.opacities.mean()
        return self.cfg.weight * torch.nan_to_num(
            opacity_loss, nan=0.0, posinf=0.0, neginf=0.0
        )
