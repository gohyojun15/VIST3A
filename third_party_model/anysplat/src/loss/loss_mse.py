from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor
import torch
from third_party_model.anysplat.src.dataset.types import BatchedExample
from third_party_model.anysplat.src.model.decoder.decoder import DecoderOutput
from third_party_model.anysplat.src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    conf: bool = False
    mask: bool = False
    alpha: bool = False


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Get alpha and valid mask from inputs
        alpha = prediction.alpha
        # valid_mask = torch.ones_like(alpha, device=alpha.device).bool()
        valid_mask = batch["context"]["valid_mask"]

        # # only for objaverse
        # if batch['context']['valid_mask'].sum() > 0:
        #     valid_mask = batch['context']['valid_mask']

        # Determine which mask to use based on config
        if self.cfg.mask:
            mask = valid_mask
        elif self.cfg.alpha:
            mask = alpha
        elif self.cfg.conf:
            mask = depth_dict["conf_valid_mask"]
        else:
            mask = torch.ones_like(alpha, device=alpha.device).bool()

        # Rearrange and mask predicted and ground truth images
        pred_img = prediction.color.permute(0, 1, 3, 4, 2)[mask]
        gt_img = ((batch["context"]["image"][:, batch["using_index"]] + 1) / 2).permute(
            0, 1, 3, 4, 2
        )[mask]

        delta = pred_img - gt_img

        return self.cfg.weight * torch.nan_to_num(
            (delta**2).mean(), nan=0.0, posinf=0.0, neginf=0.0
        )
