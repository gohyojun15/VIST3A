from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from third_party_model.anysplat.src.dataset.types import BatchedExample
from third_party_model.anysplat.src.misc.nn_module_tools import convert_to_buffer
from third_party_model.anysplat.src.model.decoder.decoder import DecoderOutput
from third_party_model.anysplat.src.model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    conf: bool = False
    alpha: bool = False
    mask: bool = False


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        depth_dict: dict | None,
        global_step: int,
    ) -> Float[Tensor, ""]:
        image = (batch["context"]["image"] + 1) / 2

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        if self.cfg.mask or self.cfg.alpha or self.cfg.conf:
            if self.cfg.mask:
                mask = batch["context"]["valid_mask"]
            elif self.cfg.alpha:
                mask = prediction.alpha
            elif self.cfg.conf:
                mask = depth_dict["conf_valid_mask"]
            b, v, c, h, w = prediction.color.shape
            expanded_mask = mask.unsqueeze(2).expand(-1, -1, c, -1, -1)
            masked_pred = prediction.color * expanded_mask
            masked_img = image * expanded_mask

            loss = self.lpips.forward(
                rearrange(masked_pred, "b v c h w -> (b v) c h w"),
                rearrange(masked_img, "b v c h w -> (b v) c h w"),
                normalize=True,
            )
        else:
            loss = self.lpips.forward(
                rearrange(prediction.color, "b v c h w -> (b v) c h w"),
                rearrange(image, "b v c h w -> (b v) c h w"),
                normalize=True,
            )
        return self.cfg.weight * torch.nan_to_num(
            loss.mean(), nan=0.0, posinf=0.0, neginf=0.0
        )
