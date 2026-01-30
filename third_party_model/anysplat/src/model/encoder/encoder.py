from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from third_party_model.anysplat.src.dataset.types import BatchedViews, DataShim

from ..types import Gaussians

T = TypeVar("T")


@dataclass
class EncoderOutput:
    gaussians: Gaussians
    pred_pose_enc_list: list[Float[Tensor, "batch view 6"]] | None
    pred_context_pose: dict | None
    depth_dict: dict | None
    infos: dict | None
    distill_infos: dict | None
    last_pred_pose_enc: Float[Tensor, "batch view 6"] | None = None


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
    ) -> Gaussians:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
