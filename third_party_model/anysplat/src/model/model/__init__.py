from typing import Optional, Union

from ..encoder import Encoder
from ..encoder.visualization.encoder_visualizer import EncoderVisualizer
from ..encoder.anysplat import EncoderAnySplat, EncoderAnySplatCfg
from ..decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg
from torch import nn
from .anysplat import AnySplat

MODELS = {
    "anysplat": AnySplat,
}

EncoderCfg = Union[EncoderAnySplatCfg]
DecoderCfg = DecoderSplattingCUDACfg


# hard code for now
def get_model(encoder_cfg: EncoderCfg, decoder_cfg: DecoderCfg) -> nn.Module:
    model = MODELS['anysplat'](encoder_cfg, decoder_cfg)
    return model
