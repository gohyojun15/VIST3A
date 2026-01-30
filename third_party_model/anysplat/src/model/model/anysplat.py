import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import huggingface_hub
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from third_party_model.anysplat.src.model.decoder.decoder_splatting_cuda import (
    DecoderSplattingCUDA,
    DecoderSplattingCUDACfg,
)
from third_party_model.anysplat.src.model.encoder.anysplat import (
    EncoderAnySplat,
    EncoderAnySplatCfg,
    OpacityMappingCfg,
)
from third_party_model.anysplat.src.model.encoder.common.gaussian_adapter import (
    GaussianAdapterCfg,
)


class AnySplat(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_cfg: EncoderAnySplatCfg,
        decoder_cfg: DecoderSplattingCUDACfg,
    ):
        super(AnySplat, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.build_encoder(encoder_cfg)
        self.build_decoder(decoder_cfg)

    def convert_nested_config(self, cfg_dict: dict, target_class: type):
        """Convert nested dictionary config to dataclass instance

        Args:
            cfg_dict: Configuration dictionary or already converted object
            target_class: Target dataclass type to convert to

        Returns:
            Instance of target_class
        """
        if isinstance(cfg_dict, dict):
            # Convert dict to dataclass
            return target_class(**cfg_dict)
        elif isinstance(cfg_dict, target_class):
            # Already converted, return as is
            return cfg_dict
        elif cfg_dict is None:
            # Handle None case
            return None
        else:
            raise ValueError(f"Cannot convert {type(cfg_dict)} to {target_class}")

    def convert_config_recursively(self, cfg_obj, conversion_map: dict):
        """Convert nested configurations recursively using a conversion map

        Args:
            cfg_obj: Configuration object to convert
            conversion_map: Dict mapping field names to their target classes
                           e.g., {'gaussian_adapter': GaussianAdapterCfg}

        Returns:
            Converted configuration object
        """
        if not hasattr(cfg_obj, "__dict__"):
            return cfg_obj

        cfg_dict = cfg_obj.__dict__.copy()

        for field_name, target_class in conversion_map.items():
            if field_name in cfg_dict:
                cfg_dict[field_name] = self.convert_nested_config(
                    cfg_dict[field_name], target_class
                )

        # Return new instance of the same type
        return type(cfg_obj)(**cfg_dict)

    def convert_encoder_config(
        self, encoder_cfg: EncoderAnySplatCfg
    ) -> EncoderAnySplatCfg:
        """Convert all nested configurations in encoder_cfg"""
        conversion_map = {
            "gaussian_adapter": GaussianAdapterCfg,
            "opacity_mapping": OpacityMappingCfg,
        }

        return self.convert_config_recursively(encoder_cfg, conversion_map)

    def build_encoder(self, encoder_cfg: EncoderAnySplatCfg):
        # Convert nested configurations using the helper method
        encoder_cfg = self.convert_encoder_config(encoder_cfg)
        self.encoder = EncoderAnySplat(encoder_cfg)

    def build_decoder(self, decoder_cfg: DecoderSplattingCUDACfg):
        self.decoder = DecoderSplattingCUDA(decoder_cfg)

    @torch.no_grad()
    def inference(
        self,
        context_image: torch.Tensor,
        training_for_stitching: bool = False,
    ):
        self.encoder.distill = False
        encoder_output = self.encoder(
            context_image,
            global_step=0,
            visualization_dump=None,
            training_for_stitching=training_for_stitching,
        )
        if training_for_stitching:
            return encoder_output
        else:
            gaussians, pred_context_pose = (
                encoder_output.gaussians,
                encoder_output.pred_context_pose,
            )
            return gaussians, pred_context_pose

    def forward(
        self,
        context_image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        near: float = 0.01,
        far: float = 100.0,
    ):
        b, v, c, h, w = context_image.shape
        device = context_image.device
        encoder_output = self.encoder(
            context_image, global_step, visualization_dump=visualization_dump
        )
        gaussians, pred_context_pose = (
            encoder_output.gaussians,
            encoder_output.pred_context_pose,
        )
        output = self.decoder.forward(
            gaussians,
            pred_context_pose["extrinsic"],
            pred_context_pose["intrinsic"],
            torch.ones(1, v, device=device) * near,
            torch.ones(1, v, device=device) * far,
            (h, w),
            "depth",
        )
        return encoder_output, output
