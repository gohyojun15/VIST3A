"""
Utility functions to load third-party models.
"""

import argparse

import torch
from diffusers import AutoencoderKLWan
from loguru import logger

from third_party_model.anysplat.src.model.model.anysplat import AnySplat


def load_feedforward_model(args: argparse.Namespace, device: torch.device):
    """
    Load a feedforward 3D model based on the provided arguments.
    """
    if args.feedforward_model == "anysplat":
        logger.info(f"Loading AnySplat model with device {device}")
        # Uses public Hugging Face weights; customize here for local checkpoints.
        model = AnySplat.from_pretrained(
            "lhjiang/anysplat",
        )
        model.to(device)
    else:
        raise NotImplementedError(
            f"Feedforward model {args.feedforward_model} is not implemented."
        )
    return model


def load_vae(args: argparse.Namespace, device: torch.device):
    """
    Load a Video VAE based on the provided arguments.
    """
    if args.video_model == "wan":
        model_id = (
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # maybe 14B and 1.3B weight are same.
        )
        logger.info(f"Loading Wan VAE from {model_id} with device {device}")
        # Only the VAE subfolder is needed for stitching.
        video_vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        video_vae.to(device)
    else:
        raise NotImplementedError(
            f"Video diffusion model {args.video_diffusion_model} is not implemented."
        )
    return video_vae


def cast_to_bfloat16(model: torch.nn.Module):
    """
    Cast the model's Conv2d and Linear layers to bfloat16, except for layers in 'head'.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            # Keep heads in fp32 for stability; cast the rest to bfloat16 for speed/memory.
            if "head" in str.lower(name):
                continue
            if isinstance(module, torch.nn.Conv2d):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)
            elif isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.to(torch.bfloat16)
                if module.bias is not None:
                    module.bias.data = module.bias.data.to(torch.bfloat16)
