import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from loguru import logger

from models.anysplat_stitched import AnySplatStitched
from third_party_model.anysplat.src.model.model.anysplat import AnySplat
from utils.wan_utils import AutoencoderKLWan as AutoencoderKLWan_wan


class StitchVAE3D(nn.Module):
    def __init__(
        self,
        diffusion_vae: nn.Module,
        feedforward_model: nn.Module,
        device: torch.device,
        stitching_layer_location: str,
        stitching_layer_config: str,
        resolution: int,
        stitching_layer_init_path: str = None,
    ):
        """
        The class to stitch a 3D feedforward model into a diffusion VAE.
        """

        super().__init__()
        self.device = device
        self.diffusion_vae = diffusion_vae
        self.vae_latent_dimension = self.get_vae_latent_dimension(
            diffusion_vae, resolution
        )
        # feedforward 3d model
        self.stitched_3d_model, self.feedforward_dimension = (
            self.preprocess_feedforward_model(
                feedforward_model, stitching_layer_location
            )
        )
        # stitching layer
        self.stitching_layer, self.pre_upsample_layer = self.get_stitching_layer(
            stitching_layer_config,
            stitching_layer_init_path,
        )

    def get_vae_latent_dimension(self, diffusion_vae: nn.Module, resolution: int):
        if isinstance(diffusion_vae, AutoencoderKLWan) or isinstance(
            diffusion_vae, AutoencoderKLWan_wan
        ):
            T = 13  # hardcoded for WAN
            H = resolution // 8
            W = resolution // 8
            C = 16
            vae_latent_dimension = torch.tensor([T, C, H, W])
        else:
            raise NotImplementedError(
                f"VAE latent dimension extraction not implemented for {type(diffusion_vae)}"
            )
        return vae_latent_dimension

    def preprocess_feedforward_model(
        self, feedforward_model: nn.Module, stitching_layer: str
    ):
        if isinstance(feedforward_model, AnySplat):
            model = AnySplatStitched(feedforward_model, stitching_layer)
            T = 13  # hardcoded for WAN
            C = 1024  # anysplat latent dimension
            feedforward_dimension = torch.tensor([T, C, 448, 448])
        else:
            raise NotImplementedError(
                f"Feedforward model preprocessing not implemented for {type(feedforward_model)}"
            )
        return model, feedforward_dimension

    def get_stitching_layer(
        self,
        stitching_layer_config: str,
        stitching_layer_init_path: str = None,
    ):
        stitching_layer = stitching_layer_config.build(
            in_channels=self.vae_latent_dimension[1]
        )

        def upsampling_layer(x):
            T_vae = x.shape[2]
            T_original = (T_vae - 1) * 4 + 1
            x = torch.nn.functional.interpolate(
                x,
                size=[
                    T_original,
                    self.vae_latent_dimension[2],
                    self.vae_latent_dimension[3],
                ],
                mode="trilinear",
                align_corners=True,
            )
            return x

        if stitching_layer_init_path is not None:
            logger.info(f"Loading the stitching layer from {stitching_layer_init_path}")
            # load the weight
            state_dict = torch.load(
                stitching_layer_init_path,
                map_location=self.device,
                weights_only=False,
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            stitching_layer.load_state_dict(state_dict, strict=True)
        return stitching_layer, upsampling_layer

    def vae_encoder_forward(self, images, decode=False):
        """
        The forward function for the VAE encoder
        Args:
            images (torch.Tensor): the input images, shape (B, C, T, H, W)
        Returns:
            latents (torch.Tensor): the latent variables, shape (B, C_out, T_out, H_out, W_out)
        """
        # we suppose that diffusion_vae follows diffuser's convention.
        latents = self.diffusion_vae.encode(images).latent_dist.sample()
        if decode:
            feedforward_image = self.diffusion_vae.decode(latents)[0]
        else:
            feedforward_image = None
        return latents, feedforward_image

    def forward(self, images, feedforward_image, train=False):
        with torch.no_grad():
            latent, feedforward_image_ = self.vae_encoder_forward(images, decode=False)

            if feedforward_image_ is not None:
                feedforward_image = F.interpolate(
                    feedforward_image_,
                    size=(
                        feedforward_image.shape[2],
                        self.feedforward_dimension[2],
                        self.feedforward_dimension[3],
                    ),
                    mode="trilinear",
                    align_corners=True,
                )
            latent = self.pre_upsample_layer(latent)

        stitching_latent = self.stitching_layer(latent)
        final_output = self.stitched_3d_model_forward(
            stitching_latent,
            feedforward_image,
            train=train,
        )
        return final_output

    def forward_with_latent(self, latent, feedforward_image, train=False):
        latent = self.pre_upsample_layer(latent)
        stitching_latent = self.stitching_layer(latent)
        final_output = self.stitched_3d_model_forward(
            stitching_latent,
            feedforward_image,
            train=train,
        )
        return final_output

    def stitched_3d_model_forward(self, latents, feedforward_image, train=False):
        if isinstance(self.stitched_3d_model, AnySplatStitched):
            output = self.stitched_3d_model(latents, feedforward_image, train=train)
        else:
            raise NotImplementedError(
                f"Stitched 3D model forward not implemented for {type(self.stitched_3d_model)}"
            )
        return output
