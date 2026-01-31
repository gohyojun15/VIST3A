"""
Utils for Wan model.
About text embedding and gradient checkpointing of Wan VAE
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.activations import get_activation
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.utils.accelerate_utils import apply_forward_hook


@torch.no_grad()
def compute_wan_text_embeddings(
    prompt, text_encoders, tokenizers, max_sequence_length=226, device=None
):
    # Tokenize and pad to a fixed length so the UNet receives consistent shapes.
    # Output shape: [B, max_sequence_length, hidden_dim].
    dtype = text_encoders.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    # batch_size = len(prompt)

    text_inputs = tokenizers(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoders(
        text_input_ids.to(device), mask.to(device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds


"""
WAN gradient checkpointing implementation.
"""


CACHE_T = 2


@dataclass
class CacheState:
    """Immutable cache state to avoid in-place modifications"""

    # Cache slots are used by causal convs to preserve temporal context across chunks.
    cache_list: List[Optional[Union[torch.Tensor, str]]]
    current_idx: int

    def get(self, idx: int) -> Optional[Union[torch.Tensor, str]]:
        if idx < len(self.cache_list):
            return self.cache_list[idx]
        return None

    def set(self, idx: int, value: Union[torch.Tensor, str]) -> "CacheState":
        """Returns a new CacheState with updated value"""
        new_list = self.cache_list.copy()
        if idx < len(new_list):
            new_list[idx] = value
        return CacheState(new_list, self.current_idx)

    def increment_idx(self) -> "CacheState":
        """Returns a new CacheState with incremented index"""
        return CacheState(self.cache_list.copy(), self.current_idx + 1)


class WanCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )

        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        # cache_x is concatenated in time to maintain causality for chunked inference.
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class WanRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class WanUpsample(nn.Upsample):
    r"""
    Perform upsampling while ensuring the output tensor has the same data type as the input.

    Args:
        x (torch.Tensor): Input tensor to be upsampled.

    Returns:
        torch.Tensor: Upsampled tensor with the same data type as the input.
    """

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = WanCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:
            self.resample = nn.Identity()
        self.gradient_checkpointing = False

    def forward(
        self, x, cache_state: Optional[CacheState] = None
    ) -> Tuple[torch.Tensor, Optional[CacheState]]:
        b, c, t, h, w = x.size()
        new_cache_state = cache_state

        if self.mode == "upsample3d":
            # For temporal upsampling, use cached frames to keep causality between chunks.
            if new_cache_state is not None:
                idx = new_cache_state.current_idx
                cached_x = new_cache_state.get(idx)

                if cached_x is None:
                    # First time, mark as "Rep"
                    new_cache_state = new_cache_state.set(
                        new_cache_state.current_idx, "Rep"
                    ).increment_idx()
                    cached_x = new_cache_state.get(new_cache_state.current_idx)
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and cached_x != "Rep"
                        and cached_x is not None
                    ):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                cached_x[:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and cached_x == "Rep"
                        and cached_x is not None
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )

                    if cached_x == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, cached_x)

                    new_cache_state = new_cache_state.set(
                        new_cache_state.current_idx, cache_x
                    ).increment_idx()

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            # For temporal downsampling, cache the last frame to align next chunk.
            if new_cache_state is not None:
                idx = new_cache_state.current_idx
                cached_x = new_cache_state.get(idx)

                if cached_x is None:
                    new_cache_state = new_cache_state.set(
                        new_cache_state.current_idx, x.clone()
                    ).increment_idx()
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([cached_x[:, :, -1:, :, :], x], 2))
                    new_cache_state = new_cache_state.set(
                        new_cache_state.current_idx, cache_x
                    ).increment_idx()

        return x, new_cache_state


class WanResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self, x, cache_state: Optional[CacheState] = None
    ) -> Tuple[torch.Tensor, Optional[CacheState]]:
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        new_cache_state = cache_state
        if new_cache_state is not None:
            # Cache a few frames for causal convs; prepend last cached frame if needed.
            # This ensures each chunk sees immediate temporal context.
            idx = new_cache_state.current_idx
            cached_x = new_cache_state.get(idx)
            cache_x = x[:, :, -CACHE_T:, :, :].clone()

            if cache_x.shape[2] < 2 and cached_x is not None:
                cache_x = torch.cat(
                    [cached_x[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )

            x = self.conv1(x, cached_x)
            new_cache_state = new_cache_state.set(
                new_cache_state.current_idx, cache_x
            ).increment_idx()
        else:
            x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        if new_cache_state is not None:
            # Repeat cache handling for the second conv.
            # Keeps causal behavior consistent across both residual convs.
            idx = new_cache_state.current_idx
            cached_x = new_cache_state.get(idx)
            cache_x = x[:, :, -CACHE_T:, :, :].clone()

            if cache_x.shape[2] < 2 and cached_x is not None:
                cache_x = torch.cat(
                    [cached_x[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )

            x = self.conv2(x, cached_x)
            new_cache_state = new_cache_state.set(
                new_cache_state.current_idx, cache_x
            ).increment_idx()
        else:
            x = self.conv2(x)

        # Add residual connection
        return x + h, new_cache_state


class WanAttentionBlock(nn.Module):
    r"""
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = WanRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        # Flatten time into batch for 2D attention over spatial tokens per frame.
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)

        x = (
            x.squeeze(1)
            .permute(0, 2, 1)
            .reshape(batch_size * time, channels, height, width)
        )

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class WanMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self, x, cache_state: Optional[CacheState] = None
    ) -> Tuple[torch.Tensor, Optional[CacheState]]:
        # First residual block
        x, new_cache_state = self.resnets[0](x, cache_state)

        # Process through attention and residual blocks
        # Attention is optional and only applied when configured at this stage.
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                if self.gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(attn, x, use_reentrant=False)
                else:
                    x = attn(x)

            if self.gradient_checkpointing:
                x, new_cache_state = torch.utils.checkpoint.checkpoint(
                    resnet, x, new_cache_state, use_reentrant=False
                )
            else:
                x, new_cache_state = resnet(x, new_cache_state)

        return x, new_cache_state


class WanEncoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(WanAttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(WanResample(out_dim, mode=mode))
                scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self, x, cache_state: Optional[CacheState] = None
    ) -> Tuple[torch.Tensor, Optional[CacheState]]:
        # x shape: [B, C, T, H, W], outputs latent features with downsampled T/H/W.
        new_cache_state = cache_state
        if new_cache_state is not None:
            idx = new_cache_state.current_idx
            cached_x = new_cache_state.get(idx)
            cache_x = x[:, :, -CACHE_T:, :, :].clone()

            if cache_x.shape[2] < 2 and cached_x is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [cached_x[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )

            x = self.conv_in(x, cached_x)
            new_cache_state = new_cache_state.set(
                new_cache_state.current_idx, cache_x
            ).increment_idx()
        else:
            x = self.conv_in(x)

        ## downsamples
        # Each WanResample may downsample spatially and optionally temporally.
        for layer in self.down_blocks:
            if isinstance(layer, WanResidualBlock):
                x, new_cache_state = layer(x, new_cache_state)
            elif isinstance(layer, WanResample):
                x, new_cache_state = layer(x, new_cache_state)
            else:  # WanAttentionBlock
                x = layer(x)

        ## middle
        x, new_cache_state = self.mid_block(x, new_cache_state)

        ## head
        # Final projection to z_dim channels before quant_conv.
        x = self.norm_out(x)
        x = self.nonlinearity(x)

        if new_cache_state is not None:
            idx = new_cache_state.current_idx
            cached_x = new_cache_state.get(idx)
            cache_x = x[:, :, -CACHE_T:, :, :].clone()

            if cache_x.shape[2] < 2 and cached_x is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [cached_x[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )

            x = self.conv_out(x, cached_x)
            new_cache_state = new_cache_state.set(
                new_cache_state.current_idx, cache_x
            ).increment_idx()
        else:
            x = self.conv_out(x)

        return x, new_cache_state


class WanUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(
        self, x, cache_state: Optional[CacheState] = None
    ) -> Tuple[torch.Tensor, Optional[CacheState]]:
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            cache_state (CacheState, optional): Cache state for causal convolutions

        Returns:
            Tuple[torch.Tensor, Optional[CacheState]]: Output tensor and updated cache state
        """
        new_cache_state = cache_state
        # Residual blocks refine features at current resolution.
        for resnet in self.resnets:
            if self.gradient_checkpointing:
                x, new_cache_state = torch.utils.checkpoint.checkpoint(
                    resnet, x, new_cache_state, use_reentrant=False
                )
            else:
                x, new_cache_state = resnet(x, new_cache_state)

        if self.upsamplers is not None:
            # Optional 2D or 3D upsampling between decoder stages.
            if self.gradient_checkpointing:
                x, new_cache_state = torch.utils.checkpoint.checkpoint(
                    self.upsamplers[0], x, new_cache_state, use_reentrant=False
                )
            else:
                x, new_cache_state = self.upsamplers[0](x, new_cache_state)

        return x, new_cache_state


class WanDecoder3d(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0:
                in_dim = in_dim // 2

            # Determine if we need upsampling
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

            # Create and add the upsampling block
            up_block = WanUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            # Update scale for next iteration
            if upsample_mode is not None:
                scale *= 2.0

        # output blocks
        self.norm_out = WanRMS_norm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, 3, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(
        self, x, cache_state: Optional[CacheState] = None
    ) -> Tuple[torch.Tensor, Optional[CacheState]]:
        # x shape: [B, z_dim, T, H, W], outputs reconstructed frames.
        ## conv1
        new_cache_state = cache_state
        if new_cache_state is not None:
            idx = new_cache_state.current_idx
            cached_x = new_cache_state.get(idx)
            cache_x = x[:, :, -CACHE_T:, :, :].clone()

            if cache_x.shape[2] < 2 and cached_x is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [cached_x[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )

            x = self.conv_in(x, cached_x)
            new_cache_state = new_cache_state.set(
                new_cache_state.current_idx, cache_x
            ).increment_idx()
        else:
            x = self.conv_in(x)

        ## middle
        if self.gradient_checkpointing:
            x, new_cache_state = torch.utils.checkpoint.checkpoint(
                self.mid_block, x, new_cache_state, use_reentrant=False
            )
        else:
            x, new_cache_state = self.mid_block(x, new_cache_state)

        ## upsamples
        # Each WanUpBlock may upsample spatially and optionally temporally.
        for up_block in self.up_blocks:
            if self.gradient_checkpointing:
                x, new_cache_state = torch.utils.checkpoint.checkpoint(
                    up_block, x, new_cache_state, use_reentrant=False
                )
            else:
                x, new_cache_state = up_block(x, new_cache_state)

        ## head
        # Final projection back to RGB.
        x = self.norm_out(x)
        x = self.nonlinearity(x)

        if new_cache_state is not None:
            idx = new_cache_state.current_idx
            cached_x = new_cache_state.get(idx)
            cache_x = x[:, :, -CACHE_T:, :, :].clone()

            if cache_x.shape[2] < 2 and cached_x is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [cached_x[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    self.conv_out, x, cached_x, use_reentrant=False
                )
            else:
                x = self.conv_out(x, cached_x)

            new_cache_state = new_cache_state.set(
                new_cache_state.current_idx, cache_x
            ).increment_idx()
        else:
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    self.conv_out, x, use_reentrant=False
                )
            else:
                x = self.conv_out(x)

        return x, new_cache_state


class AutoencoderKLWan(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [Wan 2.1].

    This model inherits from [ModelMixin]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: List[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        # Each True in temperal_downsample halves the time dimension at that stage.
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = WanEncoder3d(
            base_dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.quant_conv = WanCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(z_dim, z_dim, 1)

        self.decoder = WanDecoder3d(
            base_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency during training. for only decoder"""
        self.decoder.apply(
            lambda module: self._set_gradient_checkpointing(module, value=True)
        )

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing. for only decoder"""
        self.decoder.apply(
            lambda module: self._set_gradient_checkpointing(module, value=False)
        )

    def _count_conv3d(self, model):
        """Count the number of WanCausalConv3d layers in the model."""
        count = 0
        for m in model.modules():
            if isinstance(m, WanCausalConv3d):
                count += 1
        return count

    def _create_cache_state(self, model) -> CacheState:
        """Create an initial cache state for the model."""
        num_convs = self._count_conv3d(model)
        # Add extra for quant_conv or post_quant_conv
        return CacheState([None] * (num_convs + 1), 0)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Internal encoding method that handles caching."""
        # Encode in temporal chunks to reuse causal conv cache and reduce memory.
        cache_state = self._create_cache_state(self.encoder)

        ## cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out = None

        for i in range(iter_):
            cache_state.current_idx = 0
            if i == 0:
                # First chunk: only the first frame.
                out, cache_state = self.encoder(x[:, :, :1, :, :], cache_state)
            else:
                # Subsequent chunks: stride by 4 frames to respect temporal downsample.
                out_, cache_state = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :], cache_state
                )
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        # Split into mean/logvar for diagonal Gaussian posterior.
        mu, logvar = enc[:, : self.z_dim, :, :, :], enc[:, self.z_dim :, :, :, :]
        enc = torch.cat([mu, logvar], dim=1)
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (torch.Tensor): Input batch of images.
            return_dict (bool, *optional*, defaults to True):
                Whether to return a [~models.autoencoder_kl.AutoencoderKLOutput] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If return_dict is True, a
                [~models.autoencoder_kl.AutoencoderKLOutput] is returned, otherwise a plain tuple is returned.
        """
        if (
            hasattr(self.encoder, "gradient_checkpointing")
            and self.encoder.gradient_checkpointing
        ):
            h = torch.utils.checkpoint.checkpoint(self._encode, x, use_reentrant=False)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        """Internal decoding method that handles caching."""
        # Decode one latent time step at a time to preserve causal state.
        cache_state = self._create_cache_state(self.decoder)

        iter_ = z.shape[2]
        x = self.post_quant_conv(z)
        out = None

        for i in range(iter_):
            cache_state.current_idx = 0
            if i == 0:
                # First step: initialize decoder cache.
                # out, cache_state = self.decoder(x[:, :, i : i + 1, :, :], cache_state)
                out, cache_state = torch.utils.checkpoint.checkpoint(
                    self.decoder,
                    x[:, :, i : i + 1, :, :],
                    cache_state,
                    use_reentrant=False,
                )
            else:
                # Subsequent steps: append decoded frames to output.
                # out_, cache_state = self.decoder(x[:, :, i : i + 1, :, :], cache_state)
                out_, cache_state = torch.utils.checkpoint.checkpoint(
                    self.decoder,
                    x[:, :, i : i + 1, :, :],
                    cache_state,
                    use_reentrant=False,
                )
                out = torch.cat([out, out_], 2)

        # Clamp to match typical VAE output range used in diffusion pipelines.
        out = torch.clamp(out, min=-1.0, max=1.0)

        if not return_dict:
            return (out,)

        return DecoderOutput(sample=out)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images.

        Args:
            z (torch.Tensor): Input batch of latent vectors.
            return_dict (bool, *optional*, defaults to True):
                Whether to return a [~models.vae.DecoderOutput] instead of a plain tuple.

        Returns:
            [~models.vae.DecoderOutput] or tuple:
                If return_dict is True, a [~models.vae.DecoderOutput] is returned, otherwise a plain tuple is
                returned.
        """
        if (
            hasattr(self.decoder, "gradient_checkpointing")
            and self.decoder.gradient_checkpointing
        ):
            decoded = torch.utils.checkpoint.checkpoint(
                lambda z: self._decode(z, return_dict=False)[0], z, use_reentrant=False
            )
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Args:
            sample (torch.Tensor): Input sample.
            return_dict (bool, *optional*, defaults to True):
                Whether or not to return a [DecoderOutput] instead of a plain tuple.
        """
        # Full VAE pass: encode -> sample/mode -> decode.
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            # Stochastic sampling during training.
            z = posterior.sample(generator=generator)
        else:
            # Deterministic mode for evaluation/reconstruction.
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec

    def clear_cache(self):
        """Clear cache - for compatibility with original implementation."""
        # In the gradient checkpointing version, we create fresh cache states
        # for each forward pass, so this method is not needed but kept for compatibility
        pass
