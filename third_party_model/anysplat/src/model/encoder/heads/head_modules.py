import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops as xops
from einops import rearrange
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class UnetExtractor(nn.Module):
    def __init__(self, in_channel=3, encoder_dim=[256, 256, 256], norm_fn="group"):
        super().__init__()
        self.in_ds = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(inplace=True),
        )

        self.res1 = nn.Sequential(
            ResidualBlock(64, encoder_dim[0], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[0], encoder_dim[0], norm_fn=norm_fn),
        )
        self.res2 = nn.Sequential(
            ResidualBlock(encoder_dim[0], encoder_dim[1], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[1], encoder_dim[1], norm_fn=norm_fn),
        )
        self.res3 = nn.Sequential(
            ResidualBlock(encoder_dim[1], encoder_dim[2], stride=2, norm_fn=norm_fn),
            ResidualBlock(encoder_dim[2], encoder_dim[2], norm_fn=norm_fn),
        )

    def forward(self, x):
        x = self.in_ds(x)
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)

        return x1, x2, x3


class MultiBasicEncoder(nn.Module):
    def __init__(self, output_dim=[128], encoder_dim=[64, 96, 128]):
        super(MultiBasicEncoder, self).__init__()

        # output convolution for feature
        self.conv2 = nn.Sequential(
            ResidualBlock(encoder_dim[2], encoder_dim[2], stride=1),
            nn.Conv2d(encoder_dim[2], encoder_dim[2] * 2, 3, padding=1),
        )

        # output convolution for context
        output_list = []
        for dim in output_dim:
            conv_out = nn.Sequential(
                ResidualBlock(encoder_dim[2], encoder_dim[2], stride=1),
                nn.Conv2d(encoder_dim[2], dim[2], 3, padding=1),
            )
            output_list.append(conv_out)

        self.outputs08 = nn.ModuleList(output_list)

    def forward(self, x):
        feat1, feat2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)

        outputs08 = [f(x) for f in self.outputs08]
        return outputs08, feat1, feat2


# attention processor for appreaance head


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        mlp_ratio=4.0,
        mlp_bias=False,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=mlp_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=64,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flashatt_v2=True,
    ):
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_q = RMSNorm(head_dim, eps=1e-5)
        self.norm_k = RMSNorm(head_dim, eps=1e-5)

        self.use_flashatt_v2 = use_flashatt_v2

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if self.use_flashatt_v2:
            qkv = qkv.permute(2, 0, 1, 3, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # (B, N, H, C)
            q, k = self.norm_q(q).to(v.dtype), self.norm_k(k).to(v.dtype)
            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                p=self.attn_drop_p,
            )
            x = rearrange(x, "b n h d -> b n (h d)")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=64,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flashatt_v2=True,
    ):
        super().__init__()
        assert dim % head_dim == 0, "dim must be divisible by head_dim"
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm_q = RMSNorm(head_dim, eps=1e-5)
        self.norm_k = RMSNorm(head_dim, eps=1e-5)

        self.use_flashatt_v2 = use_flashatt_v2

    def forward(self, x_q, x_kv):
        """
        x_q: query input (B, L_q, D)
        x_kv: key-value input (B, L_kv, D)
        Returns: same shape as query input (B, L_q, D)
        """
        B, N_q, C = x_q.shape
        _, N_kv, _ = x_kv.shape

        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads)
        k = self.k(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads)
        v = self.v(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads)

        if self.use_flashatt_v2:
            q, k = self.norm_q(q).to(v.dtype), self.norm_k(k).to(v.dtype)
            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),
                p=self.attn_drop_p,
            )
            x = rearrange(x, "b n h d -> b n (h d)")

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlockSelfAttn(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
        mlp_ratio=4.0,
        mlp_bias=False,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_flashatt_v2=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, bias=False)
        self.attn = SelfAttention(
            dim,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flashatt_v2=use_flashatt_v2,
        )
        self.norm2 = norm_layer(dim, bias=False)
        self.mlp = Mlp(
            in_features=dim,
            mlp_ratio=mlp_ratio,
            mlp_bias=mlp_bias,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input
        """
        y = self.attn(self.norm1(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBlockCrossAttn(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
        mlp_ratio=4.0,
        mlp_bias=False,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_flashatt_v2=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, bias=False)
        self.attn = CrossAttention(
            dim,
            head_dim=head_dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flashatt_v2=use_flashatt_v2,
        )
        self.norm2 = norm_layer(dim, bias=False)
        self.mlp = Mlp(
            in_features=dim,
            mlp_ratio=mlp_ratio,
            mlp_bias=mlp_bias,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x_list):
        """
        x_q: (B, L_q, D)
        x_kv: (B, L_kv, D)
        Returns: same shape as input
        """
        x_q, x_kv = x_list
        y = self.attn(self.norm1(x_q), self.norm1(x_kv))
        x = x_q + y
        x = x + self.mlp(self.norm2(x))
        return x


class AppearanceTransformer(nn.Module):
    def __init__(self, num_layers, attn_dim, head_dim, ca_incides=[1, 3, 5, 7]):
        super().__init__()
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        self.ca_incides = ca_incides

        for attn_index in range(num_layers):
            self.blocks.append(TransformerBlockSelfAttn(self.attn_dim, head_dim))
            self.blocks[-1].apply(_init_weights)

    def forward(self, x, use_checkpoint=True):
        """
        input_tokens: (B, L, D)
        aggregated_tokens: List of (B, L, D)
        Returns: B and D remain the same, L might change if there are merge layers
        """
        for block in self.blocks:
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        return x


if __name__ == "__main__":
    data = torch.ones((1, 3, 1024, 1024))

    model = UnetExtractor(in_channel=3, encoder_dim=[64, 96, 128])

    x1, x2, x3 = model(data)
    print(x1.shape, x2.shape, x3.shape)
