# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
# import dust3r.utils.path_to_croco
from .dpt_block import DPTOutputAdapter, Interpolate, make_fusion_block
from .head_modules import UnetExtractor, AppearanceTransformer, _init_weights
from .postprocess import postprocess

# class DPTOutputAdapter_fix(DPTOutputAdapter):
#     """
#     Adapt croco's DPTOutputAdapter implementation for dust3r:
#     remove duplicated weigths, and fix forward for dust3r
#     """
#
#     def init(self, dim_tokens_enc=768):
#         super().init(dim_tokens_enc)
#         # these are duplicated weights
#         del self.act_1_postprocess
#         del self.act_2_postprocess
#         del self.act_3_postprocess
#         del self.act_4_postprocess
#
#         self.scratch.refinenet1 = make_fusion_block(256 * 2, False, 1, expand=True)
#         self.scratch.refinenet2 = make_fusion_block(256 * 2, False, 1, expand=True)
#         self.scratch.refinenet3 = make_fusion_block(256 * 2, False, 1, expand=True)
#         # self.scratch.refinenet4 = make_fusion_block(256 * 2, False, 1)
#
#         self.depth_encoder = UnetExtractor(in_channel=3)
#         self.feat_up = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
#         self.out_conv = nn.Conv2d(256+3+4, 256, kernel_size=3, padding=1)
#         self.out_relu = nn.ReLU(inplace=True)
#
#         self.input_merger = nn.Sequential(
#             # nn.Conv2d(256+3+3+1, 256, kernel_size=3, padding=1),
#             nn.Conv2d(256+3+3, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#
#     def forward(self, encoder_tokens: List[torch.Tensor], depths, imgs, image_size=None, conf=None):
#         assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
#         # H, W = input_info['image_size']
#         image_size = self.image_size if image_size is None else image_size
#         H, W = image_size
#         # Number of patches in height and width
#         N_H = H // (self.stride_level * self.P_H)
#         N_W = W // (self.stride_level * self.P_W)
#           
#         # Hook decoder onto 4 layers from specified ViT layers
#         layers = [encoder_tokens[hook] for hook in self.hooks]
#
#         # Extract only task-relevant tokens and ignore global tokens.
#         layers = [self.adapt_tokens(l) for l in layers]
#
#         # Reshape tokens to spatial representation
#         layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]
#
#         layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
#         # Project layers to chosen feature dim
#         layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
#
#         # get depth features
#         depth_features = self.depth_encoder(depths)
#         depth_feature1, depth_feature2, depth_feature3 = depth_features
#
#         # Fuse layers using refinement stages
#         path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
#         path_3 = self.scratch.refinenet3(torch.cat([path_4, depth_feature3], dim=1), torch.cat([layers[2], depth_feature3], dim=1))
#         path_2 = self.scratch.refinenet2(torch.cat([path_3, depth_feature2], dim=1), torch.cat([layers[1], depth_feature2], dim=1))
#         path_1 = self.scratch.refinenet1(torch.cat([path_2, depth_feature1], dim=1), torch.cat([layers[0], depth_feature1], dim=1))
#         # path_3 = self.scratch.refinenet3(path_4, layers[2], depth_feature3)
#         # path_2 = self.scratch.refinenet2(path_3, layers[1], depth_feature2)
#         # path_1 = self.scratch.refinenet1(path_2, layers[0], depth_feature1)
#
#         path_1 = self.feat_up(path_1)
#         path_1 = torch.cat([path_1, imgs, depths], dim=1)
#         if conf is not None:
#             path_1 = torch.cat([path_1, conf], dim=1)
#         path_1 = self.input_merger(path_1)
#
#         # Output head
#         out = self.head(path_1)
#
#         return out


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess
        
        self.feat_up = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        self.input_merger = nn.Sequential(
            # nn.Conv2d(256+3+3+1, 256, kernel_size=3, padding=1),
            # nn.Conv2d(3+6, 256, 7, 1, 3),
            nn.Conv2d(3, 256, 7, 1, 3),
            nn.ReLU(),
        )
        
    def forward(self, encoder_tokens: List[torch.Tensor], depths, imgs, image_size=None, conf=None):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
        
        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        direct_img_feat = self.input_merger(imgs)
        # imgs = imgs.permute(0, 2, 3, 1).flatten(1, 2).contiguous()
        # # Pachify
        # patch_size = self.patch_size
        # hh = H // patch_size[0]
        # ww = W // patch_size[1]
        # direct_img_feat = rearrange(imgs, "b (hh ph ww pw) d -> b (hh ww) (ph pw d)", hh=hh, ww=ww, ph=patch_size[0], pw=patch_size[1])

        # actually, we just do interpolate here
        # path_1 = self.feat_up(path_1)
        path_1 = F.interpolate(path_1, size=(H, W), mode='bilinear', align_corners=True)
        path_1 = path_1 + direct_img_feat
        
        # path_1 = torch.cat([path_1, imgs], dim=1)

        # Output head
        out = self.head(path_1)
        
        return out, [path_4, path_3, path_2]


class PixelwiseTaskWithDPT(nn.Module):
    """ DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        
        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, depths, imgs, img_info, conf=None):
        out, interm_feats = self.dpt(x, depths, imgs, image_size=(img_info[0], img_info[1]), conf=conf)
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out, interm_feats
    
class AttnBasedAppearanceHead(nn.Module):
    """
    Attention head Appearence Reconstruction
    """

    def __init__(self, num_channels, patch_size, feature_dim, last_dim, hooks_idx, dim_tokens, postprocess, depth_mode, conf_mode, head_type='gs_params'):
        super().__init__()

        self.num_channels = num_channels
        self.patch_size = patch_size

        self.hooks = hooks_idx

        assert len(set(dim_tokens)) == 1

        self.tokenizer = nn.Linear(3 * self.patch_size[0] ** 2, dim_tokens[0], bias=False)

        self.attn_processor = AppearanceTransformer(num_layers=4, attn_dim=dim_tokens[0] * 2, head_dim=feature_dim)

        self.token_decoder = nn.Sequential(
            nn.LayerNorm(dim_tokens[0] * 2, bias=False),
            nn.Linear(
                dim_tokens[0] * 2, self.num_channels * (self.patch_size[0] ** 2),
                bias=False,
            )
        )
        self.token_decoder.apply(_init_weights)


    def img_pts_tokenizer(self, imgs, pts3d):
        B, V, _, H, W = imgs.shape
        pts3d = pts3d.flatten(2, 3).contiguous()
        imgs = imgs.permute(0, 1, 3, 4, 2).flatten(2, 3).contiguous()
        mean = pts3d.mean(dim=-2, keepdim=True) # (B, V, 1, 3)
        z_median = torch.median(torch.norm(pts3d, dim=-1, keepdim=True), dim=2, keepdim=True)[0] # (B, V, 1, 1)
        pts3d_normed = (pts3d - mean) / (z_median + 1e-8) # (B, V, N, 3)

        input = imgs #torch.cat([pts3d_normed, imgs], dim=-1) # (B, V, H*W, 9)
        # Pachify
        patch_size = self.patch_size
        hh = H // patch_size[0]
        ww = W // patch_size[1]
        input = rearrange(input, "b v (hh ph ww pw) d -> (b v) (hh ww) (ph pw d)", hh=hh, ww=ww, ph=patch_size[0], pw=patch_size[1])
        # Tokenize the input images
        input_tokens = self.tokenizer(input)
        return input_tokens

    def forward(self, x, depths, imgs, img_info, conf=None):
        B, V, H, W = img_info
        input_tokens = rearrange(self.img_pts_tokenizer(imgs, depths), "(b v) l d -> b (v l) d", b=B, v=V)

        # Hook decoder onto 4 layers from specified ViT layers
        layer_tokens = [rearrange(x[hook].detach(), "(b v) l d -> b (v l) d", b=B, v=V) for hook in self.hooks]

        tokens = self.attn_processor(torch.cat([input_tokens, layer_tokens[-1]], dim=-1))

        gaussian_params = self.token_decoder(tokens)

        patch_size = self.patch_size
        hh = H // patch_size[0]
        ww = W // patch_size[1]
        gaussians = rearrange(gaussian_params, "b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d", v=V, hh=hh, ww=ww, ph=patch_size[0], pw=patch_size[1])
        return gaussians.view(B, V, H*W, -1)

def create_gs_dpt_head(net, has_conf=False, out_nchan=3, postprocess_func=postprocess):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = net.feature_dim
    last_dim = feature_dim//2
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    try:    
        patch_size = net.patch_size
    except:
        patch_size = (16, 16)

    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                patch_size=patch_size,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess_func,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='gs_params')