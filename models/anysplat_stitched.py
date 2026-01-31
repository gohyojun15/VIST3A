import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch.utils.checkpoint import checkpoint

from third_party_model.anysplat.src.model.encoder.encoder import EncoderOutput
from third_party_model.anysplat.src.model.encoder.vggt.models.aggregator import (
    slice_expand_and_flatten,
)
from third_party_model.anysplat.src.model.encoder.vggt.utils.geometry import (
    batchify_unproject_depth_map_to_point_map,
)
from third_party_model.anysplat.src.model.encoder.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from third_party_model.anysplat.src.model.model.anysplat import AnySplat


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W
    diff = prediction - target

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = prediction.shape[0] * prediction.shape[1] * prediction.shape[2]
    image_loss = torch.sum(image_loss) / divisor
    return image_loss


def gradient_loss_multi_scale(
    prediction, target, scales=4, gradient_loss_fn=gradient_loss
):
    """
    Compute gradient loss across multiple scales
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
        )

    total = total / scales
    return total


class TaskLossAnySplat:
    def __init__(self):
        pass

    def __call__(self, stitched_out: tuple, ff_out: tuple):
        # Loss aligns stitched outputs to the frozen feedforward AnySplat outputs.
        # Each term targets a specific prediction head (depth, Gaussians, pose, etc.).
        (
            stitched_encoder_output,
            stitched_anchor_feats,
            stitched_conf,
            stitched_depth_conf,
        ) = stitched_out
        ff_encoder_output, ff_anchor_feats, ff_conf, ff_depth_conf = ff_out

        depth_loss = F.l1_loss(
            stitched_encoder_output.depth_dict["depth"],
            ff_encoder_output.depth_dict["depth"],
        )
        depth_loss_grad = gradient_loss_multi_scale(
            stitched_encoder_output.depth_dict["depth"],
            ff_encoder_output.depth_dict["depth"],
        )
        gaussian_mean_loss = F.l1_loss(
            stitched_encoder_output.gaussians.means,
            ff_encoder_output.gaussians.means,
        )
        gaussian_covariance_loss = F.l1_loss(
            stitched_encoder_output.gaussians.covariances,
            ff_encoder_output.gaussians.covariances,
        )
        gaussian_harmonics_loss = F.l1_loss(
            stitched_encoder_output.gaussians.harmonics,
            ff_encoder_output.gaussians.harmonics,
        )
        gaussian_opacity_loss = F.l1_loss(
            stitched_encoder_output.gaussians.opacities,
            ff_encoder_output.gaussians.opacities,
        )
        gaussian_scales_loss = F.l1_loss(
            stitched_encoder_output.gaussians.scales,
            ff_encoder_output.gaussians.scales,
        )
        gaussian_rotations_loss = F.l1_loss(
            stitched_encoder_output.gaussians.rotations,
            ff_encoder_output.gaussians.rotations,
        )

        conf_loss = F.l1_loss(stitched_conf, ff_conf) * 0.01
        depth_conf_loss = F.l1_loss(stitched_depth_conf, ff_depth_conf) * 0.01
        anchor_feat_loss = F.l1_loss(stitched_anchor_feats, ff_anchor_feats)
        context_pose_extrinsic_loss = F.l1_loss(
            stitched_encoder_output.pred_context_pose["extrinsic"],
            ff_encoder_output.pred_context_pose["extrinsic"],
        )
        context_pose_intrinsic_loss = F.l1_loss(
            stitched_encoder_output.pred_context_pose["intrinsic"],
            ff_encoder_output.pred_context_pose["intrinsic"],
        )
        pred_pose_enc_list_loss = 0
        for i in range(len(stitched_encoder_output.pred_pose_enc_list)):
            pred_pose_enc_list_loss += F.l1_loss(
                stitched_encoder_output.pred_pose_enc_list[i],
                ff_encoder_output.pred_pose_enc_list[i],
            )
        pred_pose_enc_list_loss /= len(stitched_encoder_output.pred_pose_enc_list)
        loss = {}
        loss["depth_loss"] = depth_loss
        loss["depth_loss_grad"] = depth_loss_grad * 0.005
        loss["gaussian_mean_loss"] = gaussian_mean_loss
        loss["gaussian_covariance_loss"] = gaussian_covariance_loss
        loss["gaussian_harmonics_loss"] = gaussian_harmonics_loss
        loss["gaussian_opacity_loss"] = gaussian_opacity_loss
        loss["gaussian_scales_loss"] = gaussian_scales_loss * 10
        loss["gaussian_rotations_loss"] = gaussian_rotations_loss
        loss["conf_loss"] = conf_loss
        loss["depth_conf_loss"] = depth_conf_loss
        loss["anchor_feat_loss"] = anchor_feat_loss * 0.1
        loss["context_pose_extrinsic_loss"] = context_pose_extrinsic_loss
        loss["context_pose_intrinsic_loss"] = context_pose_intrinsic_loss
        loss["pred_pose_enc_list_loss"] = pred_pose_enc_list_loss
        loss["total_loss"] = sum(loss.values())
        return loss


class AnySplatStitched(AnySplat):
    def __init__(self, model: AnySplat, stitching_layer: str):
        super().__init__(model.encoder_cfg, model.decoder_cfg)
        self.load_state_dict(model.state_dict())
        self.stitching_layer = stitching_layer

        # "enc_blocks_k" -> stitch after k encoder blocks.
        self.stitched_layer_index = int(stitching_layer.split("_")[-1])

        # Convert the model to a stitched model
        self.convert_model_to_stitched_model()

        self.grad_checkpointing = True

    def convert_model_to_stitched_model(self):
        del self.encoder.aggregator.patch_embed.patch_embed
        for i in range(self.stitched_layer_index):
            logger.info(f"removing encoder block {i}")
            del self.encoder.aggregator.patch_embed.blocks[0]
        logger.info(
            f"encoder blocks after removing: length: {len(self.encoder.aggregator.patch_embed.blocks)}"
        )

    def forward(
        self, context_latent: torch.Tensor, context_image: torch.Tensor, train: bool
    ):
        B, c, S, h, w = context_image.shape
        b, lc, v, lh, lw = context_latent.shape

        # context_image is expected in [-1, 1]; AnySplat inference uses [0, 1].
        context_image = rearrange(context_image, "b c v h w -> b v c h w")
        context_image = (context_image + 1) / 2  # have to check

        """
        patch_embed forward
        """
        # Tokens: flatten spatial grid, prepend cls/register tokens, add pos enc.
        x = rearrange(context_latent, "b c v h w -> (b v) (h w) c")
        x = torch.cat(
            [
                self.encoder.aggregator.patch_embed.cls_token.expand(
                    x.shape[0], -1, -1
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.encoder.aggregator.patch_embed.interpolate_pos_encoding(x, w, h)
        if self.encoder.aggregator.patch_embed.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.encoder.aggregator.patch_embed.register_tokens.expand(
                        x.shape[0], -1, -1
                    ),
                    x[:, 1:],
                ),
                dim=1,
            )
        for blk in self.encoder.aggregator.patch_embed.blocks:
            if self.grad_checkpointing:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.encoder.aggregator.patch_embed.norm(x)
        x_norm_patchtokens = x[
            :, self.encoder.aggregator.patch_embed.num_register_tokens + 1 :
        ]
        patch_tokens = x_norm_patchtokens
        """
        Aggregator forward
        """
        _, P, C = patch_tokens.shape
        camera_token = slice_expand_and_flatten(
            self.encoder.aggregator.camera_token, B, S
        )
        register_token = slice_expand_and_flatten(
            self.encoder.aggregator.register_token, B, S
        )
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        pos = None

        if self.encoder.aggregator.rope is not None:
            pos = self.encoder.aggregator.position_getter(
                B * S,
                h // self.encoder.aggregator.patch_size,
                w // self.encoder.aggregator.patch_size,
                device=context_image.device,
            )
        if self.encoder.aggregator.patch_start_idx > 0:
            pos = pos + 1
            pos_special = (
                torch.zeros(B * S, self.encoder.aggregator.patch_start_idx, 2)
                .to(context_image.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape
        frame_idx = 0
        global_idx = 0
        output_list = []
        layer_idx = 0

        # Intermediate layers to tap for camera head; indices refer to AA depth.
        intermediate_layer_idx = [4, 11, 17, 23]  # To confirm once again
        if intermediate_layer_idx is not None:
            required_layers = set(intermediate_layer_idx)
            required_layers.add(self.encoder.aggregator.depth - 1)  # 23

        for _ in range(self.encoder.aggregator.aa_block_num):
            for attn_type in self.encoder.aggregator.aa_order:
                if attn_type == "frame":
                    if self.grad_checkpointing:
                        tokens, frame_idx, frame_intermediates = checkpoint(
                            self.encoder.aggregator._process_frame_attention,
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            frame_idx,
                            pos=pos,
                            use_reentrant=False,
                        )
                    else:
                        tokens, frame_idx, frame_intermediates = (
                            self.encoder.aggregator._process_frame_attention(
                                tokens, B, S, P, C, frame_idx, pos=pos
                            )
                        )
                        # tokens, frame_idx, frame_intermediates = (
                        #     self.encoder.aggregator._process_frame_attention(
                        #         tokens, B, S, P, C, frame_idx, pos=pos
                        #     )
                        # )
                elif attn_type == "global":
                    if self.grad_checkpointing:
                        tokens, global_idx, global_intermediates = checkpoint(
                            self.encoder.aggregator._process_global_attention,
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            global_idx,
                            pos=pos,
                            use_reentrant=False,
                        )
                    else:
                        tokens, global_idx, global_intermediates = (
                            self.encoder.aggregator._process_global_attention(
                                tokens, B, S, P, C, global_idx, pos=pos
                            )
                        )
                        # tokens, global_idx, global_intermediates = (
                        #     self.encoder.aggregator._process_global_attention(
                        #         tokens, B, S, P, C, global_idx, pos=pos
                        #     )
                        # )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if intermediate_layer_idx is not None:
                for i in range(len(frame_intermediates)):
                    # layer_idx advances by aa_block_size each inner step.
                    # current_layer corresponds to absolute depth in the AA stack.
                    current_layer = layer_idx + i
                    if current_layer in required_layers:
                        # concat frame and global intermediates, [B x S x P x 2C]
                        concat_inter = torch.cat(
                            [frame_intermediates[i], global_intermediates[i]], dim=-1
                        )
                        output_list.append(concat_inter)
                    layer_idx += self.encoder.aggregator.aa_block_size
            else:
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat(
                        [frame_intermediates[i], global_intermediates[i]], dim=-1
                    )
                    output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        aggregated_tokens_list, patch_start_idx = (
            output_list,
            self.encoder.aggregator.patch_start_idx,
        )
        # with torch.amp.autocast("cuda", enabled=False):
        with torch.amp.autocast("cuda", enabled=False):
            if self.grad_checkpointing:
                pred_pose_enc_list = checkpoint(
                    self.encoder.camera_head,
                    aggregated_tokens_list,
                    use_reentrant=False,
                )
            else:
                pred_pose_enc_list = self.encoder.camera_head(aggregated_tokens_list)

            last_pred_pose_enc = pred_pose_enc_list[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                last_pred_pose_enc,
                context_image.shape[-2:],  # TODO: confirm the resolution
            )  # only for debug

            if self.encoder.cfg.pred_head_type == "point":
                # Point head predicts 3D points directly from tokens.
                pts_all, pts_conf = self.encoder.point_head(
                    aggregated_tokens_list,
                    images=context_image,  # TODO: check the image resolution and range
                    patch_start_idx=patch_start_idx,
                )
            elif self.encoder.cfg.pred_head_type == "depth":
                # Depth head predicts depth maps then unprojects to 3D points.
                if self.grad_checkpointing:
                    depth_map, depth_conf = checkpoint(
                        self.encoder.depth_head,
                        aggregated_tokens_list,
                        images=context_image,  # TODO: check the image resolution and range
                        patch_start_idx=patch_start_idx,
                        use_reentrant=False,
                    )
                else:
                    depth_map, depth_conf = self.encoder.depth_head(
                        aggregated_tokens_list,
                        images=context_image,  # TODO: check the image resolution and range
                        patch_start_idx=patch_start_idx,
                    )
                pts_all = batchify_unproject_depth_map_to_point_map(
                    depth_map, extrinsic, intrinsic
                )
            else:
                raise ValueError(
                    f"pred_head_type is not valid, you can only choose from point {self.encoder.cfg.pred_head_type} is given"
                )
            if self.encoder.cfg.render_conf:
                conf_valid = torch.quantile(
                    depth_conf.flatten(0, 1), self.encoder.cfg.conf_threshold
                )
                conf_valid_mask = depth_conf > conf_valid
            else:
                conf_valid_mask = torch.ones_like(depth_conf, dtype=torch.bool)

            if self.grad_checkpointing:
                # gaussian_param_head outputs per-pixel Gaussian parameters + confidence.
                out = checkpoint(
                    self.encoder.gaussian_param_head,
                    aggregated_tokens_list,
                    pts_all.flatten(0, 1).permute(0, 3, 1, 2),
                    context_image,  # TODO: check the image resolution and range
                    patch_start_idx=patch_start_idx,
                    image_size=(h, w),
                    use_reentrant=False,
                )
            else:
                out = self.encoder.gaussian_param_head(
                    aggregated_tokens_list,
                    pts_all.flatten(0, 1).permute(0, 3, 1, 2),
                    context_image,  # TODO: check the image resolution and range
                    patch_start_idx=patch_start_idx,
                    image_size=(h, w),
                )
            del aggregated_tokens_list, patch_start_idx
            # torch.cuda.empty_cache()
            pts_flat = pts_all.flatten(2, 3)
            scene_scale = pts_flat.norm(dim=-1).mean().clip(min=1e-8)

            anchor_feats, conf = (
                out[:, :, : self.encoder.raw_gs_dim],
                out[:, :, self.encoder.raw_gs_dim],
            )

            neural_feats_list, neural_pts_list = [], []
            if self.encoder.cfg.voxelize:
                # Optional voxelization branch for stability; pads to max voxel count.
                print("voxelize is enabled")
                for b_i in range(b):
                    if self.grad_checkpointing:
                        neural_pts, neural_feats = checkpoint(
                            self.encoder.voxelizaton_with_fusion,
                            anchor_feats[b_i],
                            pts_all[b_i].permute(0, 3, 1, 2).contiguous(),
                            self.encoder.voxel_size,
                            conf[b_i],
                            use_reentrant=False,
                        )
                    else:
                        neural_pts, neural_feats = self.encoder.voxelizaton_with_fusion(
                            anchor_feats[b_i],
                            pts_all[b_i].permute(0, 3, 1, 2).contiguous(),
                            self.encoder.voxel_size,
                            conf=conf[b_i],
                        )
                    neural_feats_list.append(neural_feats)
                    neural_pts_list.append(neural_pts)
            else:
                for b_i in range(b):
                    neural_feats_list.append(
                        anchor_feats[b_i].permute(0, 2, 3, 1)[conf_valid_mask[b_i]]
                    )
                    neural_pts_list.append(pts_all[b_i][conf_valid_mask[b_i]])

            max_voxels = max(f.shape[0] for f in neural_feats_list)
            neural_feats = self.encoder.pad_tensor_list(
                neural_feats_list, (max_voxels,), value=-1e10
            )

            neural_pts = self.encoder.pad_tensor_list(
                neural_pts_list, (max_voxels,), -1e4
            )  # -1 == invalid voxel

            depths = neural_pts[..., -1].unsqueeze(-1)
            densities = neural_feats[..., 0].sigmoid()
            global_step = 0
            opacity = self.encoder.map_pdf_to_opacity(densities, global_step).squeeze(
                -1
            )
            if self.encoder.cfg.opacity_conf:
                shift = torch.quantile(depth_conf, self.encoder.cfg.conf_threshold)
                opacity = opacity * torch.sigmoid(depth_conf - shift)[
                    conf_valid_mask
                ].unsqueeze(0)  # little bit hacky

            gaussians = self.encoder.gaussian_adapter.forward(
                neural_pts,
                depths,
                opacity,
                neural_feats[..., 1:].squeeze(2),
            )
            extrinsic_padding = (
                torch.tensor(
                    [0, 0, 0, 1], device=context_image.device, dtype=extrinsic.dtype
                )
                .view(1, 1, 1, 4)
                .repeat(b, v, 1, 1)
            )
            # Normalize intrinsics by image width/height for downstream modules.
            intrinsic = intrinsic.clone()  # Create a new tensor
            intrinsic = torch.stack(
                [intrinsic[:, :, 0] / w, intrinsic[:, :, 1] / h, intrinsic[:, :, 2]],
                dim=2,
            )
            # gaussians
            # pred_pose_enc_list
            pred_context_pose = dict(
                # Convert to camera-to-world by inverting extrinsics.
                extrinsic=torch.cat([extrinsic, extrinsic_padding], dim=2).inverse(),
                intrinsic=intrinsic,
            )
            depth_dict = dict(depth=depth_map, conf_valid_mask=conf_valid_mask)
            infos = dict(
                scene_scale=scene_scale, voxelize_ratio=densities.shape[1] / (h * w * v)
            )
            distill_infos = None

            if train:
                return (
                    EncoderOutput(
                        gaussians=gaussians,
                        pred_pose_enc_list=pred_pose_enc_list,
                        pred_context_pose=pred_context_pose,
                        depth_dict=depth_dict,
                        infos=infos,
                        distill_infos=distill_infos,
                    ),
                    anchor_feats,
                    conf,
                    depth_conf,
                )

            else:
                return EncoderOutput(
                    gaussians=gaussians,
                    pred_pose_enc_list=pred_pose_enc_list,
                    pred_context_pose=pred_context_pose,
                    depth_dict=depth_dict,
                    infos=infos,
                    distill_infos=distill_infos,
                    last_pred_pose_enc=last_pred_pose_enc,
                )
