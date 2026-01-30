import torch
import torch.nn as nn
from copy import copy, deepcopy

from third_party_model.anysplat.src.model.encoder.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from third_party_model.anysplat.src.model.encoder.vggt.utils.rotation import mat_to_quat


def extri_intri_to_pose_encoding(
    extrinsics,
    intrinsics,
    image_size_hw=None,  # e.g., (256, 512)
    pose_encoding_type="absT_quaR_FoV",
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        intrinsics (torch.Tensor): Camera intrinsic parameters with shape BxSx3x3.
            Defined in pixels, with format:
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            where fx, fy are focal lengths and (cx, cy) is the principal point
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        # Note the order of h and w here
        # H, W = image_size_hw
        # fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        # fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        fov_h = 2 * torch.atan(0.5 / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan(0.5 / intrinsics[..., 0, 0])
        pose_encoding = torch.cat(
            [T, quat, fov_h[..., None], fov_w[..., None]], dim=-1
        ).float()
    else:
        raise NotImplementedError

    return pose_encoding


def huber_loss(x, y, delta=1.0):
    """Calculate element-wise Huber loss between x and y"""
    diff = x - y
    abs_diff = diff.abs()
    flag = (abs_diff <= delta).to(diff.dtype)
    return flag * 0.5 * diff**2 + (1 - flag) * delta * (abs_diff - 0.5 * delta)


class HuberLoss(nn.Module):
    def __init__(
        self, alpha=1.0, delta=1.0, gamma=0.6, weight_T=1.0, weight_R=1.0, weight_fl=0.5
    ):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.weight_T = weight_T
        self.weight_R = weight_R
        self.weight_fl = weight_fl

    def camera_loss_single(self, cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
        if loss_type == "l1":
            loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
            loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
            loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
        elif loss_type == "l2":
            loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(
                dim=-1, keepdim=True
            )
            loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(
                dim=-1
            )
            loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(
                dim=-1
            )
        elif loss_type == "huber":
            loss_T = huber_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3])
            loss_R = huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7])
            loss_fl = huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss_T = torch.nan_to_num(loss_T, nan=0.0, posinf=0.0, neginf=0.0)
        loss_R = torch.nan_to_num(loss_R, nan=0.0, posinf=0.0, neginf=0.0)
        loss_fl = torch.nan_to_num(loss_fl, nan=0.0, posinf=0.0, neginf=0.0)

        loss_T = torch.clamp(loss_T, min=-100, max=100)
        loss_R = torch.clamp(loss_R, min=-100, max=100)
        loss_fl = torch.clamp(loss_fl, min=-100, max=100)

        loss_T = loss_T.mean()
        loss_R = loss_R.mean()
        loss_fl = loss_fl.mean()

        return loss_T, loss_R, loss_fl

    def forward(self, pred_pose_enc_list, batch):
        context_extrinsics = batch["context"]["extrinsics"]
        context_intrinsics = batch["context"]["intrinsics"]
        image_size_hw = batch["context"]["image"].shape[-2:]

        # transform extrinsics and intrinsics to pose_enc
        GT_pose_enc = extri_intri_to_pose_encoding(
            context_extrinsics, context_intrinsics, image_size_hw
        )
        num_predictions = len(pred_pose_enc_list)
        loss_T = loss_R = loss_fl = 0

        for i in range(num_predictions):
            i_weight = self.gamma ** (num_predictions - i - 1)

            cur_pred_pose_enc = pred_pose_enc_list[i]

            loss_T_i, loss_R_i, loss_fl_i = self.camera_loss_single(
                cur_pred_pose_enc.clone(), GT_pose_enc.clone(), loss_type="huber"
            )
            loss_T += i_weight * loss_T_i
            loss_R += i_weight * loss_R_i
            loss_fl += i_weight * loss_fl_i

        loss_T = loss_T / num_predictions
        loss_R = loss_R / num_predictions
        loss_fl = loss_fl / num_predictions
        loss_camera = (
            loss_T * self.weight_T + loss_R * self.weight_R + loss_fl * self.weight_fl
        )

        loss_dict = {
            "loss_camera": loss_camera,
            "loss_T": loss_T,
            "loss_R": loss_R,
            "loss_fl": loss_fl,
        }

        # with torch.no_grad():
        #     # compute auc
        #     last_pred_pose_enc = pred_pose_enc_list[-1]

        #     last_pred_extrinsic, _ = pose_encoding_to_extri_intri(last_pred_pose_enc.detach(), image_size_hw, pose_encoding_type='absT_quaR_FoV', build_intrinsics=False)

        #     rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(last_pred_extrinsic.float(), context_extrinsics.float(), context_extrinsics.device)

        #     if rel_rangle_deg.numel() == 0 and rel_tangle_deg.numel() == 0:
        #         rel_rangle_deg = torch.FloatTensor([0]).to(context_extrinsics.device).to(context_extrinsics.dtype)
        #         rel_tangle_deg = torch.FloatTensor([0]).to(context_extrinsics.device).to(context_extrinsics.dtype)

        #     thresholds = [5, 15]
        #     for threshold in thresholds:
        #         loss_dict[f"Rac_{threshold}"] = (rel_rangle_deg < threshold).float().mean()
        #         loss_dict[f"Tac_{threshold}"] = (rel_tangle_deg < threshold).float().mean()

        #     _, normalized_histogram = calculate_auc(
        #         rel_rangle_deg, rel_tangle_deg, max_threshold=30, return_list=True
        #     )

        #     auc_thresholds = [30, 10, 5, 3]
        #     for auc_threshold in auc_thresholds:
        #         cur_auc = torch.cumsum(
        #             normalized_histogram[:auc_threshold], dim=0
        #         ).mean()
        #         loss_dict[f"Auc_{auc_threshold}"] = cur_auc

        return loss_dict
