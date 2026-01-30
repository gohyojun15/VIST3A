      
from dataclasses import dataclass
from typing import Literal

import torch
import copy
from jaxtyping import Float, Int64
from torch import Tensor
import random
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerRankCfg:
    name: Literal["rank"]
    num_context_views: int # max number of context views
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    max_img_per_gpu: int


def rotation_angle(R1, R2):
    # R1 and R2 are 3x3 rotation matrices
    R = R1.T @ R2
    # Numerical stability: clamp values into [-1,1]
    val = (torch.trace(R) - 1) / 2
    val = torch.clamp(val, -1.0, 1.0)
    angle_rad = torch.acos(val)
    angle_deg = angle_rad * 180 / torch.pi  # Convert radians to degrees
    return angle_deg

def extrinsic_distance(extrinsic1, extrinsic2, lambda_t=1.0):
    R1, t1 = extrinsic1[:3, :3], extrinsic1[:3, 3]
    R2, t2 = extrinsic2[:3, :3], extrinsic2[:3, 3]
    rot_diff = rotation_angle(R1, R2) / 180
    
    center_diff = torch.norm(t1 - t2)
    return rot_diff + lambda_t * center_diff

def rotation_angle_batch(R1, R2):
    # R1, R2: shape (N, 3, 3)
    # We want a matrix of rotation angles for all pairs.
    # We'll get R1^T R2 for each pair.
    # Expand dimensions to broadcast: 
    # R1^T: (N,3,3) -> (N,1,3,3)
    # R2: (N,3,3) -> (1,N,3,3)
    R1_t = R1.transpose(-2, -1)[:, None, :, :]  # shape (N,1,3,3)
    R2_b = R2[None, :, :, :]                          # shape (1,N,3,3)
    R_mult = torch.matmul(R1_t, R2_b)  # shape (N,N,3,3)
    # trace(R) for each pair
    trace_vals = R_mult[..., 0, 0] + R_mult[..., 1, 1] + R_mult[..., 2, 2]  # (N,N)
    val = (trace_vals - 1) / 2
    val = torch.clamp(val, -1.0, 1.0)
    angle_rad = torch.acos(val)
    angle_deg = angle_rad * 180 / torch.pi
    return angle_deg / 180.0  # normalized rotation difference

def extrinsic_distance_batch(extrinsics, lambda_t=1.0):
    # extrinsics: (N,4,4)
    # Extract rotation and translation
    R = extrinsics[:, :3, :3]  # (N,3,3)
    t = extrinsics[:, :3, 3]   # (N,3)
    # Compute all pairwise rotation differences
    rot_diff = rotation_angle_batch(R, R)  # (N,N)
    # Compute all pairwise translation differences
    # For t, shape (N,3). We want all pair differences: t[i] - t[j].
    # t_i: (N,1,3), t_j: (1,N,3)
    t_i = t[:, None, :]  # (N,1,3)
    t_j = t[None, :, :]  # (1,N,3)
    trans_diff = torch.norm(t_i - t_j, dim=2)  # (N,N)
    dists = rot_diff + lambda_t * trans_diff
    return dists


def compute_ranking(extrinsics, lambda_t=1.0, normalize=True, batched=True):
    
    if normalize:
        extrinsics = copy.deepcopy(extrinsics)
        camera_center = copy.deepcopy(extrinsics[:, :3, 3])
        camera_center_scale = torch.norm(camera_center, dim=1)
        avg_scale = torch.mean(camera_center_scale)
        extrinsics[:, :3, 3] = extrinsics[:, :3, 3] / avg_scale

    
    if batched:
        dists = extrinsic_distance_batch(extrinsics, lambda_t=lambda_t)
    else:
        N = extrinsics.shape[0]
        dists = torch.zeros((N, N), device=extrinsics.device)
        for i in range(N):
            for j in range(N):
                dists[i,j] = extrinsic_distance(extrinsics[i], extrinsics[j], lambda_t=lambda_t)
    ranking = torch.argsort(dists, dim=1)
    return ranking, dists

# class ViewSamplerRank(ViewSampler[ViewSamplerRankCfg]):
#     def schedule(self, initial: int, final: int) -> int:
#         fraction = self.global_step / self.cfg.warm_up_steps
#         return min(initial + int((final - initial) * fraction), final)
    
#     def sample(
#         self,
#         scene: str,
#         extrinsics: Float[Tensor, "view 4 4"],
#         intrinsics: Float[Tensor, "view 3 3"],
#         device: torch.device = torch.device("cpu"),
#     ) -> tuple[
#         Int64[Tensor, " context_view"],  # indices for context views
#         Int64[Tensor, " target_view"],  # indices for target views
#         Float[Tensor, " overlap"],  # overlap
#     ]:
#         num_views, _, _ = extrinsics.shape
#         # breakpoint()
#         # Compute the context view spacing based on the current global step.
#         ranking, dists = compute_ranking(extrinsics, lambda_t=1.0, normalize=True, batched=True)
#         reference_view = random.sample(range(num_views), 1)[0]

#         refview_ranking = ranking[reference_view]
#         # if self.cfg.warm_up_steps > 0:
#         #     max_gap = self.schedule(
#         #         self.cfg.initial_max_distance_between_context_views,
#         #         self.cfg.max_distance_between_context_views,
#         #     )
#         #     min_gap = self.schedule(
#         #         self.cfg.initial_min_distance_between_context_views,
#         #         self.cfg.min_distance_between_context_views,
#         #     )
#         # else:
#         max_gap = self.cfg.max_distance_between_context_views
#         min_gap = self.cfg.min_distance_between_context_views

#         index_context_left = reference_view
#         rightmost_index = random.sample(range(min_gap, max_gap + 1), 1)[0] + 1
#         index_context_right = refview_ranking[rightmost_index].item()

#         middle_indices = refview_ranking[1: rightmost_index].tolist()
#         index_target = random.sample(middle_indices, self.num_target_views)
        
#         remaining_indices = [idx for idx in middle_indices if idx not in index_target]

#         # Sample extra context views if needed
#         extra_views = []
#         num_extra_views = self.num_context_views - 2  # subtract left and right context views
#         if num_extra_views > 0 and remaining_indices:
#             extra_views = random.sample(remaining_indices, min(num_extra_views, len(remaining_indices)))
#         else:
#             extra_views = []
        
#         overlap = torch.zeros(1) 

#         return (
#             torch.tensor((index_context_left, *extra_views, index_context_right)),
#             torch.tensor(index_target),
#             overlap
#         )
    

#     @property
#     def num_context_views(self) -> int:
#         return self.cfg.num_context_views

#     @property
#     def num_target_views(self) -> int:
#         return self.cfg.num_target_views
    

class ViewSamplerRank(ViewSampler[ViewSamplerRankCfg]):
    
    def sample(
        self,
        scene: str,
        num_context_views: int,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:  
        num_views, _, _ = extrinsics.shape
        # breakpoint()
        extrinsics = extrinsics.clone()
        # Compute the context view spacing based on the current global step.
        ranking, dists = compute_ranking(extrinsics, lambda_t=1.0, normalize=True, batched=True)
        reference_view = random.sample(range(num_views), 1)[0]
        
        refview_ranking = ranking[reference_view]
        # if self.cfg.warm_up_steps > 0:
        #     max_gap = self.schedule(
        #         self.cfg.initial_max_distance_between_context_views,
        #         self.cfg.max_distance_between_context_views,
        #     )
        #     min_gap = self.schedule(
        #         self.cfg.initial_min_distance_between_context_views,
        #         self.cfg.min_distance_between_context_views,
        #     )
        # else:
        min_gap, max_gap = self.num_ctxt_gap_mapping[num_context_views]

        # min_gap = self.cfg.min_distance_between_context_views
        # max_gap = self.cfg.max_distance_between_context_views
        
        max_gap = min(max_gap, num_views-1)
        # print(f"num_context_views: {num_context_views}, min_gap: {min_gap}, max_gap: {max_gap}")
        index_context_left = reference_view
        rightmost_index = random.sample(range(min_gap, max_gap + 1), 1)[0]
        
        # #! hard code for visualization
        # rightmost_index = self.cfg.max_distance_between_context_views

        index_context_right = refview_ranking[rightmost_index].item()
        
        middle_indices = refview_ranking[1: rightmost_index].tolist()
        index_target = random.sample(middle_indices, self.num_target_views)
        
        remaining_indices = [idx for idx in middle_indices if idx not in index_target]
        
        # Sample extra context views if needed
        extra_views = []
        num_extra_views = num_context_views - 2  # subtract left and right context views
        if num_extra_views > 0 and remaining_indices:
            extra_views = random.sample(remaining_indices, min(num_extra_views, len(remaining_indices)))
        else:
            extra_views = []
        
        overlap = torch.zeros(1) 

        return (
            torch.tensor((index_context_left, *extra_views, index_context_right)),
            torch.tensor(index_target),
            overlap
        )
    

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
    
    @property
    def num_ctxt_gap_mapping_target(self) -> dict:
        mapping = dict()
        for num_ctxt in range(2, self.cfg.num_context_views + 1):
            mapping[num_ctxt] = [max(num_ctxt * 2, self.cfg.num_target_views + num_ctxt), max(self.cfg.num_target_views + num_ctxt, min(num_ctxt ** 2, self.cfg.max_distance_between_context_views))]
        return mapping
    
    @property
    def num_ctxt_gap_mapping(self) -> dict:
        mapping = dict()
        for num_ctxt in range(2, self.cfg.num_context_views + 1):
            mapping[num_ctxt] = [min(num_ctxt * 3, self.cfg.min_distance_between_context_views), min(max(num_ctxt * 5, num_ctxt ** 2), self.cfg.max_distance_between_context_views)]
        return mapping

    
