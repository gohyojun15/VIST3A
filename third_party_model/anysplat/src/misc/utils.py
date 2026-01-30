import torch

from third_party_model.anysplat.src.visualization.color_map import (
    apply_color_map_to_image,
)
import torch.distributed as dist


def inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(
        -1, 1, 1
    )
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.mul(std).add(mean)


# Color-map the result.
def vis_depth_map(result, near=None, far=None):
    if near is None and far is None:
        far = result.view(-1)[:16_000_000].quantile(0.99).log()
        try:
            near = result[result > 0][:16_000_000].quantile(0.01).log()
        except:
            print("No valid depth values found.")
            near = torch.zeros_like(far)
    else:
        near = near.log()
        far = far.log()

    result = result.log()
    result = 1 - (result - near) / (far - near)
    return apply_color_map_to_image(result, "turbo")


def confidence_map(result):
    # far = result.view(-1)[:16_000_000].quantile(0.99).log()
    # try:
    #     near = result[result > 0][:16_000_000].quantile(0.01).log()
    # except:
    #     print("No valid depth values found.")
    #     near = torch.zeros_like(far)
    # result = result.log()
    # result = 1 - (result - near) / (far - near)
    result = result / result.view(-1).max()
    return apply_color_map_to_image(result, "magma")


def get_overlap_tag(overlap):
    if 0.05 <= overlap <= 0.3:
        overlap_tag = "small"
    elif overlap <= 0.55:
        overlap_tag = "medium"
    elif overlap <= 0.8:
        overlap_tag = "large"
    else:
        overlap_tag = "ignore"

    return overlap_tag


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
