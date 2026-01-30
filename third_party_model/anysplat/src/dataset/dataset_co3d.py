from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import random
from typing import Literal
import os
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import os.path as osp
import cv2
from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization

from .shims.geometry_shim import depthmap_to_absolute_camera_coordinates

CATEGORY = {'train': 
    ["backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave",
    "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign",
    "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv",
    "umbrella", "vase", "wineglass",], 
    'test': ['teddybear']}

@dataclass
class DatasetCo3dCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    normalize_by_pts3d: bool
    intr_augment: bool
    rescale_to_1cube: bool
    mask_bg: Literal['rand', True, False] = True
    
@dataclass
class DatasetCo3dCfgWrapper:
    co3d: DatasetCo3dCfg


class DatasetCo3d(Dataset):
    cfg: DatasetCo3dCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetCo3dCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.root = cfg.roots[0]
        self.mask_bg = cfg.mask_bg
        assert self.mask_bg in ('rand', True, False)

        # load all scenes
        self.categories = CATEGORY[self.data_stage]
        self.scene_seq_dict = {}
        self.scene_ids = []
        for category in self.categories:
            with open(osp.join(self.root, f"{category}/valid_seq.json"), "r") as f:
                scene_seq_dict = json.load(f)
                for scene, seqs in scene_seq_dict.items():
                    self.scene_seq_dict[f"{category}/{scene}"] = seqs
                    self.scene_ids.append(f"{category}/{scene}")

        print(f"CO3Dv2 {self.stage}: loaded {len(self.scene_seq_dict)} scenes")

    def load_frames(self, scene_id, frame_ids):
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Create a list to store futures with their original indices
            futures_with_idx = []
            for idx, frame_id in enumerate(frame_ids):
                file_path = os.path.join(self.root, f"{scene_id}/images/frame{frame_id:06d}.jpg")
                futures_with_idx.append(
                    (
                        idx,
                        executor.submit(
                            lambda p: self.to_tensor(Image.open(p).convert("RGB")),
                            file_path,
                        ),
                    )
                )

            # Pre-allocate list with correct size to maintain order
            torch_images = [None] * len(frame_ids)
            for idx, future in futures_with_idx:
                torch_images[idx] = future.result()
            # Check if all images have the same size
            sizes = set(img.shape for img in torch_images)
            if len(sizes) == 1:
                torch_images = torch.stack(torch_images)
        # Return as list if images have different sizes
        return torch_images

    def load_npz(self, scene_id, frame_id):
        npzpath = os.path.join(self.root, f"{scene_id}/images/frame{frame_id:06d}.npz")
        imgpath = os.path.join(self.root, f"{scene_id}/images/frame{frame_id:06d}.jpg")
        img = Image.open(imgpath)
        # breakpoint()
        W, H = img.size
        npzdata = np.load(npzpath)
        intri = npzdata['camera_intrinsics']
        extri = npzdata['camera_pose']
        intri[0, 0] /= float(W)
        intri[1, 1] /= float(H)
        intri[0, 2] /= float(W)
        intri[1, 2] /= float(H)
        md = npzdata['maximum_depth']
        return intri, extri, md

    def load_depth(self, scene_id, frame_ids, mds):
        torch_depths = []
        for frame_id in frame_ids:
            depthpath = os.path.join(self.root, f"{scene_id}/depths/frame{frame_id:06d}.jpg.geometric.png")
            depth = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)/65535*np.nan_to_num(mds[frame_id])
            depth = np.nan_to_num(depth)
            torch_depths.append(torch.from_numpy(depth))
        return torch_depths
    
    def load_masks(self, scene_id, frame_ids):
        masks = []
        for frame_id in frame_ids:
            maskpath = os.path.join(self.root, f"{scene_id}/masks/frame{frame_id:06d}.png")
            maskmap = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
            maskmap = (maskmap / 255.0) > 0.1
            masks.append(torch.from_numpy(maskmap))
        return masks

    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        scene_id = self.scene_ids[index]
        seq = self.scene_seq_dict[scene_id]

        extrinsics = []
        intrinsics = []
        frame_ids = []
        mds = {}
        for frame_id in seq:
            intri, extri, md = self.load_npz(scene_id, frame_id)
            extrinsics.append(extri)
            intrinsics.append(intri)
            frame_ids.append(frame_id)
            mds[frame_id] = md

        extrinsics = np.array(extrinsics)
        intrinsics = np.array(intrinsics)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
        
        num_views = extrinsics.shape[0]
        context_indices = torch.tensor(random.sample(range(num_views), num_context_views))
        remaining_indices = torch.tensor([i for i in range(num_views) if i not in context_indices])
        target_indices = torch.tensor(random.sample(remaining_indices.tolist(), self.view_sampler.num_target_views))

        # Skip the example if the field of view is too wide.
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")

        input_frames = [frame_ids[i] for i in context_indices]
        target_frame = [frame_ids[i] for i in target_indices]

        context_images = self.load_frames(scene_id, input_frames)
        target_images = self.load_frames(scene_id, target_frame)
        context_depths = self.load_depth(scene_id, input_frames, mds)
        target_depths = self.load_depth(scene_id, target_frame, mds)

        mask_bg = (self.mask_bg == True) or (self.mask_bg == "rand" and np.random.random() < 0.5)
        if mask_bg:
            context_masks = self.load_masks(scene_id, input_frames)
            target_mask = self.load_masks(scene_id, target_frame)

            # update the depthmap with mask
            context_depths = [depth * mask for depth, mask in zip(context_depths, context_masks)]
            target_depths = [depth * mask for depth, mask in zip(target_depths, target_mask)]


        # Resize the world to make the baseline 1.
        context_extrinsics = extrinsics[context_indices]
        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                print(
                    f"Skipped {scene_id} because of baseline out of range: "
                    f"{scale:.6f}"
                )
                raise Exception("baseline out of range")
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        # self.cfg.rescale_to_1cube = True
        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(torch.abs(extrinsics[context_indices][:, :3, 3])) # target pose is not included
            # all_extrinsics = torch.cat([extrinsics[context_indices], extrinsics[target_indices]], dim=0) # [N, 4, 4]
            # scene_scale = torch.max(torch.abs(all_extrinsics[:, :3, 3]))
            rescale_factor = 1 * scene_scale
            extrinsics[:, :3, 3] /= rescale_factor

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "depth": context_depths,
                "near": self.get_bound("near", len(context_indices)),
                "far": self.get_bound("far", len(context_indices)),
                "index": context_indices,
                # "overlap": overlap,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "depth": target_depths,
                "near": self.get_bound("near", len(target_indices)),
                "far": self.get_bound("far", len(target_indices)),
                "index": target_indices,
            },
            "scene": f"CO3Dv2 {scene_id}",
        }

        if self.stage == "train" and self.cfg.intr_augment:
            intr_aug = True
        else:
            intr_aug = False

        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14), intr_aug=intr_aug)
        
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        # example_1 = copy.deepcopy(example)
        # world pts
        image_size = example["context"]["image"].shape[2:]
        context_intrinsics = example["context"]["intrinsics"].clone().detach().numpy()
        context_intrinsics[:, 0] = context_intrinsics[:, 0] * image_size[1]
        context_intrinsics[:, 1] = context_intrinsics[:, 1] * image_size[0]

        target_intrinsics = example["target"]["intrinsics"].clone().detach().numpy()
        target_intrinsics[:, 0] = target_intrinsics[:, 0] * image_size[1]
        target_intrinsics[:, 1] = target_intrinsics[:, 1] * image_size[0]

        context_pts3d_list, context_valid_mask_list = [], []    
        target_pts3d_list, target_valid_mask_list = [], []

        for i in range(len(example["context"]["depth"])):
            context_pts3d, context_valid_mask = depthmap_to_absolute_camera_coordinates(example["context"]["depth"][i].numpy(), context_intrinsics[i], example["context"]["extrinsics"][i].numpy())
            context_pts3d_list.append(torch.from_numpy(context_pts3d).to(torch.float32))
            context_valid_mask_list.append(torch.from_numpy(context_valid_mask))

        context_pts3d = torch.stack(context_pts3d_list, dim=0)
        context_valid_mask = torch.stack(context_valid_mask_list, dim=0)

        for i in range(len(example["target"]["depth"])):
            target_pts3d, target_valid_mask = depthmap_to_absolute_camera_coordinates(example["target"]["depth"][i].numpy(), target_intrinsics[i], example["target"]["extrinsics"][i].numpy())
            target_pts3d_list.append(torch.from_numpy(target_pts3d).to(torch.float32))
            target_valid_mask_list.append(torch.from_numpy(target_valid_mask))

        target_pts3d = torch.stack(target_pts3d_list, dim=0)
        target_valid_mask = torch.stack(target_valid_mask_list, dim=0)

        # normalize by context pts3d
        if self.cfg.normalize_by_pts3d:
            transformed_pts3d = context_pts3d[context_valid_mask]
            scene_factor = transformed_pts3d.norm(dim=-1).mean().clip(min=1e-8)
            
            context_pts3d /= scene_factor
            example["context"]["depth"] /= scene_factor
            example["context"]["extrinsics"][:, :3, 3] /= scene_factor
            
            target_pts3d /= scene_factor
            example["target"]["depth"] /= scene_factor
            example["target"]["extrinsics"][:, :3, 3] /= scene_factor

        example["context"]["pts3d"] = context_pts3d
        example["target"]["pts3d"] = target_pts3d
        example["context"]["valid_mask"] = context_valid_mask
        example["target"]["valid_mask"] = target_valid_mask

        if torch.isnan(example["context"]["depth"]).any() or torch.isinf(example["context"]["depth"]).any() or \
            torch.isnan(example["context"]["extrinsics"]).any() or torch.isinf(example["context"]["extrinsics"]).any() or \
            torch.isnan(example["context"]["pts3d"]).any() or torch.isinf(example["context"]["pts3d"]).any() or \
            torch.isnan(example["context"]["intrinsics"]).any() or torch.isinf(example["context"]["intrinsics"]).any() or \
            torch.isnan(example["target"]["depth"]).any() or torch.isinf(example["target"]["depth"]).any() or \
            torch.isnan(example["target"]["extrinsics"]).any() or torch.isinf(example["target"]["extrinsics"]).any() or \
            torch.isnan(example["target"]["pts3d"]).any() or torch.isinf(example["target"]["pts3d"]).any() or \
            torch.isnan(example["target"]["intrinsics"]).any() or torch.isinf(example["target"]["intrinsics"]).any():
            raise Exception("encounter nan or inf in context depth")

        for key in ["context", "target"]:
            example[key]["valid_mask"] = (torch.ones_like(example[key]["valid_mask"]) * -1).type(torch.int32)

        return example


    def __getitem__(self, index_tuple: tuple) -> dict:
        index, num_context_views, patchsize_h = index_tuple
        patchsize_w = (self.cfg.input_image_shape[1] // 14)
        try:
            return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
        except Exception as e:
            print(f"Error: {e}")
            index = np.random.randint(len(self))
            return self.__getitem__((index, num_context_views, patchsize_h))

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return len(self.scene_ids)