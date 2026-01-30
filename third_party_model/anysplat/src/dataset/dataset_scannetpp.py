import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
import random
from typing import Literal
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
import torchvision
from torch import Tensor
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from .shims.geometry_shim import depthmap_to_absolute_camera_coordinates

from .shims.load_shim import imread_cv2

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization

@dataclass
class DatasetScannetppCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    metric_thre: float
    intr_augment: bool
    make_baseline_1: bool
    rescale_to_1cube: bool
    normalize_by_pts3d: bool

@dataclass
class DatasetScannetppCfgWrapper:
    scannetpp: DatasetScannetppCfg


class DatasetScannetpp(Dataset):
    cfg: DatasetScannetppCfgWrapper
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0
    
    def __init__(
        self,
        cfg: DatasetScannetppCfgWrapper,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # load data
        self.data_root = cfg.roots[0]
        self.data_list = [] # we use dslr rather than iphone
        data_index = os.listdir(f"{self.data_root}") # we train all the scenes

        if self.stage != "train":
            with open(f"{self.data_root}/valid.json", "r") as file:
                data_index = json.load(file)[:10]
                data_index = data_index * 100
                random.shuffle(data_index)
        else:
            with open(f"{self.data_root}/valid.json", "r") as file:
                data_index = json.load(file)[10:]

        self.data_list = [
            os.path.join(self.data_root, item) for item in data_index
        ]
        
        self.scene_ids = {}
        self.scenes = {}
        index = 0
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(self.load_metadata, scene_path) for scene_path in self.data_list]
            for future in as_completed(futures):
                scene_frames, scene_id = future.result()
                self.scenes[scene_id] = scene_frames
                self.scene_ids[index] = scene_id
                index += 1
        
        # if self.stage != "train":
        #     self.scene_ids = self.scene_ids 
        #     random.shuffle(self.scene_ids)
        print(f"Scannetpp: {self.stage}: loaded {len(self.scene_ids)} scenes")
        
    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def load_metadata(self, scene_path):
        metadata_path = os.path.join(scene_path, "scene_metadata.npz")
        metadata = np.load(metadata_path, allow_pickle=True)
        intrinsics = metadata["intrinsics"]
        trajectories = metadata["trajectories"]
        images = metadata["images"]

        scene_id = scene_path.split("/")[-1].split(".")[0]
        scene_frames = [
            {
                "file_path": os.path.join(scene_path, "images", images[i].split(".")[0] + ".jpg"),
                "depth_path": os.path.join(scene_path, "depth", images[i].split(".")[0] + ".png"),
                "intrinsics": self.convert_intrinsics(intrinsics[i]),
                "extrinsics": trajectories[i],
            }
            for i in range(len(images))
        ]
        scene_frames.sort(key=lambda x: x["file_path"])  # sort by file path to ensure correct order
        return scene_frames, scene_id

    def convert_intrinsics(self, intrinsics):
        w = intrinsics[0, 2] * 2
        h = intrinsics[1, 2] * 2
        intrinsics[0, 0] = intrinsics[0, 0] / w
        intrinsics[1, 1] = intrinsics[1, 1] / h
        intrinsics[0, 2] = intrinsics[0, 2] / w
        intrinsics[1, 2] = intrinsics[1, 2] / h
        return intrinsics
        
    def blender2opencv_c2w(self, pose):
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        opencv_c2w = np.array(pose) @ blender2opencv
        return opencv_c2w.tolist()

    def load_frames(self, frames):
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Create a list to store futures with their original indices
            futures_with_idx = []
            for idx, file_path in enumerate(frames):
                file_path = file_path["file_path"]
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
            torch_images = [None] * len(frames)
            for idx, future in futures_with_idx:
                torch_images[idx] = future.result()
            # Check if all images have the same size
            sizes = set(img.shape for img in torch_images)
            if len(sizes) == 1:
                torch_images = torch.stack(torch_images)
        # Return as list if images have different sizes
        return torch_images
    
    def load_depths(self, frames):
        torch_depths = []
        for idx, frame in enumerate(frames):
            depthmap = imread_cv2(frame["depth_path"], cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0 
            torch_depths.append(torch.from_numpy(depthmap))
        return torch.stack(torch_depths) # [N, H, W]


    def getitem(self, index: int, num_context_views: int, patchsize: tuple) -> dict:
        # import time
        # start_time = time.time()

        scene = self.scene_ids[index]
        example = self.scenes[scene]
        # load poses
        extrinsics = []
        intrinsics = []
        for frame in example:
            extrinsic = frame["extrinsics"]
            intrinsic = frame["intrinsics"]
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
        
        extrinsics = np.array(extrinsics)
        intrinsics = np.array(intrinsics)
        extrinsics = torch.tensor(extrinsics, dtype=torch.float32)
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32)

        try:
            context_indices, target_indices, overlap = self.view_sampler.sample(
                "scannetpp_"+scene,
                num_context_views,
                extrinsics,
                intrinsics,
            )
        except ValueError:
            # Skip because the example doesn't have enough frames.
            raise Exception("Not enough frames")
        
        # Skip the example if the field of view is too wide.
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            raise Exception("Field of view too wide")

        # Load the images.
        input_frames = [example[i] for i in context_indices]
        target_frame = [example[i] for i in target_indices]
        
        context_images = self.load_frames(input_frames)
        target_images = self.load_frames(target_frame)

        context_depths = self.load_depths(input_frames)
        target_depths = self.load_depths(target_frame)
        
        # Skip the example if the images don't have the right shape.
        context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
        target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
            print(
                f"Skipped bad example {example['key']}. Context shape was "
                f"{context_images.shape} and target shape was "
                f"{target_images.shape}."
            )
            raise Exception("Bad example image shape")
        
        context_extrinsics = extrinsics[context_indices]

        if self.cfg.make_baseline_1:
            a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
            scale = (a - b).norm()
            if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                print(
                    f"Skipped {scene} because of baseline out of range: "
                    f"{scale:.6f}"
                )
                raise Exception("baseline out of range")
            extrinsics[:, :3, 3] /= scale
        else:
            scale = 1

        if self.cfg.relative_pose:
            extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

        if self.cfg.rescale_to_1cube:
            scene_scale = torch.max(torch.abs(extrinsics[context_indices][:, :3, 3])) # target pose is not included
            rescale_factor = 1 * scene_scale
            extrinsics[:, :3, 3] /= rescale_factor

        if torch.isnan(extrinsics).any() or torch.isinf(extrinsics).any():
            raise Exception("encounter nan or inf in input poses")

        example = {
            "context": {
                "extrinsics": extrinsics[context_indices],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "depth": context_depths,
                "near": self.get_bound("near", len(context_indices)) / scale,
                "far": self.get_bound("far", len(context_indices)) / scale,
                "index": context_indices,
                "overlap": overlap,
            },
            "target": {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "depth": target_depths,
                "near": self.get_bound("near", len(target_indices)) / scale,
                "far": self.get_bound("far", len(target_indices)) / scale,
                "index": target_indices,
            },
            "scene": f"Scannetpp {scene}",
        }
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)

        if self.stage == "train" and self.cfg.intr_augment:
            intr_aug = True
        else:
            intr_aug = False

        example = apply_crop_shim(example, (patchsize[0] * 14, patchsize[1] * 14), intr_aug=intr_aug)
        
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
            torch.isnan(example["context"]["intrinsics"]).any() or torch.isinf(example["context"]["intrinsics"]).any() or \
            torch.isnan(example["target"]["depth"]).any() or torch.isinf(example["target"]["depth"]).any() or \
            torch.isnan(example["target"]["extrinsics"]).any() or torch.isinf(example["target"]["extrinsics"]).any() or \
            torch.isnan(example["target"]["intrinsics"]).any() or torch.isinf(example["target"]["intrinsics"]).any():
            raise Exception("encounter nan or inf in context depth")
        
        for key in ["context", "target"]:
            example[key]["valid_mask"] = (torch.ones_like(example[key]["valid_mask"]) * -1).type(torch.int32)
        
        return example
    
        
    def __getitem__(self, index_tuple: tuple) -> dict:
        index, num_context_views, patchsize_h = index_tuple
        # generate a random patch size
        patchsize_w = (self.cfg.input_image_shape[1] // 14)
        try:
            return self.getitem(index, num_context_views, (patchsize_h, patchsize_w))
        except Exception as e:
            print(f"Error: {e}")
            index = np.random.randint(len(self))
            return self.__getitem__((index, num_context_views, patchsize_h))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        
        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

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
        return len(self.data_list)