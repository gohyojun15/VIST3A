import json
import os
import os.path as osp
from typing import Iterable, Optional, Union

import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()

try:
    lanczos = Image.Resampling.LANCZOS
    bicubic = Image.Resampling.BICUBIC
except AttributeError:
    lanczos = Image.LANCZOS
    bicubic = Image.BICUBIC


def rescale_image_w_crop(
    image: Image.Image,
    intrinsic: np.ndarray,
    output_width_1: int,  # for vae image
    output_width_2: int,  # for feedforward image
    pixel_center: bool = True,
):
    """
    Rescale image, depth map, and intrinsic with crop.

    Image -> VAE image scale -> VAE image crop -> Feedforward image scale.
    """

    H, W = map(float, image.size)

    scale = output_width_1 / min(H, W)

    new_h = round(H * scale)
    new_w = round(W * scale)

    # Rescale for VAE input, then center-crop to a square.
    image_first = image.resize((new_w, new_h), resample=lanczos)

    # rescale intrinsic
    intrinsic = np.copy(intrinsic)
    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    intrinsic[:2, :] = intrinsic[:2, :] * scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    x0 = max((new_w - output_width_1) // 2, 0)
    y0 = max((new_h - output_width_1) // 2, 0)
    l, t, r, b = x0, y0, x0 + output_width_1, y0 + output_width_1  # noqa: E741

    image_first = image_first.crop((l, t, r, b))

    # camera matrix crop
    intrinsic = intrinsic.copy()
    intrinsic[0, 2] -= l
    intrinsic[1, 2] -= t

    # Feedforward image scale (can be different from VAE size).
    H, W = map(float, image_first.size)
    scale = output_width_2 / min(H, W)
    final_new_h = round(H * scale)
    final_new_w = round(W * scale)
    image_second = image_first.resize((final_new_w, final_new_h), resample=lanczos)
    intrinsic = intrinsic.copy()

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5
    intrinsic[:2, :] = intrinsic[:2, :] * scale
    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5
    return image_second, intrinsic, image_first


class Re10KNVSDataset(Dataset):
    def __init__(
        self,
        Re10K_DIR,
        split="test",
        load_img_size: int = 512,
        feedforward_img_size: int = 448,
        sort_by_filename=False,
        cache_file="evaluation/datasets/re10k_nvs_cache.npy",
        seq_file=None,
    ):
        self.Re10K_DIR = Re10K_DIR
        print(f"[Re10K-{split}] Re10K_DIR is {Re10K_DIR}")
        self.load_img_size = load_img_size
        self.feedforward_img_size = feedforward_img_size
        self.split = split

        if osp.exists(cache_file):
            # Cache stores pre-parsed intrinsics/extrinsics for faster startup.
            print(f"[Re10K-{split}] Loading from cache file: {cache_file}")
            self.metadata = np.load(cache_file, allow_pickle=True).item()
            self.sequence_list = sorted(list(self.metadata.keys()))

        elif split == "test":
            if seq_file is not None:
                with open(seq_file, "r") as f:
                    seq_list = f.readlines()
                self.sequence_list = [x.strip() for x in seq_list]
            else:
                self.sequence_list = os.listdir(Re10K_DIR)

            self.metadata = {}
            # Build metadata by reading per-sequence annotations.json.
            for seq in tqdm(
                self.sequence_list, desc=f"[Re10K-{split}] Creating metadata..."
            ):
                anno_path = osp.join(Re10K_DIR, seq, "annotations.json")
                try:
                    with open(anno_path, "r") as f:
                        annos = json.load(f)
                except Exception:
                    print(f"[Re10K-{split}] Failed to load {anno_path}")
                    continue

                seq_info = []
                for anno in annos:
                    seq_info.append(
                        {
                            "idx": anno["idx"],
                            "filepath": anno["filepath"],
                            "intrinsics": torch.tensor(anno["intrinsics"]),
                            "extrinsics": torch.tensor(anno["extrinsics"]),
                        }
                    )

                self.metadata[seq] = seq_info
            np.save(cache_file, self.metadata)
        elif split == "train":
            raise ValueError("We don't want to train on Re10K")
        else:
            raise ValueError("please specify correct set")

        self.sort_by_filename = sort_by_filename

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(
        self, index: Optional[int] = None, sequence_name: Optional[str] = None
    ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        return len(self.metadata[sequence_name])

    def __getitem__(self, idx_N):
        """Fetch item by index and a dynamic variable n_per_seq."""

        # Different from most datasets: we receive (sequence_index, n_per_seq)
        # to sample a variable number of frames per sequence.

        index, n_per_seq = idx_N
        sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]
        ids = np.random.choice(len(metadata), n_per_seq, replace=False)
        return self.get_data(index=index, ids=ids)

    def get_data(
        self,
        index: Optional[int] = None,
        sequence_name: Optional[str] = None,
        ids: Union[Iterable, None] = None,
    ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        metadata = self.metadata[sequence_name]

        if ids is None:
            ids = np.arange(len(metadata))
        annos = [metadata[i] for i in ids]
        if self.sort_by_filename:
            annos = sorted(annos, key=lambda x: x["filepath"])

        # Pre-allocate lists for images and camera params for consistent ordering.
        image_paths: list = [""] * len(annos)
        images: list = [0] * len(annos)
        vae_images: list = [0] * len(annos)
        extrinsics: torch.Tensor = torch.eye(4, 4)[None].repeat(len(annos), 1, 1)
        intrinsics: torch.Tensor = torch.eye(3, 3)[None].repeat(len(annos), 1, 1)

        for idx, anno in enumerate(annos):
            filepath = anno["filepath"]
            impath = osp.join(self.Re10K_DIR, filepath)
            rgb_image: Image.Image = Image.open(impath)

            rgb_image, intrinsic, vae_image = rescale_image_w_crop(
                image=rgb_image,
                intrinsic=anno["intrinsics"],
                output_width_1=self.load_img_size,
                output_width_2=self.feedforward_img_size,
            )

            image_paths[idx] = impath
            extrinsics[idx] = anno["extrinsics"]
            intrinsics[idx] = torch.from_numpy(intrinsic)
            images[idx] = to_tensor(rgb_image)
            vae_images[idx] = to_tensor(vae_image)

        batch = {"seq_id": sequence_name, "n": len(metadata), "ind": torch.tensor(ids)}
        batch["image_paths"] = image_paths
        batch["extrs"] = extrinsics
        batch["intrs"] = intrinsics
        batch["images"] = torch.stack(images, dim=0)
        batch["vae_images"] = torch.stack(vae_images, dim=0)

        return batch
