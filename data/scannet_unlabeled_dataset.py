"""
Dataset for unlabeled ScanNet dataset

Expected folder structure:
<root_dir>/
├── <sequence_1>/
│   ├── frames/
│   │   ├── color/
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   └── ...
"""

import json
import os
import random
from glob import glob

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms

from data.image_preprocessing import load_image, resize_shorter_crop_square_batch


class ScannetUnlabeledDataaset(Dataset):
    def __init__(
        self,
        root_path: str,
        num_images_from_unit_scene: int,
        num_frames_per_unit_scene: int,
        image_resolution: int = 512,
        feedforward_image_resolution: int = 448,
        color_augmentation: bool = True,
    ):
        super(ScannetUnlabeledDataaset, self).__init__()
        self.root_path = root_path
        self.num_images_from_unit_scene = num_images_from_unit_scene
        self.num_frames_per_unit_scene = num_frames_per_unit_scene
        self.image_resolution = image_resolution
        self.feedforward_image_resolution = feedforward_image_resolution
        self.color_augmentation = color_augmentation
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        if self.color_augmentation:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.1,
            )
        self.sequences = glob(str(root_path) + "/*/")  # total 10571 sequences
        with open("data/train_name_list.json") as f:
            train_name_list = json.load(f)
        print("total sequences", len(self.sequences))
        # filter only train sequences
        self.sequences = [
            seq for seq in self.sequences if seq.split("/")[-2] in train_name_list
        ]
        self.sequences = sorted(self.sequences)
        print("total train sequences", len(self.sequences))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_folder_path = os.path.join(self.sequences[idx], "frames", "color")
        image_file_names = os.listdir(image_folder_path)
        image_file_names = sorted(image_file_names, key=lambda x: int(x.split(".")[0]))
        num_frames = len(image_file_names)  # total frames in the sequence

        # Sample a random window, then pick a subset of frames for conditioning.
        frames_per_scene = random.randint(
            self.num_images_from_unit_scene, self.num_frames_per_unit_scene
        )
        # 2. random start index
        unit_scene_start_sampling = np.random.randint(0, num_frames - frames_per_scene)
        image_index = random.sample(
            range(1, frames_per_scene), self.num_images_from_unit_scene - 1
        )
        # 3. always include the first frame
        image_index.append(0)
        image_index.sort()
        image_index = [int(x) for x in image_index]
        image_index = np.array(image_index) + unit_scene_start_sampling
        image_index = image_index.tolist()

        # load image
        vae_imgs = []
        for i in image_index:
            image_path = os.path.join(image_folder_path, image_file_names[i])
            image = load_image(image_path, self.transforms)
            vae_imgs.append(image)
        vae_tensor = torch.stack(vae_imgs)
        # processing image (resize + color augmentation)
        vae_tensor = resize_shorter_crop_square_batch(
            vae_tensor, target_size=self.image_resolution
        )
        if self.color_augmentation:
            vae_tensor = self.color_jitter(vae_tensor)

        # We suppose that feedforward model and video vae have different image resolution
        feedforward_imgs = [
            transforms.Resize(
                (self.feedforward_image_resolution, self.feedforward_image_resolution)
            )(img)
            for img in vae_tensor
        ]
        # normalize to [-1, 1]
        vae_tensor = (
            rearrange(vae_tensor, "t c h w -> c t h w").unsqueeze(0).float() * 2 - 1
        )
        feedforward_tensor = (
            rearrange(torch.stack(feedforward_imgs), "t c h w -> c t h w")
            .unsqueeze(0)
            .float()
            * 2
            - 1
        )
        return dict(
            vae_image_tensor=vae_tensor[0],
            feedforward_image_tensor=feedforward_tensor[0],
        )
