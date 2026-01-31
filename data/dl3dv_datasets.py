"""
DL3DV Dataset Dataloader

Expected Folder Structure:

<root_dir>/
├── 1K/
│   └── <scene_name>/
│       ├── images_4/
│       │   ├── <image_000.png>
│       │   ├── <image_001.png>
│       │   └── ...
│       └── transforms.json
├── 2K/
├── 3K/
│   ...
└── 11K/

- <root_dir>: Root directory containing different resolution folders (1K to 11K)
- <scene_name>: Specific scene folder under each resolution
- images_4/: Directory containing 8 images per scene
- transforms.json: Contains camera parameters and metadata for the scene
"""

import json
import os
import random
import re
import traceback
from glob import glob
from pathlib import Path

import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms

from data.image_preprocessing import load_image, resize_shorter_crop_square_batch


class DL3DVStitchingDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        num_images_from_unit_scene: int,
        num_frames_per_unit_scene: int,
        image_resolution: int = 512,
        feedforward_image_resolution: int = 448,
        color_augmentation: bool = True,
    ):
        super(DL3DVStitchingDataset, self).__init__()
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

        """crawl scene lists"""
        self.sequences = sorted(glob(str(root_path) + "/*/*/"))  # total 9286 sequences

    def __len__(self):
        return len(self.sequences)

    def get_scene_images(
        self,
        image_scene_path,
        scene_start_index,
        scene_end_index,
        image_index,
    ):
        all_images_paths = sorted(
            f
            for f in os.listdir(image_scene_path / "images_4")
            if re.match(r"^frame_\d+", f)
        )
        all_images_paths = all_images_paths[scene_start_index:scene_end_index]
        file_paths = [
            image_scene_path / "images_4" / f_info for f_info in all_images_paths
        ]

        # load image
        vae_imgs = []
        for i in image_index:
            vae_imgs.append(load_image(file_paths[i], self.transforms))
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
        return vae_tensor, feedforward_tensor

    def get_item_sub(self, idx):
        sequence_path = Path(self.sequences[idx])

        try:
            num_frames = len(
                sorted(
                    f
                    for f in os.listdir(sequence_path / "images_4")
                    if re.match(r"^frame_\d+", f)
                )
            )

            # Sampling strategy:
            # 1) Choose a random window length in [num_images_from_unit_scene, num_frames_per_unit_scene].
            # 2) Randomly choose a start index for that window.
            # 3) Randomly pick num_images_from_unit_scene frames within the window.
            frames_per_scene = random.randint(
                self.num_images_from_unit_scene, self.num_frames_per_unit_scene
            )
            # 2. randomly select a start index according to frames_per_scene
            start = random.randint(0, num_frames - frames_per_scene - 1)
            # 3. determine end index
            end = start + frames_per_scene
            # 4. randomly sample num_images_from_unit_scene indices from the selected frames_per_scene
            image_index = random.sample(
                range(1, frames_per_scene), self.num_images_from_unit_scene - 1
            )
            # 5. add the first frame index and sort
            image_index.append(0)
            image_index.sort()
            vae_image_tensor, feedforward_image_tensor = self.get_scene_images(
                sequence_path, start, end, image_index
            )

        except Exception as e:
            print(f"Error reading sequence at {sequence_path}: {e}")
            raise

        return dict(
            # remove batch dim( c t h w)
            vae_image_tensor=vae_image_tensor[0],
            feedforward_image_tensor=feedforward_image_tensor[0],
        )

    def __getitem__(self, idx):
        # since we found some sequences have issues, we try to load another sequence when error occurs
        try:
            return self.get_item_sub(idx)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            traceback.print_exc()
            while True:
                idx = random.randint(0, len(self.sequences) - 1)
                try:
                    return self.get_item_sub(idx)
                except Exception as e:
                    print(f"Error at index {idx}: {e}")
                    traceback.print_exc()
                    continue


class DL3DVTextPairedDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        num_images_from_unit_scene: int,
        num_frames_per_unit_scene: int = 32,
        image_resolution: int = 512,
        text_annotation_path: str = "data/dl3dv_text_label_980P.json",
    ) -> None:
        super(DL3DVTextPairedDataset, self).__init__()
        self.root_path = root_path
        self.text_annotation_path = text_annotation_path
        self.num_images_from_unit_scene = num_images_from_unit_scene
        self.num_frames_per_unit_scene = num_frames_per_unit_scene
        self.image_resolution = image_resolution
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        with open(self.text_annotation_path, "r") as f:
            dl3dv_list = json.load(f)
        self.dl3dv_scene_dict = {v["scene_name"]: v for v in dl3dv_list}
        self.sequences = sorted(glob(str(self.root_path) + "/*/*/"))

        # sequence filtering: only keep sequences that are in the annotation file
        filtered_sequences = []
        for seq in self.sequences:
            seq_name = os.path.basename(os.path.normpath(seq))
            if seq_name in self.dl3dv_scene_dict:
                filtered_sequences.append(seq)
        print(f"Total {len(filtered_sequences)} sequences found in {self.root_path}")
        self.sequences = filtered_sequences

    def __len__(self):
        return len(self.sequences)

    def get_item_sub(self, idx):
        sequence_path = Path(self.sequences[idx])
        sequence_name = os.path.basename(sequence_path)

        all_images = sorted(list((sequence_path / "images_4").glob("*.png")))

        scene_caption = self.dl3dv_scene_dict[sequence_name]["caption"]
        # num_available_caption = len(scene_caption)

        keys = list(scene_caption.keys())
        selected_caption_dict_key = random.choice(keys)
        selected_caption = scene_caption[selected_caption_dict_key]

        start_frame_index = selected_caption_dict_key.split("_")[-2]
        end_frame_index = selected_caption_dict_key.split("_")[-1]

        # check how many frames are available in the selected range
        all_images_in_range = []
        for img_path in all_images:
            img_index = int(re.findall(r"frame_(\d+)\.[^.]+$", img_path.name)[0])
            if int(start_frame_index) <= img_index <= int(end_frame_index):
                all_images_in_range.append(img_path)

        images = []
        for image_path_in_range in all_images_in_range:
            try:
                img = load_image(image_path_in_range, self.transforms)
            except Exception as e:
                print(f"Error loading image {image_path_in_range}: {e}")
                continue  # skip corrupted images
            img = transforms.Resize((self.image_resolution, self.image_resolution))(img)
            images.append(img)

        # if num_images_from_unit_scene > len(unit_frames), we just repeat the last frame
        if self.num_images_from_unit_scene >= len(images):
            image_index = list(range(len(images)))
            while len(image_index) < self.num_images_from_unit_scene:
                image_index.append(len(images) - 1)
        else:
            # randomly select indexs for num_images_from_unit_scene
            image_index = random.sample(
                range(1, len(images) - 1), self.num_images_from_unit_scene - 2
            )
            image_index.append(0)
            image_index.append(len(images) - 1)
            image_index.sort()

        images = [images[i] for i in image_index]

        image_tensor = torch.stack(images)
        image_tensor = (
            rearrange(image_tensor, "t c h w -> c t h w").unsqueeze(0).float() * 2 - 1
        )
        return dict(
            # remove batch dim( c t h w)
            image_tensor=image_tensor[0],
            caption=selected_caption,
        )

    def __getitem__(self, idx):
        # since we found some sequences have issues, we try to load another sequence when error occurs
        try:
            return self.get_item_sub(idx)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            traceback.print_exc()
            max_trys = 10
            trials = 0
            while True:
                idx = random.randint(0, len(self.sequences) - 1)
                try:
                    return self.get_item_sub(idx)
                except Exception as e:
                    print(f"Error at index {idx}: {e}")
                    traceback.print_exc()
                    trials += 1
                    if trials >= max_trys:
                        raise RuntimeError(
                            "Exceeded maximum retry attempts for data loading"
                        )
                    continue
