import argparse

import torch.distributed as dist
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from data.dl3dv_datasets import DL3DVStitchingDataset, DL3DVTextPairedDataset
from data.prompt_dataset import TextPromptDataset
from data.scannet_unlabeled_dataset import ScannetUnlabeledDataaset


def _build_single_stitching_dataset(
    name: str,
    root_path: str,
    num_images_from_unit_scene: int,
    num_frames_per_unit_scene: int,
    image_resolution: int,
    feedforward_image_resolution: int,
    augmentation: bool = False,
):
    # Map dataset names to loader classes and expected directory layouts.
    # Add new datasets here and in the README.
    if name == "dl3dv":
        return DL3DVStitchingDataset(
            root_path=root_path,
            num_images_from_unit_scene=num_images_from_unit_scene,
            num_frames_per_unit_scene=num_frames_per_unit_scene,
            image_resolution=image_resolution,
            feedforward_image_resolution=feedforward_image_resolution,
            color_augmentation=augmentation,
        )
    elif name == "scannet":
        return ScannetUnlabeledDataaset(
            root_path=root_path,
            num_images_from_unit_scene=num_images_from_unit_scene,
            num_frames_per_unit_scene=num_frames_per_unit_scene,
            image_resolution=image_resolution,
            feedforward_image_resolution=feedforward_image_resolution,
            color_augmentation=augmentation,
        )
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def create_stitching_dataloader(
    datasets: list[tuple[str, str]],  # <── list of (name, root) pairs
    args: argparse.Namespace,
    augmentation: bool = False,
):
    logger.info("Creating dataloader")
    for n, r in datasets:
        logger.info(f"  Dataset: {n} at {r}")

    # build each dataset individually
    ds_objects = [
        _build_single_stitching_dataset(
            name=n,
            root_path=r,
            num_images_from_unit_scene=args.num_images_from_unit_scene,
            num_frames_per_unit_scene=args.num_frames_per_unit_scene,
            image_resolution=args.resolution,
            feedforward_image_resolution=args.feedforward_resolution,
            augmentation=augmentation,
        )
        for n, r in datasets
    ]

    train_set = ds_objects[0] if len(ds_objects) == 1 else ConcatDataset(ds_objects)
    sampler = DistributedSampler(
        train_set,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=22,
    )
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=3,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
    )

    logger.info(
        f"DataLoader has {len(train_set)} samples from "
        f"{len(ds_objects)} dataset(s); batch size {args.batch_size}, "
        f"resolution {args.resolution}×{args.resolution}"
    )

    return loader, sampler


def create_vdm_tuning_dataloader(
    datasets: list[tuple[str, str]], args: argparse.Namespace
):
    logger.info("Creating VDM tuning dataloader")
    for n, r in datasets:
        logger.info(f"  Dataset: {n} at {r}")

    for n, r in datasets:
        if n == "text":
            text_dataset = TextPromptDataset(r)
            text_dataloader = DataLoader(
                text_dataset,
                batch_size=1,
                sampler=None,
                shuffle=False,
                num_workers=1,
                collate_fn=TextPromptDataset.collate_fn,
            )
        elif n == "dl3dv":
            dl3dv_dataset = DL3DVTextPairedDataset(
                root_path=r,
                num_images_from_unit_scene=args.num_images_from_unit_scene,
                num_frames_per_unit_scene=args.num_frames_per_unit_scene,
                image_resolution=args.resolution,
                text_annotation_path="data/dl3dv_text_label_980P.json",
            )
            dl3dv_dataloader = DataLoader(
                dl3dv_dataset,
                batch_size=args.batch_size,
                sampler=None,
                shuffle=False,
                num_workers=3,
                pin_memory=False,
                persistent_workers=False,
                drop_last=True,
            )

    return text_dataloader, None, dl3dv_dataloader, None
