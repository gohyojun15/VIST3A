# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Random sampling under a constraint
# --------------------------------------------------------
import numpy as np
import torch
from typing import Callable, Iterable, Optional
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler, BatchSampler
import random

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable batch sizes
    
    Args:
        batch: A list where each element could be either:
            - A single tuple (idx, num_images, ...)
            - A list of tuples [(idx1, num_images1, ...), (idx2, num_images2, ...)]
    """
    # If batch contains lists (variable batch size case)
    breakpoint()
    if isinstance(batch[0], list):
        # Flatten the batch
        flattened = []
        for item in batch:
            flattened.extend(item)
        batch = flattened
    
    # Now batch is a list of tuples, process normally
    return torch.utils.data.default_collate(batch)

class BatchedRandomSampler:
    """Random sampling under a constraint: each sample in the batch has the same feature,
    which is chosen randomly from a known pool of 'features' for each batch.

    For instance, the 'feature' could be the image aspect-ratio.

    The index returned is a tuple (sample_idx, feat_idx).
    This sampler ensures that each series of `batch_size` indices has the same `feat_idx`.
    """

    def __init__(
        self, dataset, batch_size, num_context_views, min_patch_num=20, max_patch_num=32, world_size=1, rank=0, drop_last=True
    ):
        self.batch_size = batch_size
        self.num_context_views = num_context_views

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size * world_size) if drop_last else N
        self.min_patch_num = min_patch_num
        self.max_patch_num = max_patch_num
        assert (
            world_size == 1 or drop_last
        ), "must drop the last batch in distributed mode"

        # distributed sampler
        self.world_size = world_size
        self.rank = rank
        self.epoch = None

    def __len__(self):


        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert (
                self.world_size == 1 and self.rank == 0
            ), "use set_epoch() if distributed mode is used"
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)
        
        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)
        
        # random feat_idxs (same across each batch)
        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
        num_imgs = rng.integers(low=2, high=self.num_context_views, size=n_batches)
        # num_imgs = (np.ones(n_batches) * self.num_context_views).astype(np.int64) # same number of context views for each batch
        num_imgs = np.broadcast_to(num_imgs[:, None], (n_batches, self.batch_size))
        num_imgs = num_imgs.ravel()[: self.total_size]

        # put them together
        idxs = np.c_[sample_idxs, num_imgs]  # shape = (total_size, 2)

        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * (
            (self.total_size + self.world_size * self.batch_size - 1)
            // (self.world_size * self.batch_size)
        )
        idxs = idxs[self.rank * size_per_proc : (self.rank + 1) * size_per_proc]

        yield from (tuple(idx) for idx in idxs)

class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 image_num_range,
                 h_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.image_num_range = image_num_range
        self.h_range = h_range
        self.rng = random.Random()
        
        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: float(num_images**2) for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])
        
        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number and aspect ratio
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_ps_h = np.random.randint(low=(self.h_range[0] // 14), high=(self.h_range[1] // 14)+1)

                # Update sampler parameters
                self.sampler.update_parameters(
                    image_num=random_image_num,
                    ps_h=random_ps_h
                )
                
                # Calculate batch size based on max images per GPU and current image number
                batch_size = self.max_img_per_gpu / random_image_num
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.image_num = None
        self.ps_h = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num, self.ps_h, )

    def update_parameters(self, image_num, ps_h):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.image_num = image_num
        self.ps_h = ps_h

class MixedBatchSampler(BatchSampler):
    """Sample one batch from a selected dataset with given probability.
    Compatible with datasets at different resolution
    """

    def __init__(
        self, src_dataset_ls, batch_size, num_context_views, world_size=1, rank=0, prob=None, sampler=None, generator=None
    ):
        self.base_sampler = None
        self.batch_size = batch_size
        self.num_context_views = num_context_views
        self.world_size = world_size
        self.rank = rank
        self.drop_last = True
        self.generator = generator

        self.src_dataset_ls = src_dataset_ls
        self.n_dataset = len(self.src_dataset_ls)
        
        # Dataset length
        self.dataset_length = [len(ds) for ds in self.src_dataset_ls]
        self.cum_dataset_length = [
            sum(self.dataset_length[:i]) for i in range(self.n_dataset)
        ]  # cumulative dataset length
        
        # BatchSamplers for each source dataset
        self.src_batch_samplers = []
        for ds in self.src_dataset_ls:
            sampler = DynamicDistributedSampler(ds, num_replicas=self.world_size, rank=self.rank, seed=42, shuffle=True)
            sampler.set_epoch(0)

            if hasattr(ds, "epoch"):
                ds.epoch = 0
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(0)
            batch_sampler = DynamicBatchSampler(
                sampler, 
                [2, ds.cfg.view_sampler.num_context_views], 
                ds.cfg.input_image_shape,
                seed=42,
                max_img_per_gpu=ds.cfg.view_sampler.max_img_per_gpu
            )
            self.src_batch_samplers.append(batch_sampler)
        
        # self.src_batch_samplers = [
        #     BatchedRandomSampler(
        #         ds,
        #         num_context_views=ds.cfg.view_sampler.num_context_views,
        #         world_size=self.world_size,
        #         rank=self.rank,
        #         batch_size=self.batch_size,
        #         drop_last=self.drop_last,
        #     )
        #     for ds in self.src_dataset_ls
        # ]
        # set epoch here
        print("Setting epoch for all underlying BatchedRandomSamplers")
        # for sampler in self.src_batch_samplers:
        #     sampler.set_epoch(0)
        self.raw_batches = [
            list(bs) for bs in self.src_batch_samplers
        ]  # index in original dataset
        self.n_batches = [len(b) for b in self.raw_batches]
        self.n_total_batch = sum(self.n_batches)
        # print("Total batch num is ", self.n_total_batch)
        # sampling probability
        if prob is None:
            # if not given, decide by dataset length
            self.prob = torch.tensor(self.n_batches) / self.n_total_batch
        else:
            self.prob = torch.as_tensor(prob)
    
    def __iter__(self):
        """Yields batches of indices in the format of (sample_idx, feat_idx) tuples,
        where indices correspond to ConcatDataset of src_dataset_ls
        """
        for _ in range(self.n_total_batch):
            idx_ds = torch.multinomial(
                self.prob, 1, replacement=True, generator=self.generator
            ).item()
            
            if 0 == len(self.raw_batches[idx_ds]):
                self.raw_batches[idx_ds] = list(self.src_batch_samplers[idx_ds])
            
            # get a batch from list - this is already in (sample_idx, feat_idx) format
            batch_raw = self.raw_batches[idx_ds].pop()

            # shift only the sample_idx by cumulative dataset length, keep feat_idx unchanged
            shift = self.cum_dataset_length[idx_ds]
            processed_batch = []

            for item in batch_raw:
                # item[0] is the sample index, item[1] is the number of images
                processed_item = (item[0] + shift, item[1], item[2])
                processed_batch.append(processed_item)
            yield processed_batch
        
    def set_epoch(self, epoch):
        """Set epoch for all underlying BatchedRandomSamplers"""
        for sampler in self.src_batch_samplers:
            sampler.set_epoch(epoch)
        # Reset raw_batches after setting new epoch
        self.raw_batches = [list(bs) for bs in self.src_batch_samplers]

    def __len__(self):
        return self.n_total_batch

def round_by(total, multiple, up=False):
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple