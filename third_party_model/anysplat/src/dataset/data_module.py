import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from third_party_model.anysplat.src.dataset import *
from third_party_model.anysplat.src.global_cfg import get_cfg


from ..misc.step_tracker import StepTracker
from ..misc.utils import get_world_size, get_rank
from . import DatasetCfgWrapper, get_dataset
from .types import DataShim, Stage
from .data_sampler import BatchedRandomSampler, MixedBatchSampler, custom_collate_fn
from .validation_wrapper import ValidationWrapper


def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


# the training ratio of datasets (example)
prob_mapping = {DatasetScannetpp: 0.5, DatasetDL3DV: 0.5, DatasetCo3d: 0.5}


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfgs: list[DatasetCfgWrapper]
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfgs: list[DatasetCfgWrapper],
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        self.generator = generator
        return self.generator

    def train_dataloader(self):
        dataset, datasets_ls = get_dataset(
            self.dataset_cfgs, "train", self.step_tracker, self.dataset_shim
        )
        world_size = get_world_size()
        rank = get_rank()
        # breakpoint()
        prob_ls = [prob_mapping[type(dataset)] for dataset in datasets_ls]
        # we assume all the dataset share the same num_context_views

        if len(datasets_ls) > 1:
            prob = prob_ls
            context_num_views = [
                dataset.cfg.view_sampler.num_context_views for dataset in datasets_ls
            ]
        else:
            prob = None
            dataset_key = next(iter(get_cfg()["dataset"]))
            dataset_cfg = get_cfg()["dataset"][dataset_key]
            context_num_views = dataset_cfg["view_sampler"]["num_context_views"]

        sampler = MixedBatchSampler(
            datasets_ls,
            batch_size=self.data_loader_cfg.train.batch_size,  # Not used here!
            num_context_views=context_num_views,
            world_size=world_size,
            rank=rank,
            prob=prob,
            generator=self.get_generator(self.data_loader_cfg.train),
        )
        sampler.set_epoch(0)
        self.train_loader = DataLoader(
            dataset,
            # self.data_loader_cfg.train.batch_size,
            # shuffle=not isinstance(dataset, IterableDataset),
            batch_sampler=sampler,
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.generator,
            worker_init_fn=worker_init_fn,
            # collate_fn=custom_collate_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )
        # breakpoint()
        # Set epoch for train and validation loaders (if applicable)
        if hasattr(self.train_loader, "dataset") and hasattr(
            self.train_loader.dataset, "set_epoch"
        ):
            print("Training: Set Epoch in DataModule")
            self.train_loader.dataset.set_epoch(0)
        if hasattr(self.train_loader, "sampler") and hasattr(
            self.train_loader.sampler, "set_epoch"
        ):
            print("Training: Set Epoch in DataModule")
            self.train_loader.sampler.set_epoch(0)

        return self.train_loader

    def val_dataloader(self):
        dataset, datasets_ls = get_dataset(
            self.dataset_cfgs, "val", self.step_tracker, self.dataset_shim
        )
        world_size = get_world_size()
        rank = get_rank()
        # here, we random select one dataset for val
        dataset_key = next(iter(get_cfg()["dataset"]))
        dataset_cfg = get_cfg()["dataset"][dataset_key]
        if len(datasets_ls) > 1:
            prob = [0.5] * len(datasets_ls)
        else:
            prob = None
        sampler = MixedBatchSampler(
            datasets_ls,
            batch_size=self.data_loader_cfg.train.batch_size,
            num_context_views=dataset_cfg["view_sampler"]["num_context_views"],
            world_size=world_size,
            rank=rank,
            prob=prob,
            generator=self.get_generator(self.data_loader_cfg.train),
        )
        sampler.set_epoch(0)
        self.val_loader = DataLoader(
            dataset,
            self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            sampler=sampler,
            generator=self.get_generator(self.data_loader_cfg.val),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
        )
        if hasattr(self.val_loader, "dataset") and hasattr(
            self.val_loader.dataset, "set_epoch"
        ):
            print("Validation: Set Epoch in DataModule")
            self.val_loader.dataset.set_epoch(0)
        if hasattr(self.val_loader, "sampler") and hasattr(
            self.val_loader.sampler, "set_epoch"
        ):
            print("Validation: Set Epoch in DataModule")
            self.val_loader.sampler.set_epoch(0)
        return self.val_loader

    def test_dataloader(self):
        dataset = get_dataset(
            self.dataset_cfgs, "test", self.step_tracker, self.dataset_shim
        )
        data_loader = DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
        )

        return data_loader
