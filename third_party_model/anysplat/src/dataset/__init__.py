from dataclasses import fields
from typing import Callable
from torch.utils.data import Dataset, ConcatDataset
import bisect

from ..misc.step_tracker import StepTracker
from .types import Stage
from .view_sampler import get_view_sampler
from .dataset_dl3dv import DatasetDL3DV, DatasetDL3DVCfgWrapper
from .dataset_scannetpp import DatasetScannetpp, DatasetScannetppCfgWrapper
from .dataset_co3d import DatasetCo3d, DatasetCo3dCfgWrapper

DATASETS: dict[str, Dataset] = {
    "co3d": DatasetCo3d,
    "scannetpp": DatasetScannetpp,
    "dl3dv": DatasetDL3DV,
}

DatasetCfgWrapper = DatasetDL3DVCfgWrapper | DatasetScannetppCfgWrapper | DatasetCo3dCfgWrapper

class TestDatasetWarpper(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, idx):

        return self.dataset[(idx, self.dataset.view_sampler.num_context_views, self.dataset.cfg.input_image_shape[1] // 14)] # fake parameters here, to fit the input of dataset
    
    def __len__(self):
        return len(self.dataset)

        
    
class CustomConcatDataset(ConcatDataset):

    def __getitem__(self, idx_tuple):

        if isinstance(idx_tuple, list):
            idx_tuple = idx_tuple[0]

        idx = idx_tuple[0]
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][(sample_idx, idx_tuple[1], idx_tuple[2])]


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
    dataset_shim: Callable[[Dataset, str], Dataset]
) -> list[Dataset]:
    datasets = []
    if stage != "test":
        if stage == "val":
            cfgs = [cfgs[0]]
        for cfg in cfgs:
            (field,) = fields(type(cfg))
            cfg = getattr(cfg, field.name)
            
            view_sampler = get_view_sampler(
                cfg.view_sampler,
                stage,
                cfg.overfit_to_scene is not None,
                cfg.cameras_are_circular,
                step_tracker,
            )
            dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
            dataset = dataset_shim(dataset, stage)
            datasets.append(dataset)

        return CustomConcatDataset(datasets), datasets
    elif stage == "test":
        assert len(cfgs) == 1
        cfg = cfgs[0]
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)
        
        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        dataset = dataset_shim(dataset, stage)

        return TestDatasetWarpper(dataset)
    else:
        NotImplementedError(f"Stage {stage} is not supported")
