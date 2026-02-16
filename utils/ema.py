# Copied from another repo, but I can't remember exactly which one.

# import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch

# import torch.distributed as dist
# import torch.distributed.checkpoint as dist_cp
# from torch.distributed.checkpoint.default_planner import (
#     DefaultLoadPlanner,
#     DefaultSavePlanner,
# )


@dataclass
class EMAConfig:
    decay: float = 0.99
    update_step_interval: int = 1
    ema_dtype: torch.dtype = torch.float32


class FSDPEMAWrapper:
    def __init__(
        self, model: torch.nn.Module, cfg: EMAConfig, only_trainable: bool = True
    ):
        self.cfg = cfg

        self.param_names: list[str] = []
        self.params: Dict[str, torch.nn.Parameter] = {}
        self.ema: Dict[str, torch.Tensor] = {}
        self._backup: Optional[Dict[str, torch.Tensor]] = None

        for name, p in model.named_parameters():
            if only_trainable and (not p.requires_grad):
                continue

            self.param_names.append(name)
            self.params[name] = p

            ema_t = p.detach().clone()
            if ema_t.is_floating_point():
                ema_t = ema_t.to(dtype=cfg.ema_dtype)
            self.ema[name] = ema_t

    def get_current_decay(self, step: int) -> float:
        return min((1 + step) / (10 + step), self.cfg.decay)

    @torch.no_grad()
    def step(self, step: int):
        if (step + 1) % self.cfg.update_step_interval != 0:
            return

        d = self.get_current_decay(step)
        one_minus = 1.0 - d

        for name in self.param_names:
            p = self.params[name].detach()
            ema_t = self.ema[name]

            src = p
            if ema_t.is_floating_point():
                src = src.to(dtype=ema_t.dtype)

            # ema = d*ema + (1-d)*p
            ema_t.mul_(d).add_(src, alpha=one_minus)

    @torch.no_grad()
    def copy_ema_to(self, store_temp: bool = True):
        if store_temp:
            self._backup = {
                n: self.params[n].detach().clone() for n in self.param_names
            }

        for n in self.param_names:
            p = self.params[n]
            src = self.ema[n]
            if p.is_floating_point():
                src = src.to(dtype=p.dtype)
            p.copy_(src)

    @torch.no_grad()
    def copy_temp_to(self):
        if self._backup is None:
            return
        for n in self.param_names:
            self.params[n].copy_(self._backup[n])
        self._backup = None

    def dcp_state_dict(self) -> dict:
        return {"ema": self.ema}
