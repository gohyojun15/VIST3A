#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .layers import Conv2d, Linear, LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    # Freeze all params except LoRA (and optionally biases).
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    # Export only LoRA (and optionally bias) weights for lightweight checkpoints.
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {
            k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k
        }
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


@dataclass
class LoraConfig:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.0
    bias: str = "lora_only"
    target_modules: Optional[Tuple[str, ...]] = None  # Optional!
    fan_in_fan_out: bool = False
    finetune_encoder: bool = False
    freeze_head: bool = False


def parse_lora_mode(spec: str):
    cfg = LoraConfig()  # start from defaults
    # Spec format: r<rank>,a<alpha>,d<dropout>,b<bias>,t<targets>,f<0/1>,enc,fix_head
    pattern = re.compile(
        r"""
        (?P<key>[radbf t])            # first letter
        (?:
            (?P<num>[\d.]+)           # number (r, a, d, f)
            |
            (?P<str>[^,]+)            # string (b, t)
        )
        """,
        re.VERBOSE,
    )

    for chunk in spec.split(","):
        m = pattern.fullmatch(chunk.strip())

        chunk = chunk.strip().lower()
        # --- special boolean flags --------------------------------------
        if chunk == "enc":
            cfg.finetune_encoder = True
            continue
        if chunk in {"fix_head", "fixhead"}:
            cfg.freeze_head = True
            continue

        # --- numeric / keyed options ------------------------------------
        m = pattern.fullmatch(chunk)
        if not m:
            raise ValueError(f"Bad LoRA chunk: {chunk!r}")
        k = m["key"]
        if k == "r":
            cfg.r = int(m["num"])
        elif k == "a":
            cfg.alpha = int(m["num"])
        elif k == "d":
            cfg.dropout = float(m["num"])
        elif k == "b":
            cfg.bias = m["str"]
            if cfg.bias not in {"none", "all", "lora_only"}:
                raise ValueError("b chunk must be none|all|lora_only")
        elif k == "t":
            cfg.target_modules = tuple(m["str"].split("|"))
        elif k == "f":
            cfg.fan_in_fan_out = bool(int(m["num"]))
        else:
            raise AssertionError("impossible")

    return cfg


def _get_parent(model, dotted_name: str):
    """
    Returns the parent module that owns the sub-module called *dotted_name* and
    the last attribute in that path.

    Example
    -------
    dotted_name = "transformer.h.3.attn.q_proj"
    → returns (transformer.h.3.attn      , "q_proj")
    """
    if "." not in dotted_name:
        return model, dotted_name
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def add_lora(
    model: nn.Module,
    target_modules=None,
    r: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    fan_in_fan_out: bool = False,
):
    # Replace target Linear/Conv2d modules with LoRA-wrapped versions in-place.
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) and not isinstance(module, nn.Conv2d):
            continue

        if target_modules and not any(t in full_name for t in target_modules):
            continue

        if isinstance(module, nn.Linear):
            # Build a LoRA Linear with identical I/O dimensions
            lora_layer = Linear(
                module.in_features,
                module.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                fan_in_fan_out=fan_in_fan_out,
                bias=module.bias is not None,
            )
        elif isinstance(module, nn.Conv2d):
            assert module.kernel_size[0] == module.kernel_size[1], (
                "Only square kernels are supported for Conv2d LoRA layers."
            )
            lora_layer = Conv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size[0],
                stride=module.stride,
                padding=module.padding[0],
                dilation=module.dilation[0],
                groups=module.groups,
                bias=module.bias is not None,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
            )
        lora_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            lora_layer.bias.data.copy_(module.bias.data)
        module.weight = lora_layer.weight
        parent, child_name = _get_parent(model, full_name)
        setattr(parent, child_name, lora_layer)
    return model
