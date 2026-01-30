"""
Utility functions for distributed training.
"""

import os
import socket

import torch
import torch.distributed as dist

DEVICE = None


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


def setup_dist():
    if dist.is_initialized():
        return

    if os.environ.get("MASTER_ADDR", None) is None:
        hostname = socket.gethostbyname(socket.getfqdn())
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    global DEVICE
    DEVICE = device


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def device():
    if not dist.is_initialized():
        raise NameError
    return DEVICE


def torch_device_setup():
    assert torch.cuda.is_available(), "We basically require GPUs."
    setup_dist()
    torch_device = torch.device(device())
    dist.barrier()
    return torch_device


def is_main_process() -> bool:
    """Return True on rank‑0 (or non‑distributed run)."""
    return (
        (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    )
