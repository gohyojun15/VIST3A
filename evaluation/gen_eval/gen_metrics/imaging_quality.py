import os
import subprocess

import torch
from pyiqa.archs.musiq_arch import MUSIQ
from torchvision import transforms
from tqdm import tqdm


def transform(images, preprocess_mode="shorter"):
    if preprocess_mode.startswith("shorter"):
        _, h, w = images.size()
        if min(h, w) > 512:
            scale = 512.0 / min(h, w)
            images = transforms.Resize(
                size=(int(scale * h), int(scale * w)), antialias=False
            )(images)
            if preprocess_mode == "shorter_centercrop":
                images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == "longer":
        _, h, w = images.size()
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            images = transforms.Resize(
                size=(int(scale * h), int(scale * w)), antialias=False
            )(images)

    return images


def evaluate_imaging_quality(model, pil_list, device):
    process_mode = "shorter"
    video_results = []
    for image_one in pil_list:
        # images = load_video(video_path)
        image = transforms.ToTensor()(image_one)
        image = transform(image, process_mode)
        score = model(image.unsqueeze(0).to(device))
        video_results.append(score[0][0])
    average_score = sum(video_results) / len(video_results)
    average_score = average_score / 100.0
    return average_score.item()


def load_MUSIQ_model(device):
    CACHE_DIR = "/iopsstor/scratch/cscs/hyojgo"
    musiq_spaq_path = f"{CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
    if not os.path.isfile(musiq_spaq_path):
        wget_command = [
            "wget",
            "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth",
            "-P",
            os.path.dirname(musiq_spaq_path),
        ]
        subprocess.run(wget_command, check=True)
    # submodules_dict[dimension] = {"model_path": musiq_spaq_path}

    model = MUSIQ(pretrained_model_path=musiq_spaq_path)
    model.to(device)
    model.training = False
    return model
