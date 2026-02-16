import os
import subprocess
from urllib.request import urlretrieve

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageSequence
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)
from tqdm import tqdm

# from .distributed import (
#     all_gather,
#     barrier,
#     distribute_list_to_rank,
#     gather_list_of_dict,
#     get_rank,
#     get_world_size,
# )
try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
batch_size = 32


def clip_transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC, antialias=False),
            CenterCrop(n_px),
            # transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def get_aesthetic_model(cache_folder, device):
    """load the aethetic model"""
    path_to_model = cache_folder + "/sa_0_4_vit_l_14_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        # download aesthetic predictor
        if not os.path.isfile(path_to_model):
            try:
                print(f"trying urlretrieve to download {url_model} to {path_to_model}")
                urlretrieve(
                    url_model, path_to_model
                )  # unable to download https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true to pretrained/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth
            except:
                print(
                    f"unable to download {url_model} to {path_to_model} using urlretrieve, trying wget"
                )
                wget_command = ["wget", url_model, "-P", os.path.dirname(path_to_model)]
                subprocess.run(wget_command)
    m = nn.Linear(768, 1)
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    m.to(device)
    return m, clip_model


def compute_aesthetic_quality(aesthetic_model, clip_model, pil_list, device):
    aesthetic_model.eval()
    clip_model.eval()
    aesthetic_avg = 0.0
    num = 0
    video_results = []
    # for video_path in tqdm(video_list, disable=get_rank() > 0):
    image_transform = clip_transform(224)

    aesthetic_scores_list = []
    for i in range(0, len(pil_list)):
        image_batch = pil_list[i]
        image_batch = image_transform(image_batch)
        image_batch = image_batch.to(device)

        with torch.no_grad():
            image_feats = clip_model.encode_image(image_batch.unsqueeze(0)).to(
                torch.float32
            )
            image_feats = F.normalize(image_feats, dim=-1, p=2)
            aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)

        aesthetic_scores_list.append(aesthetic_scores)

    aesthetic_scores = torch.cat(aesthetic_scores_list, dim=0)
    normalized_aesthetic_scores = aesthetic_scores / 10
    cur_avg = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True)
    aesthetic_avg += cur_avg.item()
    num += 1

    aesthetic_avg /= num
    return aesthetic_avg
