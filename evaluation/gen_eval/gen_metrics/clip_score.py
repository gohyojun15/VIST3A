import torch
from PIL import Image, ImageOps
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor


def pil_to_clip_tensor(img: Image.Image) -> torch.Tensor:
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    # (C,H,W) uint8 [0..255]
    img = pil_to_tensor(img)
    # img = Resize(224)(img)
    return img


def calculate_clip_score(model, pil_list, prompt, device):
    video_results = []
    for image_one in pil_list:
        # images = load_video(video_path)
        image = pil_to_clip_tensor(image_one)
        score = model(image.unsqueeze(0).to(device), prompt)
        video_results.append(score)
    average_score = sum(video_results) / len(video_results)
    average_score = average_score.item()

    return average_score
