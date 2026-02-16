import os
from typing import List

import cv2
from PIL import Image


def sample_video_frames(video_path, num_samples=8):
    """Load video and return equally sampled frames as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / num_samples) for i in range(num_samples)]

    pil_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        # Convert BGR (OpenCV) -> RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return pil_frames


def load_images(prompt, video_or_image_list) -> List[Image.Image]:
    if isinstance(video_or_image_list, list):
        # image file_path branch
        a = 3
    elif isinstance(video_or_image_list, str):
        # video file_path branch
        image_list = sample_video_frames(video_or_image_list)
    else:
        raise ValueError(f"Invalid video_or_image_list: {video_or_image_list}")
    return image_list


def get_file_list_with_pair(args):
    sequence_list = os.listdir(args.folder_path)
    pair_dict = {}
    for sequence in sequence_list:
        sequence_folder_path = os.path.join(args.folder_path, sequence)
        gaussian_video = os.path.join(sequence_folder_path, "gs.mp4")
        prompt = os.path.join(sequence_folder_path, "prompt.txt")
        with open(prompt, "r") as f:
            prompt = f.readlines()
        pair_dict[prompt[0].strip()] = gaussian_video
    return pair_dict
