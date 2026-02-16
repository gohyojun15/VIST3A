import argparse
import json
import os
from typing import List

import cv2
import numpy as np
import pyiqa
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm

from evaluation.gen_eval.gen_metrics.aesthetic_quality import (
    compute_aesthetic_quality,
    get_aesthetic_model,
)
from evaluation.gen_eval.gen_metrics.clip_score import calculate_clip_score
from evaluation.gen_eval.gen_metrics.imaging_quality import (
    evaluate_imaging_quality,
    load_MUSIQ_model,
)
from evaluation.gen_eval.gen_metrics.unified_reward import (
    get_unified_reward_model,
    get_unified_reward_output,
)


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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return pil_frames


def get_file_list_with_pair(args):
    """Build a dict mapping prompt -> video path from the folder structure.

    Expected folder structure:
        folder_path/
            sequence_001/
                gs.mp4
                prompt.txt
            sequence_002/
                gs.mp4
                prompt.txt
            ...
    """
    sequence_list = os.listdir(args.folder_path)
    pair_dict = {}
    for sequence in sequence_list:
        sequence_folder_path = os.path.join(args.folder_path, sequence)
        if not os.path.isdir(sequence_folder_path):
            continue
        gaussian_video = os.path.join(sequence_folder_path, "gs.mp4")
        prompt_file = os.path.join(sequence_folder_path, "prompt.txt")
        if not os.path.exists(gaussian_video) or not os.path.exists(prompt_file):
            print(f"Skipping {sequence}: missing gs.mp4 or prompt.txt")
            continue
        with open(prompt_file, "r") as f:
            prompt = f.readline().strip()
        pair_dict[prompt] = gaussian_video
    return pair_dict


def load_images(prompt, video_or_image_list) -> List[Image.Image]:
    """Load frames from a video path or return images from a list."""
    if isinstance(video_or_image_list, list):
        return video_or_image_list
    elif isinstance(video_or_image_list, str):
        return sample_video_frames(video_or_image_list)
    else:
        raise ValueError(f"Invalid video_or_image_list: {video_or_image_list}")


def main(args):
    pair_dict = get_file_list_with_pair(args)
    print(f"Found {len(pair_dict)} prompt-video pairs.")

    # Init metrics
    aesthetic_model, clip_model = get_aesthetic_model(args.cache_folder, device="cuda")
    musiq_model = load_MUSIQ_model(device="cuda")
    clipscore_model = CLIPScore(
        model_name_or_path="openai/clip-vit-base-patch16",
    ).to("cuda")
    longclip_score_model = CLIPScore(
        model_name_or_path="zer0int/LongCLIP-L-Diffusers",
    ).to("cuda")
    unified_reward_model, unified_reward_processor = get_unified_reward_model()

    eval_results = []
    for prompt, video_or_image_list in tqdm(pair_dict.items()):
        pil_image_list = load_images(prompt, video_or_image_list)
        (
            unified_reward_alignment_score,
            unified_reward_coherence_score,
            unified_reward_style_score,
        ) = get_unified_reward_output(
            unified_reward_model, unified_reward_processor, pil_image_list, prompt
        )
        clip_score = calculate_clip_score(
            clipscore_model, pil_image_list, prompt, "cuda"
        )
        longclip_score = calculate_clip_score(
            longclip_score_model, pil_image_list, prompt, "cuda"
        )
        aesthetic_avg = compute_aesthetic_quality(
            aesthetic_model, clip_model, pil_image_list, "cuda"
        )
        imaging_avg = evaluate_imaging_quality(musiq_model, pil_image_list, "cuda")
        eval_results.append(
            {
                "prompt": prompt,
                "unified_reward_alignment_score": unified_reward_alignment_score,
                "unified_reward_coherence_score": unified_reward_coherence_score,
                "unified_reward_style_score": unified_reward_style_score,
                "clip_score": clip_score,
                "longclip_score": longclip_score,
                "aesthetic_avg": aesthetic_avg,
                "imaging_avg": imaging_avg,
            }
        )

    all_average = {
        "unified_reward_alignment_score": np.mean(
            [r["unified_reward_alignment_score"] for r in eval_results]
        ).item(),
        "unified_reward_coherence_score": np.mean(
            [r["unified_reward_coherence_score"] for r in eval_results]
        ).item(),
        "unified_reward_style_score": np.mean(
            [r["unified_reward_style_score"] for r in eval_results]
        ).item(),
        "clip_score": np.mean([r["clip_score"] for r in eval_results]).item(),
        "longclip_score": np.mean([r["longclip_score"] for r in eval_results]).item(),
        "aesthetic_avg": np.mean([r["aesthetic_avg"] for r in eval_results]).item(),
        "imaging_avg": np.mean([r["imaging_avg"] for r in eval_results]).item(),
    }
    all_eval_results = {
        "all_average": all_average,
        "eval_results": eval_results,
    }
    with open(args.eval_save_path, "w") as f:
        json.dump(all_eval_results, f, indent=4)
    print(f"Results saved to {args.eval_save_path}")
    print(f"Average scores: {json.dumps(all_average, indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation quality evaluation.")
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="Path to the folder containing generated results.",
    )
    parser.add_argument(
        "--eval_save_path",
        type=str,
        default="eval_results.json",
        help="Path to save the evaluation results JSON.",
    )
    parser.add_argument(
        "--cache_folder",
        type=str,
        default=None,
        help="Cache folder for model weights (e.g., aesthetic model).",
    )
    args = parser.parse_args()
    main(args)
