import json
import os
import random
import warnings

import requests
import torch
import tqdm
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

warnings.filterwarnings("ignore")
import re
from typing import Dict, Optional

_FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _find_first_float(text: str) -> Optional[float]:
    m = re.search(_FLOAT, text)
    return float(m.group()) if m else None


def parse_unified_scores(text: str) -> Dict[str, float]:
    def grab(label: str) -> Optional[float]:
        # e.g., "Alignment Score (1-5): 3.62" or "Alignment: 3.62"
        pattern = rf"{label}\s*Score?(?:\s*\(.*?\))?\s*[:\-]\s*({_FLOAT})"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        # fallback: find line containing the label and extract first float
        for line in text.splitlines():
            if label.lower() in line.lower():
                val = _find_first_float(line)
                if val is not None:
                    return val
        return None

    out = {
        "alignment": grab("Alignment"),
        "coherence": grab("Coherence"),
        "style": grab("Style"),
    }
    # Optional: raise if any missing
    missing = [k for k, v in out.items() if v is None]
    if missing:
        raise ValueError(f"Could not parse: {', '.join(missing)}")
    return out


def get_unified_reward_model():
    model_path = "CodeGoat24/UnifiedReward-qwen-7b"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="bfloat16",  # , device_map="auto"
    ).cuda()
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


def get_unified_reward_output(model, processor, pil_image_list, prompt):
    alignment_score = []
    coherence_score = []
    style_score = []
    for image in pil_image_list:
        question = (
            "You are presented with a generated image and its associated text caption. Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n\n"
            "1. Evaluate each word in the caption based on how well it is visually represented in the image. Assign a numerical score to each word using the format:\n"
            '   Word-wise Scores: [["word1", score1], ["word2", score2], ..., ["wordN", scoreN], ["[No_mistakes]", scoreM]]\n'
            "   - A higher score indicates that the word is less well represented in the image.\n"
            "   - The special token [No_mistakes] represents whether all elements in the caption were correctly depicted. A high score suggests no mistakes; a low score suggests missing or incorrect elements.\n\n"
            "2. Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
            "- Alignment Score: How well the image matches the caption in terms of content.\n"
            "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
            "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
            "Output your evaluation using the format below:\n\n"
            "---\n\n"
            'Word-wise Scores: [["word1", score1], ..., ["[No_mistakes]", scoreM]]\n\n'
            "Alignment Score (1-5): X\n"
            "Coherence Score (1-5): Y\n"
            "Style Score (1-5): Z\n\n"
            f"Your task is provided as follows:\nText Caption: [{prompt}]\nASSISTANT:\n"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        chat_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[chat_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=4096, use_cache=True
                )
                generated_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output = processor.batch_decode(
                    generated_trimmed, skip_special_tokens=True
                )[0]
                scores = parse_unified_scores(output)
                alignment_score.append(scores["alignment"])
                coherence_score.append(scores["coherence"])
                style_score.append(scores["style"])
        except:
            continue

    # calculate avearge
    alignment_score = sum(alignment_score) / len(alignment_score)
    coherence_score = sum(coherence_score) / len(coherence_score)
    style_score = sum(style_score) / len(style_score)
    return alignment_score, coherence_score, style_score
