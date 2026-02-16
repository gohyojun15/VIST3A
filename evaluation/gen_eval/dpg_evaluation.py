import argparse
import os.path as osp
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from evaluation.gen_eval.utils import get_file_list_with_pair, load_images


def parse_args():
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="evaluation/gen_eval/dpg_bench_sampled_answers.csv",
    )
    parser.add_argument(
        "--res_path",
        type=str,
        default="dpg_bench_results.txt",
    )
    parser.add_argument(
        "--pic-num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--vqa-model",
        type=str,
        default="mplug",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="CodeGoat24/UnifiedReward-qwen-7b",
        help="Path or HuggingFace model ID for the VQA model.",
    )
    parser.add_argument(
        "--generative_model_name",
        type=str,
        default="ours",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default=None,
        help="Path to the folder containing generated results.",
    )
    parser.add_argument(
        "--eval_save_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt_file_path",
        type=str,
        default="data/resources/dpg_bench_sampled_prompts.txt",
    )

    args = parser.parse_args()
    return args


class MPLUG(torch.nn.Module):
    def __init__(self, model_path="CodeGoat24/UnifiedReward-qwen-7b", device="gpu"):
        super().__init__()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="bfloat16",
        ).cuda()
        processor = AutoProcessor.from_pretrained(model_path)
        model.eval()
        self.model = model
        self.processor = processor

    def vqa(self, image, question):
        question = f"Please answer the following question with only one word 'Yes' or 'No': {question}\nASSISTANT:\n"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        chat_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=4096, use_cache=True
            )
            generated_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = self.processor.batch_decode(
                generated_trimmed, skip_special_tokens=True
            )[0]
        return output


def prepare_dpg_data(args):
    previous_id = ""
    current_id = ""
    question_dict = dict()
    data = pd.read_csv(args.csv)
    for _, line in data.iterrows():
        current_id = line.item_id
        qid = int(line.proposition_id)
        dependency_list_str = line.dependency.split(",")
        dependency_list_int = []
        for d in dependency_list_str:
            d_int = int(d.strip())
            dependency_list_int.append(d_int)

        if current_id == previous_id:
            question_dict[line.text]["qid2tuple"][qid] = line.tuple
            question_dict[line.text]["qid2dependency"][qid] = dependency_list_int
            question_dict[line.text]["qid2question"][qid] = (
                line.question_natural_language
            )
        else:
            question_dict[line.text] = dict(
                qid2tuple={qid: line.tuple},
                qid2dependency={qid: dependency_list_int},
                qid2question={qid: line.question_natural_language},
            )

        previous_id = current_id

    return question_dict


def compute_dpg_one_sample(args, question_dict, prompt, images, vqa_model):
    value = question_dict.get(prompt, None)
    if value is None:
        print(f"Warning: prompt not found in question_dict: {prompt[:80]}...")
        return None, None, None

    qid2tuple = value["qid2tuple"]
    qid2question = value["qid2question"]
    qid2dependency = value["qid2dependency"]

    qid2scores = dict()

    scores = []
    for img in images:
        for id, question in qid2question.items():
            answer = vqa_model.vqa(img, question)
            qid2scores[id] = float(answer == "Yes")

        qid2scores_orig = qid2scores.copy()

        for id, parent_ids in qid2dependency.items():
            # zero-out scores if parent questions are answered 'no'
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                if qid2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores[id] = 0

        score = sum(qid2scores.values()) / len(qid2scores)
        scores.append(score)
    average_score = sum(scores) / len(scores)

    return average_score, qid2tuple, qid2scores_orig


def main():
    args = parse_args()
    pair_dict = get_file_list_with_pair(args)
    accelerator = Accelerator()

    question_dict = prepare_dpg_data(args)

    if accelerator.is_main_process:
        with open(args.res_path, "w") as f:
            pass

    device = str(accelerator.device)
    if args.vqa_model == "mplug":
        vqa_model = MPLUG(model_path=args.model_path, device=device)
    else:
        raise NotImplementedError
    vqa_model = accelerator.prepare(vqa_model)
    vqa_model = getattr(vqa_model, "module", vqa_model)

    local_scores = []
    local_category2scores = defaultdict(list)
    for prompt, image_path_list in tqdm(pair_dict.items()):
        pil_image_list = load_images(prompt, image_path_list)
        try:
            score, qid2tuple, qid2scores = compute_dpg_one_sample(
                args=args,
                question_dict=question_dict,
                prompt=prompt,
                images=pil_image_list,
                vqa_model=vqa_model,
            )
            if score is None:
                continue

            local_scores.append(score)

            for qid in qid2tuple.keys():
                category = qid2tuple[qid].split("(")[0].strip()
                qid_score = qid2scores[qid]
                local_category2scores[category].append(qid_score)
        except Exception as e:
            print("Failed filename:", prompt, e)
            continue

    mean_dpg_score = np.mean(local_scores)

    global_categories = set(local_category2scores.keys())
    global_category2scores = dict()
    for category in global_categories:
        global_category2scores[category] = local_category2scores.get(category, [])

    global_category2scores_l1 = defaultdict(list)
    for category in global_categories:
        l1_category = category.split("-")[0].strip()
        global_category2scores_l1[l1_category].extend(global_category2scores[category])

    output = "L1 category scores:\n"
    for l1_category in global_category2scores_l1.keys():
        output += f"\t{l1_category}: {np.mean(global_category2scores_l1[l1_category]) * 100}\n"

    output += "L2 category scores:\n"
    for category in sorted(global_categories):
        output += f"\t{category}: {np.mean(global_category2scores[category]) * 100}\n"

    output += f"Folder path: {args.folder_path}\n"
    output += f"Save results to: {args.res_path}\n"
    output += f"DPG-Bench score: {mean_dpg_score * 100}"

    with open(args.res_path, "a") as f:
        f.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
