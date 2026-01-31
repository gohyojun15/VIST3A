import argparse
import os
import os.path as osp
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from evaluation.novel_view_synthesis_bench.message import write_csv
from third_party_model.anysplat.src.evaluation.metrics import (
    compute_lpips,
    compute_psnr,
    compute_ssim,
)
from utils.argument import stitching_nvs_evaluation_argument


def compute_metrics(pred_image, image):
    psnr = compute_psnr(pred_image, image)
    ssim = compute_ssim(pred_image, image)
    lpips = compute_lpips(pred_image.cuda(), image.cuda()).cpu()
    return psnr, ssim, lpips


def main(args: argparse.Namespace):
    # Expects outputs in: output_dir/images/<seq>/{gt,pred}/000000.png ...
    image_path = Path(args.output_dir) / "images"
    to_tensor = transforms.ToTensor()

    seq_list = os.listdir(image_path)
    psnr_list = []
    ssim_list = []
    lpips_list = []
    for seq in tqdm(seq_list):
        seq_path = image_path / seq
        gt_path = seq_path / "gt"
        pred_path = seq_path / "pred"
        gt_images = [
            to_tensor(Image.open(gt_path / f"{i:0>6}.png"))
            for i in range(len(os.listdir(gt_path)))
        ]
        pred_images = [
            to_tensor(Image.open(pred_path / f"{i:0>6}.png"))
            for i in range(len(os.listdir(pred_path)))
        ]

        psnr_sequence = []
        ssim_sequence = []
        lpips_sequence = []
        for gt_image, pred_image in zip(gt_images, pred_images):
            psnr, ssim, lpips = compute_metrics(
                pred_image.unsqueeze(0), gt_image.unsqueeze(0)
            )
            psnr_sequence.append(psnr)
            ssim_sequence.append(ssim)
            lpips_sequence.append(lpips)

        psnr_list.append(np.mean(psnr_sequence))
        ssim_list.append(np.mean(ssim_sequence))
        lpips_list.append(np.mean(lpips_sequence))
        # psnr, ssim, lpips = compute_metrics(pred_images, gt_images)
        write_csv(
            osp.join(args.output_dir, f"_all_samples.csv"),
            {
                "seq": seq,
                "PSNR": np.mean(psnr_sequence),
                "SSIM": np.mean(ssim_sequence),
                "LPIPS": np.mean(lpips_sequence),
            },
        )
    # num_samples = len(ssim_list)
    metric_dict = {
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "LPIPS": np.mean(lpips_list),
    }
    statistics_file = osp.join(args.output_dir, f"overall-metric")  # + ".csv"
    if getattr(args, "save_suffix", None) is not None:
        statistics_file += f"-{args.save_suffix}"
    statistics_file += ".csv"
    write_csv(statistics_file, metric_dict)


if __name__ == "__main__":
    args = stitching_nvs_evaluation_argument().parse_args()
    with torch.no_grad():
        main(args)
