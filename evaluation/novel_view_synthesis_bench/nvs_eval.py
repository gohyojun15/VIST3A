import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from evaluation.datasets.re10k_nvs import Re10KNVSDataset
from models.stitched_model import StitchVAE3D
from third_party_model.anysplat.src.misc.image_io import save_image
from utils.argument import stitching_nvs_evaluation_argument
from utils.lora_util.utils import add_lora, parse_lora_mode
from utils.utils_for_thirdparty import (
    cast_to_bfloat16,
    load_feedforward_model,
    load_vae,
)


def load_stitching_model(args: argparse.Namespace):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # we only use one gpu for evaluation
    feed_forward_model = load_feedforward_model(args, device)
    video_vae = load_vae(args, device)
    video_vae.eval()
    stitched_model = StitchVAE3D(
        diffusion_vae=video_vae,
        feedforward_model=feed_forward_model,
        device=device,
        stitching_layer_location=args.stitching_layer_location,
        stitching_layer_config=args.stitching_layer_config,
        resolution=args.resolution,
        stitching_layer_init_path=args.initialization_weight_path,
    )
    lora_cfg = parse_lora_mode(args.lora_config)
    add_lora(
        stitched_model.stitched_3d_model,
        target_modules=lora_cfg.target_modules,
        r=lora_cfg.r,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        fan_in_fan_out=lora_cfg.fan_in_fan_out,
    )
    state_dict = torch.load(
        args.checkpoint_path, weights_only=False, map_location="cpu"
    )
    stitched_model.stitched_3d_model.load_state_dict(state_dict["lora"], strict=False)
    stitched_model.stitching_layer.weight.data = state_dict["stitching_layer"]["weight"]
    stitched_model.stitching_layer.bias.data = state_dict["stitching_layer"]["bias"]
    stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.mask_token.data = (
        state_dict["mask_token"]
    )
    stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.cls_token.data = (
        state_dict["cls_token"]
    )
    stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.register_tokens.data = state_dict[
        "register_tokens"
    ]
    cast_to_bfloat16(stitched_model)
    return stitched_model.to(device)


def get_nvs_eval_dataset(datasets: list[tuple[str, str]], args: argparse.Namespace):
    n, r = datasets[0]  # only support single dataset for now
    if n == "re10k":
        logger.info(f"Loading Re10K NVS evaluation dataset from {r}")
        dataset = Re10KNVSDataset(
            Re10K_DIR=r,
            split="test",
            load_img_size=args.resolution,
            feedforward_img_size=args.feedforward_resolution,
            seq_file="evaluation/datasets/re10k_test.txt",
        )
    else:
        raise NotImplementedError(f"NVS evaluation dataset {n} is not implemented.")
    return dataset


def inference_nvs(
    images: torch.Tensor,
    vae_images: torch.Tensor,
    model: torch.nn.Module,
    target_view_index: list[int],
):
    source_index = [i for i in list(range(len(images))) if i not in target_view_index]
    ctx_images = images.unsqueeze(0).to("cuda")[:, source_index]
    ctx_images = torch.cat((ctx_images, ctx_images[:, -1].unsqueeze(1)), dim=1)
    ctx_vae_images = vae_images.unsqueeze(0).to("cuda")[:, source_index]
    ctx_vae_images = torch.cat(
        (ctx_vae_images, ctx_vae_images[:, -1].unsqueeze(1)), dim=1
    )
    num_context_view = ctx_images.shape[1]
    with torch.cuda.amp.autocast(enabled=True):
        encoder_output = model(
            images=ctx_vae_images.permute(0, 2, 1, 3, 4).cuda() * 2 - 1,
            feedforward_image=ctx_images.permute(0, 2, 1, 3, 4).cuda() * 2 - 1,
            train=False,
        )
    gaussians, pred_context_pose = (
        encoder_output.gaussians,
        encoder_output.pred_context_pose,
    )
    tgt_images = images.unsqueeze(0).to("cuda")[:, target_view_index]
    tgt_vae_images = vae_images.unsqueeze(0).to("cuda")[:, target_view_index]
    # duplication
    vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1)
    vggt_input_vae_image = torch.cat((ctx_vae_images, tgt_vae_images), dim=1)
    with torch.cuda.amp.autocast(enabled=True):
        predictions = model(
            images=vggt_input_vae_image.permute(0, 2, 1, 3, 4).cuda() * 2 - 1,
            feedforward_image=vggt_input_image.permute(0, 2, 1, 3, 4).cuda() * 2 - 1,
            train=False,
        )
    pred_all_extrinsic, pred_all_intrinsic = (
        predictions.pred_context_pose["extrinsic"],
        predictions.pred_context_pose["intrinsic"],
    )

    pred_all_context_extrinsic, pred_all_target_extrinsic = (
        pred_all_extrinsic[:, :num_context_view],
        pred_all_extrinsic[:, num_context_view:],
    )
    _, pred_all_target_intrinsic = (
        pred_all_intrinsic[:, :num_context_view],
        pred_all_intrinsic[:, num_context_view:],
    )
    scale_factor = (
        pred_context_pose["extrinsic"][:, :, :3, 3].mean()
        / pred_all_context_extrinsic[:, :, :3, 3].mean()
    )
    pred_all_target_extrinsic[..., :3, 3] = (
        pred_all_target_extrinsic[..., :3, 3] * scale_factor
    )
    pred_all_context_extrinsic[..., :3, 3] = (
        pred_all_context_extrinsic[..., :3, 3] * scale_factor
    )
    print("scale_factor:", scale_factor)
    v = tgt_images.shape[1]
    h, w = 448, 448
    output = model.stitched_3d_model.decoder.forward(
        gaussians,
        pred_all_target_extrinsic,
        pred_all_target_intrinsic.float(),
        torch.ones(1, v, device="cuda") * 0.01,
        torch.ones(1, v, device="cuda") * 100,
        (h, w),
    )
    predicted_target_view_images = output.color[0]
    return predicted_target_view_images


def main(args):
    model = load_stitching_model(args)
    model.eval()
    logger.info(
        f"Loaded model from {args.checkpoint_path} with stitching vae: feedforward {args.feedforward_model} and video vae {args.video_model}"
    )

    if len(args.dataset) > 1:
        raise NotImplementedError(
            "Currently only single dataset evaluation is supported."
        )
    dataset = get_nvs_eval_dataset(args.dataset, args)
    logger.info(f"Loaded NVS evaluation dataset with {len(dataset)} scenes.")

    with open(args.seq_id_map, "r") as f:
        seq_id_map = json.load(f)

    tbar = tqdm(dataset.sequence_list, desc=f"[{args.dataset} eval]")

    for seq_name in tbar:
        ids = seq_id_map[seq_name]
        batch = dataset.get_data(sequence_name=seq_name, ids=ids)
        images = batch["images"]
        vae_images = batch["vae_images"]
        target_view_index = list(range(len(ids)))[::-1][:4]
        gt_images = images[target_view_index]
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            predicted_target_view_images = inference_nvs(
                images,
                vae_images,
                model,
                target_view_index,
            )
        save_path = Path(f"{args.output_dir}/images/{seq_name}")
        for idx, (gt_image, predicted_target_view_image) in enumerate(
            zip(gt_images, predicted_target_view_images)
        ):
            save_image(gt_image, save_path / "gt" / f"{idx:0>6}.png")
            save_image(
                predicted_target_view_image, save_path / "pred" / f"{idx:0>6}.png"
            )


if __name__ == "__main__":
    args = stitching_nvs_evaluation_argument().parse_args()
    with torch.no_grad():
        main(args)
