import math
import os
import random

import torch
import torch.distributed as dist
import transformers
from einops import rearrange
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import utils.dist_util as dist_util
from data.dataset_util import create_stitching_dataloader
from models.anysplat_stitched import AnySplatStitched, TaskLossAnySplat
from models.stitched_model import StitchVAE3D
from third_party_model.anysplat.src.model.model.anysplat import AnySplat
from utils.argument import stitching_training_argument
from utils.dist_util import torch_device_setup
from utils.lora_util.utils import (
    add_lora,
    lora_state_dict,
    mark_only_lora_as_trainable,
    parse_lora_mode,
)
from utils.utils_for_thirdparty import (
    cast_to_bfloat16,
    load_feedforward_model,
    load_vae,
)


def save_checkpoint(path, epoch, stitched_model, optimizer, scheduler, args):
    logger.info(f"Saving resume checkpoint at epoch {epoch}.")
    save_folder_for_epoch = os.path.join(path, f"epoch_{epoch}")
    if not os.path.exists(save_folder_for_epoch):
        os.makedirs(save_folder_for_epoch)
    resume_checkpoint_path = os.path.join(
        save_folder_for_epoch, "resume_checkpoint.pth"
    )
    # save optimizer and scheduler state
    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": args,
        },
        resume_checkpoint_path,
    )

    # save model
    save_model_path = os.path.join(
        save_folder_for_epoch, f"stitched_model_epoch_{epoch}.pth"
    )
    logger.info(f"Saving model at epoch {epoch}.")
    state_dict = {
        "lora": lora_state_dict(stitched_model.stitched_3d_model, bias="lora_only"),
        "stitching_layer": stitched_model.stitching_layer.state_dict(),
    }
    state_dict["mask_token"] = (
        stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.mask_token.data
    )
    state_dict["cls_token"] = (
        stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.cls_token.data
    )
    state_dict["register_tokens"] = (
        stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.register_tokens.data
    )
    torch.save(state_dict, save_model_path)


def training_one_epoch_loop(
    stitched_model,
    feed_forward_model,
    optimizer,
    scheduler,
    train_dataloader,
    loss_criterion,
    epoch,
    wandb_logger=None,
):
    logger.info(f"Epoch {epoch} training started.")
    stitched_model.train()
    feed_forward_model.eval()

    iterable = (
        tqdm(train_dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=False)
        if dist_util.is_main_process()
        else train_dataloader
    )

    for step, data in enumerate(iterable):
        # ──────────────────────────────────────────────────────────────────────
        # 0. move data to GPU
        # ──────────────────────────────────────────────────────────────────────
        if dist.get_rank() == 0:
            # used_view_for_this_iter = random.choice([9, 13])
            used_view_for_this_iter = random.choice([9, 13, 17, 21])
        else:
            used_view_for_this_iter = 0

        tensor = torch.tensor([used_view_for_this_iter], device=dist_util.device())
        dist.broadcast(tensor, src=0)
        used_view_for_this_iter = tensor.item()

        vae_images = data["vae_image_tensor"].to(stitched_model.device)
        feedforward_img = data["feedforward_image_tensor"].to(stitched_model.device)
        vae_images = vae_images[:, :, :used_view_for_this_iter]
        feedforward_img = feedforward_img[:, :, :used_view_for_this_iter]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # ──────────────────────────────────────────────────────────────────────
            # 1. forward pass for stitched model
            # ──────────────────────────────────────────────────────────────────────
            stitched_out = stitched_model(
                images=vae_images,
                feedforward_image=feedforward_img,
                train=True,
            )

            # ──────────────────────────────────────────────────────────────────────
            # 2. forward pass for feed_forward_model
            # ──────────────────────────────────────────────────────────────────────
            with torch.no_grad():
                if isinstance(feed_forward_model, AnySplat):
                    ff_out = feed_forward_model.inference(
                        rearrange(
                            (feedforward_img + 1) * 0.5, "b c v h w -> b v c h w"
                        ),
                        training_for_stitching=True,
                    )

        # ──────────────────────────────────────────────────────────────────────
        # 3. loss
        # ──────────────────────────────────────────────────────────────────────
        loss = loss_criterion(stitched_out, ff_out)

        if isinstance(feed_forward_model, AnySplat):
            depth_loss = loss["depth_loss"]
            depth_loss_grad = loss["depth_loss_grad"]
            gaussian_mean_loss = loss["gaussian_mean_loss"]
            gaussian_covariance_loss = loss["gaussian_covariance_loss"]
            gaussian_harmonics_loss = loss["gaussian_harmonics_loss"]
            gaussian_opacity_loss = loss["gaussian_opacity_loss"]
            gaussian_scales_loss = loss["gaussian_scales_loss"]
            gaussian_rotations_loss = loss["gaussian_rotations_loss"]
            conf_loss = loss["conf_loss"]
            depth_conf_loss = loss["depth_conf_loss"]
            anchor_feat_loss = loss["anchor_feat_loss"]
            context_pose_extrinsic_loss = loss["context_pose_extrinsic_loss"]
            context_pose_intrinsic_loss = loss["context_pose_intrinsic_loss"]
            pred_pose_enc_list_loss = loss["pred_pose_enc_list_loss"]
            total_loss = loss["total_loss"]

        # ──────────────────────────────────────────────────────────────────────
        # 4. backpropagation step + logging
        # ──────────────────────────────────────────────────────────────────────
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in stitched_model.parameters() if p.requires_grad), 1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if wandb_logger is not None:
            log_dict = {
                "loss": total_loss.item(),
                "depth_loss": depth_loss.item(),
                "gaussian_mean_loss": gaussian_mean_loss.item(),
                "gaussian_covariance_loss": gaussian_covariance_loss.item(),
                "gaussian_harmonics_loss": gaussian_harmonics_loss.item(),
                "gaussian_opacity_loss": gaussian_opacity_loss.item(),
                "gaussian_scales_loss": gaussian_scales_loss.item(),
                "gaussian_rotations_loss": gaussian_rotations_loss.item(),
                "conf_loss": conf_loss.item(),
                "depth_conf_loss": depth_conf_loss.item(),
                "anchor_feat_loss": anchor_feat_loss.item(),
                "context_pose_extrinsic_loss": context_pose_extrinsic_loss.item(),
                "context_pose_intrinsic_loss": context_pose_intrinsic_loss.item(),
                "pred_pose_enc_list_loss": pred_pose_enc_list_loss.item(),
                "depth_loss_grad": depth_loss_grad.item(),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
            }
            wandb_logger.log(log_dict)


def main(args):
    device = torch_device_setup()

    # initalize model
    logger.info("Loading models...")
    ## feedforward_3d_model
    feed_forward_model = load_feedforward_model(args, device)
    ## video vae
    video_vae = load_vae(args, device)
    video_vae.eval()
    logger.info("Initializing stitching model")
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
    # parameter freeze
    for param in stitched_model.diffusion_vae.parameters():
        param.requires_grad = False
    for param in stitched_model.stitching_layer.parameters():
        param.requires_grad = True
    for param in stitched_model.stitched_3d_model.parameters():
        param.requires_grad = False
    logger.info("Adding LoRA to feedforward model.")
    add_lora(
        stitched_model.stitched_3d_model,
        target_modules=lora_cfg.target_modules,
        r=lora_cfg.r,
        alpha=lora_cfg.alpha,
        dropout=lora_cfg.dropout,
        fan_in_fan_out=lora_cfg.fan_in_fan_out,
    )
    mark_only_lora_as_trainable(stitched_model.stitched_3d_model, bias="lora_only")

    # enable special tokens training for anysplat in stitching model
    if isinstance(stitched_model.stitched_3d_model, AnySplatStitched):
        stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.register_tokens.requires_grad = True
        stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.cls_token.requires_grad = True
        stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.mask_token.requires_grad = True

    if args.resume_checkpoint_path is not None:
        logger.info(f"Resuming from checkpoint: {args.resume_checkpoint_path}")
        # checkpoint for optimizer and scheduler
        checkpoint = torch.load(
            os.path.join(args.resume_checkpoint_path, "resume_checkpoint.pth"),
            weights_only=False,
            map_location="cpu",
        )
        epoch_start = checkpoint["epoch"]

        model_state_dict = torch.load(
            os.path.join(
                args.resume_checkpoint_path, f"stitched_model_epoch_{epoch_start}.pth"
            ),
            weights_only=False,
            map_location="cpu",
        )
        epoch_start = epoch_start + 1  # to compensate for the first epoch
        stitched_model.stitched_3d_model.load_state_dict(
            model_state_dict["lora"], strict=False
        )
        stitched_model.stitching_layer.weight.data = model_state_dict[
            "stitching_layer"
        ]["weight"]
        stitched_model.stitching_layer.bias.data = model_state_dict["stitching_layer"][
            "bias"
        ]
        if isinstance(stitched_model.stitched_3d_model, AnySplatStitched):
            stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.mask_token.data = model_state_dict[
                "mask_token"
            ]
            stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.cls_token.data = model_state_dict[
                "cls_token"
            ]
            stitched_model.stitched_3d_model.encoder.aggregator.patch_embed.register_tokens.data = model_state_dict[
                "register_tokens"
            ]

    stitched_model.to(device)
    # cast to bfloat16
    cast_to_bfloat16(stitched_model)
    cast_to_bfloat16(feed_forward_model)

    stitched_model = DDP(
        stitched_model,
        find_unused_parameters=True,  # set False once you’re sure everything is used
    )

    train_dataloader, distributed_sampler = create_stitching_dataloader(
        args.dataset, args, augmentation=True
    )

    # optimizer setup
    logger.info("Setting up optimizer.")
    logger.info("configuration:Optimizer: AdamW")
    logger.info(f"configuration:Learning rate: {args.learning_rate}")
    logger.info(f"configuration:Weight decay: {args.weight_decay}")
    logger.info(f"configuration:Warmup steps: {args.warmup_steps}")

    optimizer = torch.optim.AdamW(
        (p for p in stitched_model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_dataloader))
    total_updates = args.num_epochs * updates_per_epoch
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_updates,
    )

    # Wandb logger
    if args.wandb_logging and dist.get_rank() == 0:
        logger.info("Setting up wandb logger.")
        import wandb

        wandb_logger = wandb.init(
            project=args.wandb_project_name, name=args.exp_name, config=args
        )
    else:
        wandb_logger = None

    # loss
    loss_criterion = TaskLossAnySplat()
    unwrapped = (
        stitched_model.module if isinstance(stitched_model, DDP) else stitched_model
    )
    unwrapped.stitched_3d_model.encoder.cfg.voxelize = False
    feed_forward_model.encoder.cfg.voxelize = False

    if args.resume_checkpoint_path is not None:
        logger.info(
            f"Resuming from checkpoint -> optimizer & scheduler: {args.resume_checkpoint_path}"
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        epoch_start = 0

    for epoch in range(epoch_start, args.num_epochs):
        # set epoch for distributed sampler
        distributed_sampler.set_epoch(epoch)
        # training loop
        training_one_epoch_loop(
            stitched_model,
            feed_forward_model,
            optimizer,
            scheduler,
            train_dataloader,
            loss_criterion,
            epoch,
            wandb_logger,
        )
        if dist_util.is_main_process():
            save_checkpoint(
                args.save_path,
                epoch,
                unwrapped,
                optimizer,
                scheduler,
                args,
            )


if __name__ == "__main__":
    args = stitching_training_argument().parse_args()
    main(args)
