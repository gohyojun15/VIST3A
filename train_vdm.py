import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.nn.functional as F
from accelerate import Accelerator, FullyShardedDataParallelPlugin, cpu_offload
from accelerate.utils import ProjectConfiguration
from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from einops import rearrange
from loguru import logger
from peft import LoraConfig, get_peft_model
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from tqdm import tqdm

import wandb
from data.dataset_util import create_vdm_tuning_dataloader
from evaluation.novel_view_synthesis_bench.nvs_eval import load_stitching_model
from utils.argument import training_vdm_argument
from utils.ema import EMAConfig, FSDPEMAWrapper
from utils.reward import calculate_reward, pickscore_mix2clip_loss_fn
from utils.wan_utils import AutoencoderKLWan


def save_checkpoint(output_dir, step, accelerator, pipeline, ema, optimizer):
    ckpt_dir = f"{output_dir}/checkpoints/checkpoint-{step}"
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # 1) LoRA
    lora_dir = os.path.join(ckpt_dir, "lora")
    if accelerator.is_main_process:
        os.makedirs(lora_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(pipeline.transformer).save_pretrained(
        lora_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(pipeline.transformer),
    )
    accelerator.wait_for_everyone()

    # 2) LoRA EMA (swap → save → swap back)
    lora_ema_dir = os.path.join(ckpt_dir, "lora_ema")
    if accelerator.is_main_process:
        os.makedirs(lora_ema_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    with torch.no_grad():
        ema.copy_ema_to(store_temp=True)
    accelerator.wait_for_everyone()
    accelerator.unwrap_model(pipeline.transformer).save_pretrained(
        lora_ema_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(pipeline.transformer),
    )
    accelerator.wait_for_everyone()
    with torch.no_grad():
        ema.copy_temp_to()
    accelerator.wait_for_everyone()

    # 3) Optim (DCP)
    optim_dir = os.path.join(ckpt_dir, "optim")
    if accelerator.is_main_process:
        os.makedirs(optim_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    state = {"optimizer": optimizer.state_dict()}
    dist_cp.save(
        state_dict=state,
        storage_writer=dist_cp.FileSystemWriter(optim_dir),
        planner=DefaultSavePlanner(),
    )
    accelerator.wait_for_everyone()

    # 4) meta
    if accelerator.is_main_process:
        torch.save({"step": step}, os.path.join(ckpt_dir, "meta.pt"))
    accelerator.wait_for_everyone()

    # 5) ema_shadow
    ema_shadow_dir = os.path.join(ckpt_dir, "ema_shadow")
    state = ema.dcp_state_dict()  # {"ema": Dict[str, Tensor]}
    dist_cp.save(
        state_dict=state,
        storage_writer=dist_cp.FileSystemWriter(ema_shadow_dir),
        planner=DefaultSavePlanner(),
    )
    accelerator.wait_for_everyone()


def choose_and_sync_two_indices(n: int, device="cuda") -> torch.Tensor:
    """
    returns: shape (2,) int64 tensor, same on all ranks
    """
    if not dist.is_initialized():
        return torch.randperm(n, device=device)[:2]

    rank = dist.get_rank()
    idx = torch.empty(2, dtype=torch.int64, device=device)

    if rank == 0:
        idx.copy_(torch.randperm(n, device=device)[:2])

    dist.broadcast(idx, src=0)
    return idx


def choose_and_sync_steps(low: int = 25, high: int = 50) -> int:
    """
    Chooses a random number of inference steps on rank 0 and broadcasts it to
    all other ranks so every process uses the same value.
    """
    if not dist.is_initialized():  # single-GPU / single-process run
        return random.randint(low, high)

    rank = dist.get_rank()
    # Rank 0 picks the number, others create a dummy placeholder
    steps_tensor = torch.tensor(
        [random.randint(low, high) if rank == 0 else 0],
        dtype=torch.int32,
        device="cuda",  # or whatever device you are already using
    )

    # Make rank 0’s value visible to everyone else
    dist.broadcast(steps_tensor, src=0)

    # Convert back to an int for normal Python use
    return int(steps_tensor.item())


def augment_camera_prompt(prompt: str):
    base = f"`{prompt}`"

    possible_prompts = [
        "base",
        # === 1. Pan (Horizontal Sweep) ===
        f"The camera pans smoothly from left to right across the scene: {base}. The horizontal motion reveals new spatial elements with each frame.",
        f"The camera performs a fast horizontal sweep, scanning the environment around the main subject: {base}.",
        f"A gentle left-to-right camera pan introduces the scene: {base}. The motion builds anticipation as more details appear.",
        f"The camera quickly pans from right to left, revealing the opposite side of the scene: {base}.",
        f"Pan the camera horizontally to uncover the subject and background in a fluid movement: {base}.",
        f"The camera moves in a slow panoramic motion across the horizon: {base}. This reveals a wide, cinematic field of view.",
        f"The camera performs a smooth 360° panoramic rotation around the scene: {base}. The motion fully encircles the environment.",
        # === 2. Orbit (Circular Rotation) ===
        f"The camera orbits around the main subject: {base}. This motion provides multiple perspectives of the central focus.",
        f"A circular orbit movement reveals all sides of the object in: {base}. The subject remains centered while the environment shifts naturally.",
        f"The camera rotates around the scene, maintaining constant distance: {base}. The orbiting trajectory captures 3D structure and consistency.",
        f"The camera performs a full circular path, orbiting around the main focus: {base}.",
        f"The camera glides around the subject in a semicircular arc, showing it from both front and side views: {base}.",
        # === 3. Dolly / Push / Pull ===
        f"The camera dollies inward toward the subject: {base}. The forward motion increases immersion and depth.",
        f"A slow dolly-out reveals the full environment behind the subject: {base}.",
        f"The camera pushes forward into the center of the scene: {base}. This close approach emphasizes detail and perspective.",
        f"The camera pulls backward from the subject, gradually exposing the surrounding world: {base}.",
        f"A dolly-in transition draws attention to the main object in: {base}. The camera motion builds intensity and focus.",
        # === 4. Zoom (Optical Scaling) ===
        f"The camera zooms in slowly to magnify the central details of: {base}.",
        f"The camera performs a fast zoom-out to show the full 3D layout of: {base}.",
        f"A gentle zoom-in enhances focus on the core region of: {base}.",
        f"Zoom the camera lens steadily to emphasize the subject in: {base}.",
        f"The camera zooms out gradually from a close-up view, unveiling the complete composition: {base}.",
        # === 5. Tilt (Vertical Rotation) ===
        f"The camera tilts upward from the base to the sky: {base}. The vertical movement highlights height and scale.",
        f"The camera tilts downward toward the ground: {base}. This viewpoint emphasizes spatial grounding.",
        f"A smooth upward tilt reveals tall architectural structures in: {base}.",
        f"The camera performs a vertical sweep from top to bottom: {base}. The tilt motion enriches the perception of depth.",
        f"The camera tilts slightly while maintaining focus on the subject: {base}.",
        # === 6. Fly-through / Aerial ===
        f"The camera flies smoothly through the 3D environment: {base}. The flight motion provides a sense of exploration.",
        f"The camera glides like a drone over the terrain: {base}. The aerial trajectory emphasizes continuity and scale.",
        f"The camera flies low across the scene: {base}. The close pass accentuates ground details and parallax.",
        f"The camera navigates through narrow spaces in: {base}. It moves dynamically, avoiding obstacles.",
        f"A cinematic fly-through motion traverses the environment: {base}. The continuous travel conveys immersion.",
        # === 7. Arc (Curved Path) ===
        f"The camera moves along a curved arc around the subject: {base}. The motion reveals both profile and depth.",
        f"A smooth arc path captures the subject from multiple diagonal angles: {base}.",
        f"The camera glides through an arc trajectory at mid-height: {base}.",
        f"The arc-shaped movement maintains focus on the central point while changing background parallax: {base}.",
        f"The camera performs a half-orbit arc, revealing the subject's side and back view: {base}.",
        # === 8. Spiral ===
        f"The camera spirals upward around the object: {base}. The motion combines rotation and elevation.",
        f"The camera follows a helical path circling the subject: {base}.",
        f"A downward spiral descends smoothly toward the scene center: {base}.",
        f"The camera performs a spiral ascent around the 3D environment: {base}.",
        f"A slow, tightening spiral focuses attention on the subject at the core: {base}.",
        # === 9. Tracking / Follow ===
        f"The camera tracks a moving subject through the space: {base}. The perspective stays consistent during motion.",
        f"A tracking shot keeps the subject centered as it moves dynamically through: {base}.",
        f"The camera follows the target’s trajectory with cinematic smoothness: {base}.",
        f"A continuous tracking motion moves alongside the subject: {base}.",
        f"The camera mirrors the subject’s motion path, maintaining constant distance: {base}.",
        # === 10. Crane / Lift ===
        f"The camera rises vertically like a crane shot: {base}. The elevation change provides an aerial overview.",
        f"A slow crane movement lowers the camera toward the scene: {base}.",
        f"The camera lifts steadily upward from ground level: {base}. The ascending motion reveals overall spatial layout.",
        f"A crane motion elevates the viewpoint to a higher perspective: {base}.",
        f"The camera descends smoothly back down to focus on details: {base}.",
        # === 11. Rotation-in-place ===
        f"The camera rotates 360° around its axis at a fixed point: {base}.",
        f"A stationary spin reveals every direction of the surrounding scene: {base}.",
        f"The camera performs a slow turn-in-place while keeping balance: {base}.",
        f"A gentle rotational sweep captures panoramic surroundings of: {base}.",
        f"The camera spins steadily to record all angles of the subject: {base}.",
        # === 12. Handheld / Natural ===
        f"The camera captures {base} with a subtle handheld feel, adding realism and intimacy.",
        f"A natural, slightly shaky handheld motion records: {base}.",
        f"The handheld camera follows the subject closely, simulating human perspective: {base}.",
        f"The shot feels organic, as if captured by a person exploring: {base}.",
        f"The handheld style gives {base} a dynamic and lifelike tone.",
        # === 13. Composite / Complex ===
        f"The camera starts with a dolly-in and transitions to a circular orbit: {base}.",
        f"A horizontal pan merges into a tilt-up movement: {base}.",
        f"The motion begins as a zoom-in, then arcs around the object: {base}.",
        f"The camera begins with a fly-through and ends with a spiral ascent: {base}.",
        f"A dolly-out ends with a 360° in-place rotation: {base}.",
        # === 14. Temporal / Speed cues ===
        f"The camera slowly accelerates over time while capturing: {base}.",
        f"A rapid start transitions into a steady glide through the scene: {base}.",
        f"The motion starts slowly, then speeds up near the subject: {base}.",
        f"The camera eases in at the start, then gently slows as it completes the movement: {base}.",
        f"The motion evolves gradually during the sequence: {base}.",
        # === 15. Emotional / Cinematic tone ===
        f"The camera glides gracefully with cinematic smoothness across: {base}.",
        f"A dramatic sweeping camera move emphasizes the grandeur of: {base}.",
        f"The slow, contemplative camera motion captures the serene atmosphere of: {base}.",
        f"A dynamic, energetic camera movement enhances the intensity of: {base}.",
        f"A suspenseful tracking motion builds tension throughout: {base}.",
        # === 16. Experimental / Creative ===
        f"The camera rolls diagonally while approaching the scene: {base}.",
        f"The camera oscillates subtly, mimicking breathing motion: {base}.",
        f"A free-floating camera drifts unpredictably through: {base}.",
        f"The shot involves alternating zoom and pan motions to emphasize rhythm: {base}.",
        f"The camera performs a parallax sweep that dynamically layers depth: {base}.",
    ]

    return random.choice(possible_prompts)


@torch.no_grad()
def compute_wan_text_embeddings(
    prompt, text_encoders, tokenizers, max_sequence_length=226, device=None
):
    dtype = text_encoders.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    # batch_size = len(prompt)

    text_inputs = tokenizers(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoders(
        text_input_ids.to(device), mask.to(device)
    ).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds


def main(args):
    output_dir = args.save_path
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(output_dir),
        automatic_checkpoint_naming=True,
    )
    loggers = ["wandb"] if args.wandb_logging else None
    mp = {
        "param_dtype": torch.bfloat16,
        "reduce_dtype": torch.bfloat16,
        "output_dtype": torch.bfloat16,
    }
    fsdp2_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["WanTransformerBlock"],
        state_dict_type="SHARDED_STATE_DICT",
        mixed_precision_policy=mp,
        use_orig_params=True,
        reshard_after_forward=True,
    )
    accelerator = Accelerator(
        log_with=loggers,
        mixed_precision="bf16",
        project_config=accelerator_config,
        fsdp_plugin=fsdp2_plugin,
    )

    if args.wandb_logging and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project_name,
            config=args,
            init_kwargs={"wandb": {"name": output_dir}},
        )

    ########################
    # Building models
    ########################
    logger.info("Building models...")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae.enable_gradient_checkpointing()
    vae.config.scale_factor_temporal = 4
    vae.config.scale_factor_spatial = 8

    pipeline = WanPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=3.0,
    )
    pipeline.scheduler = scheduler

    pipeline.transformer.to(dtype=torch.bfloat16, device=accelerator.device)
    pipeline.vae.to(device=accelerator.device)
    pipeline.vae.eval()
    # pipeline.text_encoder.to(dtype=torch.bfloat16, device="cpu")
    pipeline.text_encoder.to(dtype=torch.bfloat16, device=accelerator.device)
    pipeline.text_encoder.eval()
    # pipeline.text_encoder = cpu_offload(
    #     pipeline.text_encoder, execution_device=accelerator.device
    # )

    if args.resume_checkpoint_path is not None and os.path.isdir(
        args.resume_checkpoint_path
    ):
        # resuming from a LoRA checkpoint
        from peft import PeftModel

        lora_ckpt_dir = os.path.join(args.resume_checkpoint_path, "lora")
        logger.info(f"Loading LoRA adapter weights from {lora_ckpt_dir}")
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer,
            lora_ckpt_dir,
            is_trainable=True,
        )
    else:
        # training with lora
        modules = [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
        ]
        transformer_lora_config = LoraConfig(
            r=8,  # 32
            lora_alpha=16,  # 64
            init_lora_weights=True,
            target_modules=modules,
        )
        pipeline.transformer = get_peft_model(
            pipeline.transformer, transformer_lora_config
        )

    pipeline.transformer.enable_gradient_checkpointing()
    pipeline.transformer.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        params=[p for p in pipeline.transformer.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Load stitched decoder
    logger.info("loading stitched decoder")
    stitched_decoder = load_stitching_model(
        args
    ).eval()  # gradient checkpointing already applied.

    text_dataloader, text_sampler, dl3dv_dataloader, dl3dv_sampler = (
        create_vdm_tuning_dataloader(
            datasets=[("text", Path(args.text_dataset_path))] + args.dataset,
            args=args,
        )
    )
    with torch.no_grad():
        neg_prompt_embed = compute_wan_text_embeddings(
            [""],
            text_encoders=pipeline.text_encoder,
            tokenizers=pipeline.tokenizer,
            max_sequence_length=226,
            device=accelerator.device,
        )

    logger.info("Preparing everything with accelerator...")
    (
        pipeline.transformer,
        optimizer,
        text_dataloader,
        dl3dv_dataloader,
    ) = accelerator.prepare(
        pipeline.transformer,
        optimizer,
        text_dataloader,
        dl3dv_dataloader,
    )

    ema = FSDPEMAWrapper(
        pipeline.transformer,
        cfg=EMAConfig(decay=0.99, update_step_interval=1, ema_dtype=torch.float32),
        only_trainable=True,
    )

    if args.resume_checkpoint_path is not None and os.path.isdir(
        args.resume_checkpoint_path
    ):
        logger.info(f"Resuming EMA from {args.resume_checkpoint_path}...")
        ema_shadow_dir = os.path.join(args.resume_checkpoint_path, "ema_shadow")
        state = ema.dcp_state_dict()
        dist_cp.load(
            state_dict=state,
            storage_reader=dist_cp.FileSystemReader(ema_shadow_dir),
            planner=DefaultLoadPlanner(),
        )

        logger.info(f"Resuming optimizer from {args.resume_checkpoint_path}...")
        state = {"optimizer": optimizer.state_dict()}
        optim_dir = os.path.join(args.resume_checkpoint_path, "optim")
        dist_cp.load(
            state_dict=state,
            storage_reader=dist_cp.FileSystemReader(optim_dir),
            planner=DefaultLoadPlanner(),
        )
        optimizer.load_state_dict(state["optimizer"])

        meta_path = os.path.join(args.resume_checkpoint_path, "meta.pt")
        meta = torch.load(meta_path, map_location="cpu")
        step = meta["step"]
        logger.info(f"Resuming from step {step}...")
    else:
        step = 0

    loss_fn = pickscore_mix2clip_loss_fn(
        inference_dtype=torch.bfloat16,
        device=accelerator.device,
        pickscore_weight=args.qual_coeff,
        peclip_weight=args.qual_coeff,
    )

    text_len = len(text_dataloader)
    skip_text = step % text_len
    if skip_text > 0:
        logger.info(f"Skipping {skip_text} text batches...")
        text_dataloader = accelerator.skip_first_batches(text_dataloader, skip_text)
        text_epoch = step // text_len
    else:
        text_epoch = 0
    video_len = len(dl3dv_dataloader)
    skip_video = step % video_len
    if skip_video > 0:
        logger.info(f"Skipping {skip_video} video batches...")
        dl3dv_dataloader = accelerator.skip_first_batches(dl3dv_dataloader, skip_video)
        dl3dv_epoch = step // video_len
    else:
        dl3dv_epoch = 0
    dl3dv_iter = iter(dl3dv_dataloader)
    pipeline.transformer.train()

    logger.info(f"dl3dv_epoch: {dl3dv_epoch}, text_epoch: {text_epoch}")
    dl3dv_dataloader.set_epoch(dl3dv_epoch)
    text_dataloader.set_epoch(text_epoch)

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(accelerator.device, torch.float32)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(accelerator.device, torch.float32)
    trainable_params = [p for p in pipeline.transformer.parameters() if p.requires_grad]

    for _ in range(1000000):
        for batch in text_dataloader:
            if step % args.save_freq == 0 and step > 0:
                save_checkpoint(output_dir, step, accelerator, pipeline, ema, optimizer)

            prompt = batch[0]
            print("Current prompt:", prompt)
            try:
                video_batch = next(dl3dv_iter)
            except StopIteration:
                dl3dv_dataloader.set_epoch(dl3dv_epoch)
                dl3dv_iter = iter(dl3dv_dataloader)
                video_batch = next(dl3dv_iter)
                dl3dv_epoch += 1

            with accelerator.autocast():
                # Conditioning text embeddings
                with torch.no_grad():
                    if args.enable_rl:
                        prompt_embeds = compute_wan_text_embeddings(
                            augment_camera_prompt(prompt),
                            text_encoders=pipeline.text_encoder,
                            tokenizers=pipeline.tokenizer,
                            max_sequence_length=226,
                            device=accelerator.device,
                        )
                    sft_prompt = compute_wan_text_embeddings(
                        video_batch["caption"],
                        text_encoders=pipeline.text_encoder,
                        tokenizers=pipeline.tokenizer,
                        max_sequence_length=226,
                        device=accelerator.device,
                    )
                # VAE encode and target
                with torch.no_grad():
                    z_0 = vae.encode(
                        video_batch["image_tensor"].to(accelerator.device)
                    ).latent_dist.sample()
                    z_0 = (z_0.float() - latents_mean) * latents_std
                    eps = torch.randn_like(
                        z_0, device=accelerator.device, dtype=torch.float32
                    )
                    sigma = torch.rand(
                        z_0.shape[:1], device=accelerator.device, dtype=torch.float32
                    )  # * 0.4 + 0.6
                    sigma_bcthw = sigma[:, None, None, None, None]
                    z_sigma = (1 - sigma_bcthw) * z_0 + sigma_bcthw * eps
                    target = eps - z_0

                pred = pipeline.transformer(
                    hidden_states=z_sigma.detach(),
                    timestep=(sigma * 1000).float(),
                    encoder_hidden_states=sft_prompt,
                    return_dict=False,
                )[0]
            diffusion_loss = torch.nn.functional.mse_loss(pred.float(), target.float())

            # roll out reward branch
            if args.enable_rl:
                num_inference_steps = choose_and_sync_steps(low=10, high=50)
                if step % 10 == 0:
                    num_inference_steps = 50  # for logging clarity
                pipeline.scheduler.set_timesteps(
                    num_inference_steps, device=accelerator.device
                )
                timesteps = pipeline.scheduler.timesteps

                latents = torch.randn(
                    [1] + list(z_0.shape[1:]),
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                guidance_scale = random.random() * 2 + 4.0  #  # 4.0 ~ 6.0
                t_train_idx = choose_and_sync_two_indices(
                    len(timesteps), device=accelerator.device
                )
                t_train = timesteps[t_train_idx]

                for i, t in tqdm(enumerate(timesteps)):
                    with accelerator.autocast():
                        backprop = t.item() in t_train
                        # if the last step, always backprop
                        if i == len(timesteps) - 1:
                            backprop = True
                        z_in = torch.cat([latents, latents], dim=0)
                        t_in = torch.stack([t] * 2, dim=0)
                        prompt_embeds_in = torch.cat(
                            [prompt_embeds, neg_prompt_embed], dim=0
                        )
                        if backprop:
                            pred = pipeline.transformer(
                                hidden_states=z_in.detach(),
                                timestep=t_in,
                                encoder_hidden_states=prompt_embeds_in,
                                return_dict=False,
                            )[0]
                            noise_pred, noise_uncond = pred.chunk(2, dim=0)
                            noise_pred = noise_uncond + guidance_scale * (
                                noise_pred - noise_uncond
                            )
                        else:
                            with torch.no_grad():
                                pred = pipeline.transformer(
                                    hidden_states=z_in.detach(),
                                    timestep=t_in,
                                    encoder_hidden_states=prompt_embeds_in,
                                    return_dict=False,
                                )[0]
                                noise_pred = pred.chunk(2, dim=0)
                                noise_pred, noise_uncond = pred.chunk(2, dim=0)
                                noise_pred = noise_uncond + guidance_scale * (
                                    noise_pred - noise_uncond
                                )
                    latents = pipeline.scheduler.step(
                        noise_pred.float(), t, latents.float(), return_dict=False
                    )[0]
                latents = latents / latents_std + latents_mean
                video = pipeline.vae.decode(latents, return_dict=False)[0]

                reward = calculate_reward(
                    latents, video, stitched_decoder, loss_fn, prompt, accelerator
                )
                reward_loss = reward[0]

                img_list_decode = reward[1][0]
                img_list_render = reward[1][1]
            else:
                reward_loss = 0

            total_loss = diffusion_loss + reward_loss

            accelerator.backward(total_loss)

            grad_norm = accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                ema.step(step)

            #### Logging
            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "total_loss": total_loss.mean().detach().cpu().item(),
                        "diffusion_loss": diffusion_loss.mean().detach().cpu().item(),
                        "pick_score_loss": reward_loss.mean().detach().cpu().item()
                        if args.enable_rl
                        else 0.0,
                        "grad_norm_2": grad_norm.detach().cpu().item(),
                    },
                    step=step,
                )
                if args.enable_rl:
                    if step % 10 == 0:
                        images_log = torch.cat(
                            [
                                F.interpolate(
                                    rearrange(img_list_decode, "b h w c -> b c h w"),
                                    size=(448, 448),
                                    mode="bilinear",
                                ),
                                F.interpolate(
                                    rearrange(img_list_render, "b h w c -> b c h w"),
                                    size=(448, 448),
                                    mode="bilinear",
                                ),
                            ],
                            dim=0,
                        )

                        images_to_log = []
                        for i in range(images_log.shape[0]):
                            x = images_log[i].detach().float().cpu()
                            x = x.permute(1, 2, 0).numpy()
                            images_to_log.append(wandb.Image(x, caption=f"{i}"))

                        accelerator.log({"images": images_to_log}, step=step)
            step += 1
        text_epoch += 1
        text_dataloader.set_epoch(text_epoch)


if __name__ == "__main__":
    args = training_vdm_argument().parse_args()
    main(args)
