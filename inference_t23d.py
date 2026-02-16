import os
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from diffusers import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from einops import rearrange
from loguru import logger
from peft import PeftModel
from pytorch_lightning import seed_everything
from tqdm import tqdm

from evaluation.novel_view_synthesis_bench.nvs_eval import load_stitching_model
from third_party_model.anysplat.src.misc.image_io import save_interpolated_video
from third_party_model.anysplat.src.model.ply_export import export_ply
from utils.argument import inference_vist3a_argument
from utils.dist_util import setup_dist


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def main(args):
    # setup distributed environment
    setup_dist()
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    with open(args.input_texts_path, "r") as f:
        prompt_list = [line.strip() for line in f.readlines()]

    # split prompt list into ranks
    prompt_list = prompt_list[rank :: dist.get_world_size()]
    seed_everything(12413)

    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        num_train_timesteps=1000,
        use_flow_sigmas=True,
        flow_shift=args.flow_shift,
    )

    # load model
    pipe = WanPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        args.transformer_lora_path,
    )
    pipe.transformer.eval()
    pipe.scheduler = scheduler
    pipe.to(device)

    # load decoder
    stitched_decoder = load_stitching_model(args).eval()

    for qq, prompt in tqdm(enumerate(prompt_list)):
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                prompt_input = f"The camera rotates around the scene, maintaining constant distance: `{prompt}`. The orbiting trajectory captures 3D structure and consistency."

                negative_prompt = [
                    "Background blur, Blurred background, Blurred scene, Artifacts, not aesthetic, not realistic, rendered noise, low quality movement, low quality video, low quality image, deformed, disfigured, distorted, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted legs, fused fingers, too many fingers, long neck"
                ]
                # negative_prompt = [""]
                outputs = pipe(
                    prompt=prompt_input,
                    width=512,
                    height=512,
                    negative_prompt=negative_prompt,
                    num_frames=args.num_frames,
                    num_inference_steps=50,
                    guidance_scale=float(args.cfg_scale),
                    output_type="latent",
                )
                latents = outputs["frames"]  # .to(vae.dtype)
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                    1, pipe.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                samples = pipe.vae.decode(latents, return_dict=False)[0]

        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                feedforward = F.interpolate(
                    samples,
                    (samples.shape[2], 448, 448),
                    mode="trilinear",
                    align_corners=False,
                )

        cur_save_path = f"{args.output_dir}/{prompt[:100].replace('/', '')}/"
        os.makedirs(cur_save_path)
        with open(cur_save_path + "prompt.txt", "w") as f:
            f.write(prompt)
        logger.info(f"video and prompt are saved: {cur_save_path}")

        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                output = stitched_decoder.forward_with_latent(
                    latents.to(stitched_decoder.device),
                    feedforward_image=feedforward.to(stitched_decoder.device),
                    train=False,
                )
            gaussians, pred_context_pose, last_pred_pose_enc = (
                output.gaussians,
                output.pred_context_pose,
                output.last_pred_pose_enc,
            )
            pred_all_extrinsic = pred_context_pose["extrinsic"]
            pred_all_intrinsic = pred_context_pose["intrinsic"]
            with torch.no_grad():
                video, depth_colored = save_interpolated_video(
                    pred_all_extrinsic,
                    pred_all_intrinsic,
                    1,
                    448,
                    448,
                    gaussians,
                    cur_save_path,
                    stitched_decoder.stitched_3d_model.decoder,
                )
            plyfile = os.path.join(cur_save_path, "gaussians.ply")
            export_ply(
                gaussians.means[0],
                gaussians.scales[0],
                gaussians.rotations[0],
                gaussians.harmonics[0],
                gaussians.opacities[0],
                Path(plyfile),
                # save_sh_dc_only=True,
                save_sh_dc_only=True,
            )


if __name__ == "__main__":
    args = inference_vist3a_argument().parse_args()
    main(args)
