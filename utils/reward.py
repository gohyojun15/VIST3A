import open_clip
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from transformers import AutoModel, AutoProcessor


def pickscore_mix2clip_loss_fn(
    inference_dtype=None,
    device=None,
    pe_model_id="hf-hub:apple/DFN5B-CLIP-ViT-H-14-378",
    pickscore_weight=0.25,
    peclip_weight=0.25,
    add_pickscore_suffix=False,
    pickscore_target=1.0,
    pickscore_div=100.0,
    cache_dir=None,
):
    """
    PickScore + (optional) another CLIP (default: apple/DFN5B-CLIP-ViT-H-14-378) mixed reward loss.

    Args:
        inference_dtype: torch.dtype
        device: torch.device
        pe_model_id: open_clip model id (hf-hub:apple/DFN5B-CLIP-ViT-H-14-378)
        pickscore_weight: weight for PickScore loss term
        peclip_weight: weight for PE-CLIP loss term
        add_pickscore_suffix: whether to append realism suffix to prompts for pickscore branch
        pickscore_target: target value for pickscore loss (1.0 means try to make score ~= pickscore_div)
        pickscore_div: divisor used to scale pickscore (your original uses /100)
        cache_dir: open_clip cache_dir

    Returns:
        loss_fn(im_pix_un, prompts) -> (loss, mixed_score, dict_of_scores)
    """

    # -----------------------
    # 1) PickScore branch
    # -----------------------
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    pick_processor = AutoProcessor.from_pretrained(
        processor_name_or_path, torch_dtype=inference_dtype
    )
    pick_model = (
        AutoModel.from_pretrained(
            model_pretrained_name_or_path, torch_dtype=inference_dtype
        )
        .eval()
        .to(device)
    )
    pick_model.requires_grad_(False)
    pick_model._set_gradient_checkpointing(True)

    # pickscore normalization (CLIP mean/std)
    pick_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
    pick_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    # reproduce your pickscore preprocessing as-is
    def _pickscore_preprocess(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        im_pix = im_pix * 255.0

        if im_pix.shape[2] < im_pix.shape[3]:
            height = 224
            width = im_pix.shape[3] * height // im_pix.shape[2]
        else:
            width = 224
            height = im_pix.shape[2] * width // im_pix.shape[3]

        im_pix = torchvision.transforms.Resize(
            (height, width),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True,
        )(im_pix)

        im_pix = im_pix.permute(0, 2, 3, 1)  # BCHW -> BHWC

        startx = width // 2 - 112
        starty = height // 2 - 112
        im_pix = im_pix[:, starty : starty + 224, startx : startx + 224, :]

        im_pix = im_pix * 0.00392156862745098  # /255
        im_pix = (im_pix - pick_mean) / pick_std
        im_pix = im_pix.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return im_pix

    # -----------------------
    # 2) PE-CLIP branch (open_clip)
    # -----------------------
    pe_clip_model, _, _ = open_clip.create_model_and_transforms(
        pe_model_id, cache_dir=cache_dir
    )
    pe_clip_tokenizer = open_clip.get_tokenizer(pe_model_id)
    pe_clip_model.eval()
    pe_clip_model.set_grad_checkpointing(True)
    pe_clip_model.to(device, dtype=inference_dtype)

    # DFN5B default image size is 378 in your HPS code
    pe_target_size = (378, 378)
    pe_normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    def _peclip_preprocess(im_pix_un):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1)
        x = torchvision.transforms.Resize(pe_target_size)(im_pix)
        x = pe_normalize(x).to(im_pix.dtype)
        return x

    # -----------------------
    # 3) Mixed loss_fn
    # -----------------------
    def loss_fn(im_pix_un, prompts):
        """
        Args:
            im_pix_un: (B,C,H,W) in [-1,1]
            prompts: list[str] length B
        Returns:
            loss: scalar
            mixed_score: scalar (weighted sum of each branch score)
            scores: dict with keys pickscore_raw, pickscore_scaled, peclip_score
        """
        B = im_pix_un.shape[0]

        # ---- PickScore score ----
        pick_im = _pickscore_preprocess(im_pix_un)

        if add_pickscore_suffix:
            pick_prompts = [
                p + ", realistic, photorealistic, hyper-realistic, naturalistic"
                for p in prompts
            ]
        else:
            pick_prompts = prompts

        pick_text_inputs = pick_processor(
            text=pick_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        # PickScore embeddings
        pick_image_embs = pick_model.get_image_features(pixel_values=pick_im)
        pick_image_embs = pick_image_embs / torch.norm(
            pick_image_embs, dim=-1, keepdim=True
        )
        with torch.no_grad():
            pick_text_embs = pick_model.get_text_features(**pick_text_inputs)
            pick_text_embs = pick_text_embs / torch.norm(
                pick_text_embs, dim=-1, keepdim=True
            )

        # score: (B,B) then diag
        pick_logits = pick_model.logit_scale.exp() * (
            pick_text_embs @ pick_image_embs.T
        )
        pick_diag = torch.diagonal(pick_logits)  # (B,)
        pick_score_scaled = pick_diag / pickscore_div  # roughly ~[0,1] target

        # loss term: match target (default 1.0)
        pick_loss = (pickscore_target - pick_score_scaled).abs().mean()

        # ---- PE-CLIP score ----
        pe_im = _peclip_preprocess(im_pix_un)
        pe_text = pe_clip_tokenizer(pick_prompts).to(device)

        pe_img_feat = pe_clip_model.encode_image(pe_im, normalize=True)
        with torch.no_grad():
            pe_txt_feat = pe_clip_model.encode_text(pe_text, normalize=True)

        pe_logits = pe_img_feat @ pe_txt_feat.T
        pe_diag = torch.diagonal(pe_logits)  # cosine similarity (B,)
        pe_loss = (1.0 - pe_diag).mean()

        # ---- Mix ----
        loss = pickscore_weight * pick_loss + peclip_weight * pe_loss

        mixed_score = (
            pickscore_weight * pick_score_scaled.mean() + peclip_weight * pe_diag.mean()
        )

        scores = {
            "pickscore_raw": pick_diag.mean().detach(),
            "pickscore_scaled": pick_score_scaled.mean().detach(),
            "peclip_score": pe_diag.mean().detach(),
        }
        return loss, mixed_score, scores

    return loss_fn


def calculate_reward(
    gen_latents, videos, stitched_decoder, loss_fn, prompt, accelerator
):
    for b in range(gen_latents.shape[0]):
        feedforward_image = videos[b].unsqueeze(0)
        feedforward_image = F.interpolate(
            feedforward_image,
            size=(feedforward_image.shape[2], 448, 448),
            mode="trilinear",
            align_corners=True,
        )

        with accelerator.autocast():
            output = stitched_decoder.forward_with_latent(
                gen_latents, feedforward_image=feedforward_image.to(dtype=torch.float32)
            )
        gaussians, pred_context_pose = output.gaussians, output.pred_context_pose
        renderer = stitched_decoder.stitched_3d_model.decoder

        num_frames = 13
        random_index_img = torch.randperm(pred_context_pose["intrinsic"][0].shape[0])[
            :num_frames
        ]

        target_intrinsics = pred_context_pose["intrinsic"][0][random_index_img]
        target_extrinsics = pred_context_pose["extrinsic"][0][random_index_img]

        rendered_image = renderer.forward(
            gaussians,
            target_extrinsics.float().unsqueeze(0),
            target_intrinsics.float().unsqueeze(0),
            torch.ones(1, num_frames, device=target_extrinsics.device) * 0.1,
            torch.ones(1, num_frames, device=target_extrinsics.device) * 100,
            (448, 448),
        )
        rendered_video, _ = (
            rendered_image.color[0].clip(min=0, max=1),
            rendered_image.depth[0],
        )
        rendered_video_scaled = rendered_video * 2 - 1  # B C H W
        with accelerator.autocast():
            rendered_pick = loss_fn(
                rendered_video_scaled, prompt * len(random_index_img)
            )
        num_frames = 1
        random_index_gt_img = torch.randint(
            0, pred_context_pose["intrinsic"][0].shape[0], (num_frames,)
        )
        decoded_video = feedforward_image[:, :, random_index_gt_img][0].permute(
            1, 0, 2, 3
        )  # B C H W
        with accelerator.autocast():
            pick_video = loss_fn(decoded_video, prompt * len(random_index_gt_img))
        pick_score = rendered_pick[0] + pick_video[0]
        img_list = [
            rearrange((decoded_video + 1) / 2, "b c h w -> b h w c"),
            rearrange((rendered_video_scaled + 1) / 2, "b c h w -> b h w c"),
        ]
    return pick_score, img_list
