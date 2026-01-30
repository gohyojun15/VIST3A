import argparse
import math
import os

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan
from einops import rearrange
from loguru import logger
from tqdm import tqdm

from data.dataset_util import create_stitching_dataloader
from third_party_model.anysplat.src.model.model.anysplat import AnySplat
from utils.argument import find_layer_stitching_argument
from utils.dist_util import torch_device_setup
from utils.utils_for_thirdparty import load_feedforward_model, load_vae


@torch.no_grad()
def unfold3d(x, kernel_size, stride, padding, dilation=(1, 1, 1)):
    """Return a view of every sliding kT×kH×kW block as a separate axis.

    x             : (N, C, T, H, W)  – contiguous or channels-first stride
    kernel_size   : (kT, kH, kW)
    stride        : (sT, sH, sW)
    padding       : (pT, pH, pW)
    dilation      : (dT, dH, dW)
    result shape  : (N, C, kT, kH, kW, T_out, H_out, W_out)
    """
    kT, kH, kW = kernel_size
    sT, sH, sW = stride
    pT, pH, pW = padding
    dT, dH, dW = dilation

    x = torch.nn.functional.pad(x, (pW, pW, pH, pH, pT, pT))

    N, C, T, H, W = x.shape
    To = (T - (dT * (kT - 1) + 1)) // sT + 1
    Ho = (H - (dH * (kH - 1) + 1)) // sH + 1
    Wo = (W - (dW * (kW - 1) + 1)) // sW + 1

    sN, sC, sT0, sH0, sW0 = x.stride()

    return x.as_strided(
        size=(N, C, kT, kH, kW, To, Ho, Wo),
        stride=(sN, sC, sT0 * dT, sH0 * dH, sW0 * dW, sT0 * sT, sH0 * sH, sW0 * sW),
    )


@torch.no_grad()
def fit_conv3d_streaming(
    conv,
    Z_batches,  # tensors (B, C_in, F, H, W)
    Y_batches,  # tensors (B, C_out, F_y, H_y, W_y)
    ridge=1e-4,
    dtype=torch.float64,
):
    """
    Solve (XᵀX + λI)W = XᵀY  without ever instantiating X.
    Result is written directly into `conv.weight` (+ bias if present).
    """
    device = conv.weight.device
    kT, kH, kW = conv.kernel_size
    C_in = conv.in_channels
    C_out = conv.out_channels
    d = C_in * kT * kH * kW  # cols of X

    # Accumulate XᵀX and XᵀY
    XtX = torch.zeros(d, d, device=device, dtype=dtype)
    XtY = torch.zeros(d, C_out, device=device, dtype=dtype)
    n_rows = 0  # for bias later

    def upsampling_layer(x):
        T_vae = x.shape[2]
        T_original = (T_vae - 1) * 4 + 1
        x = nn.functional.interpolate(
            x,
            size=[T_original, Z_batches.shape[3], Z_batches.shape[4]],
            mode="trilinear",
            align_corners=True,
        )
        return x

    for Z_b, Y_b in tqdm(zip(Z_batches, Y_batches)):
        Z_b = Z_b.to(device, non_blocking=True, dtype=dtype)  # C, T, H, W
        Y_b = Y_b.to(device, non_blocking=True, dtype=dtype)  # C, T, H, W
        Z_b = Z_b.unsqueeze(0)
        Z_b = upsampling_layer(Z_b)  # 1, C, T, H, W
        Y_b = rearrange(
            Y_b,
            "T (H W) D -> D T H W",
            H=int(math.sqrt(Y_b.shape[1])),
            W=int(math.sqrt(Y_b.shape[1])),
        ).unsqueeze(0)  # 1, C_out, T_out, H_out, W_out

        patches = unfold3d(
            Z_b,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
        )
        # X_b : (B·T_out·H_out·W_out, d)
        X_b = patches.permute(0, 5, 6, 7, 1, 2, 3, 4).reshape(-1, d).to(dtype)
        # Y_b_vec : (same rows, C_out)
        Y_b_vec = Y_b.permute(0, 2, 3, 4, 1).reshape(-1, C_out).to(dtype)

        XtX += X_b.T @ X_b
        XtY += X_b.T @ Y_b_vec
        n_rows += X_b.shape[0]

        del Z_b, Y_b, patches, X_b, Y_b_vec
        torch.cuda.empty_cache()

    # Ridge ⇒ add λI
    XtX.diagonal().add_(ridge)
    W = torch.linalg.solve(XtX, XtY)  # (d, C_out)
    conv.weight.copy_(W.T.reshape(C_out, C_in, kT, kH, kW).to(conv.weight.dtype))

    # bias = mean residual
    if conv.bias is not None:
        sum_residual = torch.zeros(C_out, device=device, dtype=dtype)

        # second pass cheaper: residual uses matrix–vector product only
        for Z_b, Y_b in tqdm(zip(Z_batches, Y_batches)):
            Z_b = Z_b.to(device, non_blocking=True, dtype=dtype)  # C, T, H, W
            Y_b = Y_b.to(device, non_blocking=True, dtype=dtype)  # C, T, H, W
            Z_b = Z_b.unsqueeze(0)
            Z_b = upsampling_layer(Z_b)  # 1, C, T, H, W

            Y_b = rearrange(
                Y_b,
                "T (H W) D -> D T H W",
                H=int(math.sqrt(Y_b.shape[1])),
                W=int(math.sqrt(Y_b.shape[1])),
            ).unsqueeze(0)  # 1, C_out, T_out, H_out, W_out

            patches = unfold3d(
                Z_b,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
            )
            X_b = patches.permute(0, 5, 6, 7, 1, 2, 3, 4).reshape(-1, d).to(dtype)
            Y_b_vec = Y_b.permute(0, 2, 3, 4, 1).reshape(-1, C_out).to(dtype)

            sum_residual += (Y_b_vec - X_b @ W).sum(0)
            del Z_b, Y_b, patches, X_b, Y_b_vec
            torch.cuda.empty_cache()

        conv.bias.copy_((sum_residual / n_rows).to(conv.weight.dtype))

    return conv


@torch.no_grad()
def extract_features(
    feedforward_model: torch.nn.Module,
    vae_model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
):
    batch_size = data_loader.batch_size
    vae_features = []
    feedforward_features = {}

    def make_hook_for_feedforward3Dmodel(name):
        def hook(module, input, output):
            if isinstance(feedforward_model, AnySplat):
                feature = rearrange(
                    output,
                    "(b v) l d -> b v l d",
                    b=batch_size,
                )[
                    :, :, 5:, :
                ]  # remove the first 5 tokens (corresponding prefix tokens)
            else:
                raise NotImplementedError(
                    f"Hook not implemented for {type(feedforward_model)}"
                )

            # store the feature by quantizing to float16 to save memory
            feature_cpu = feature.detach().cpu().clone().to(dtype=torch.float16)
            if name not in feedforward_features:
                feedforward_features[name] = []
            feedforward_features[name].append(feature_cpu)
            del feature

        return hook

    handles = []
    if isinstance(feedforward_model, AnySplat):
        for idx, block in enumerate(
            feedforward_model.encoder.aggregator.patch_embed.blocks
        ):
            handle = block.register_forward_hook(
                make_hook_for_feedforward3Dmodel(f"enc_blocks_{idx + 1}")
            )
            handles.append(handle)
    else:
        raise NotImplementedError(
            f"Hook registration not implemented for {type(feedforward_model)}"
        )

    for step, data in tqdm(
        enumerate(data_loader), total=args.iterations_for_feature_extraction
    ):
        if step > args.iterations_for_feature_extraction:
            break
        vae_images = data["vae_image_tensor"].to(vae_model.device)
        feedforward_img = data["feedforward_image_tensor"].to(vae_model.device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if isinstance(feedforward_model, AnySplat):
                    input_img = (feedforward_img + 1) / 2
                    input_img = rearrange(input_img, "b c v h w -> b v c h w")
                    _ = feedforward_model.inference(input_img)
                else:
                    raise NotImplementedError(
                        f"Feature extraction not implemented for {type(feedforward_model)}"
                    )

                if isinstance(vae_model, AutoencoderKLWan):
                    latents = vae_model.encode(vae_images).latent_dist.sample()
                else:
                    raise NotImplementedError(
                        f"VAE encoding not implemented for {type(vae_model)}"
                    )
        # store VAE features by quantizing to float16 to save memory
        vae_features.append(latents.detach().cpu().to(dtype=torch.float16))

    for handle in handles:
        handle.remove()

    for name, feature in feedforward_features.items():
        feedforward_features[name] = torch.cat(feature, dim=0)
    vae_features = torch.cat(vae_features, dim=0)
    return vae_features, feedforward_features


def main(args: argparse.Namespace):
    device = torch_device_setup()

    # feedforward_3d_model
    feed_forward_model = load_feedforward_model(args, device)
    # video vae
    video_vae = load_vae(args, device)
    train_dataloader, _ = create_stitching_dataloader(
        args.dataset, args, augmentation=False
    )

    # First step: extract features and save them
    os.makedirs(args.feature_save_path, exist_ok=True)
    feature_file_path = os.path.join(
        args.feature_save_path,
        "features.pt",
    )
    if os.path.exists(feature_file_path):
        logger.info(f"Feature file already exists at {feature_file_path}.")
        features = torch.load(
            feature_file_path,
            map_location="cpu",
            weights_only=False,
        )
        vae_features = features["vae_features"]
        feedforward_features = features["feedforward_features"]
    else:
        logger.info(f"Extracting features and saving to {feature_file_path}.")
        vae_features, feedforward_features = extract_features(
            feedforward_model=feed_forward_model,
            vae_model=video_vae,
            data_loader=train_dataloader,
        )
        torch.save(
            {
                "vae_features": vae_features,
                "feedforward_features": feedforward_features,
            },
            feature_file_path,
        )
        logger.info(f"Extracted features saved at {feature_file_path}.")

    # second step: MSE optimization
    for layer_key in feedforward_features.keys():
        stitching_layer = args.stitching_layer_config.build(
            in_channels=vae_features.shape[1]
        )
        state_dict_file_path = os.path.join(
            args.feature_save_path, f"state_dict_{layer_key}.pt"
        )
        if os.path.exists(state_dict_file_path):
            logger.info(
                f"Stitching layer state dict already exists at {state_dict_file_path}."
            )
            logger.info(f"Loading stitching layer state dict for layer {layer_key}.")
            state_dict = torch.load(
                state_dict_file_path,
                map_location="cpu",
                weights_only=False,
            )
            stitching_layer.load_state_dict(state_dict, strict=True)
        else:
            # fitting stitching layer
            stitching_layer = fit_conv3d_streaming(
                stitching_layer,
                vae_features,
                feedforward_features[layer_key],
                ridge=1e-4,
            )
            torch.save(
                stitching_layer.state_dict(),
                state_dict_file_path,
            )
            logger.info(
                f"Fitted stitching layer state dict saved at {state_dict_file_path}."
            )
        with torch.no_grad():
            mse, vox = 0.0, 0
            for Z_b, Y_b in tqdm(zip(vae_features, feedforward_features[layer_key])):

                def upsampling_layer(x):
                    T_vae = x.shape[2]
                    T_original = (T_vae - 1) * 4 + 1
                    x = nn.functional.interpolate(
                        x,
                        size=[
                            T_original,
                            vae_features.shape[3],
                            vae_features.shape[4],
                        ],
                        mode="trilinear",
                        align_corners=True,
                    )
                    return x

                Y_b = rearrange(
                    Y_b,
                    "T (H W) D -> D T H W",
                    H=int(math.sqrt(Y_b.shape[1])),
                    W=int(math.sqrt(Y_b.shape[1])),
                ).unsqueeze(0)  # 1, C_out, T_out, H_out, W_out
                Z_b = Z_b.unsqueeze(0)
                Z_b = upsampling_layer(Z_b)
                pred = stitching_layer(Z_b.float())
                mse += (pred - Y_b).square().sum().item()
                vox += pred.numel()
            print("Training MSE:", mse / vox)
            with open(
                os.path.join(args.feature_save_path, f"mse_{layer_key}.txt"), "w"
            ) as f:
                f.write(f"{mse / vox}\n")

    layers = list(feedforward_features.keys())
    mses = []
    for layer_key in feedforward_features.keys():
        with open(
            os.path.join(args.feature_save_path, f"mse_{layer_key}.txt"), "r"
        ) as f:
            mse = float(f.readline().strip())
            mses.append(mse)

    # give recommendation
    best_layer = layers[mses.index(min(mses))]
    logger.info(f"Best stitching layer: {best_layer} with MSE: {min(mses)}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # argument: iterations
    args = find_layer_stitching_argument().parse_args()
    main(args)
