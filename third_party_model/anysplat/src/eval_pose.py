import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from lpips import LPIPS
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from third_party_model.anysplat.src.misc.cam_utils import (
    camera_normalization,
    get_pnp_pose,
    pose_auc,
    rotation_6d_to_matrix,
    update_pose,
)
from third_party_model.anysplat.src.model.encoder.vggt.utils.load_fn import (
    load_and_preprocess_images,
)
from third_party_model.anysplat.src.model.encoder.vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
)
from third_party_model.anysplat.src.model.model.anysplat import AnySplat
from third_party_model.anysplat.src.utils.pose import (
    align_to_first_camera,
    calculate_auc_np,
    convert_pt3d_RT_to_opencv,
    se3_to_relative_pose_error,
)


def setup_args():
    """Set up command-line arguments for the CO3D evaluation script."""
    parser = argparse.ArgumentParser(description="Test AnySplat on CO3D dataset")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (only test on specific category)",
    )
    parser.add_argument(
        "--use_ba", action="store_true", default=False, help="Enable bundle adjustment"
    )
    parser.add_argument(
        "--fast_eval",
        action="store_true",
        default=False,
        help="Only evaluate 10 sequences per category",
    )
    parser.add_argument(
        "--min_num_images",
        type=int,
        default=50,
        help="Minimum number of images for a sequence",
    )
    parser.add_argument(
        "--num_frames", type=int, default=10, help="Number of frames to use for testing"
    )
    parser.add_argument(
        "--co3d_dir", type=str, required=True, help="Path to CO3D dataset"
    )
    parser.add_argument(
        "--co3d_anno_dir", type=str, required=True, help="Path to CO3D annotations"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    return parser.parse_args()


lpips = LPIPS(net="vgg")


def rendering_loss(pred_image, image):
    lpips_loss = lpips.forward(
        rearrange(pred_image, "b v c h w -> (b v) c h w"),
        rearrange(image, "b v c h w -> (b v) c h w"),
        normalize=True,
    )
    delta = pred_image - (image + 1) / 2
    mse_loss = (delta**2).mean()
    return mse_loss + 0.05 * lpips_loss.mean()


def process_sequence(
    model,
    seq_name,
    seq_data,
    category,
    co3d_dir,
    min_num_images,
    num_frames,
    use_ba,
    device,
    dtype,
):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: AnySplat model
        seq_name: Sequence name
        seq_data: Sequence data
        category: Category name
        co3d_dir: CO3D dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        use_ba: Whether to use bundle adjustment
        device: Device to run on
        dtype: Data type for model inference

    Returns:
        rError: Rotation errors
        tError: Translation errors
    """
    if len(seq_data) < min_num_images:
        return None, None

    metadata = []
    for data in seq_data:
        # Make sure translations are not ridiculous
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None, None

        extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append(
            {
                "filepath": data["filepath"],
                "extri": extri_opencv,
            }
        )

    ids = np.random.choice(len(metadata), num_frames, replace=False)
    image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)

    max_size = max(Image.open(image_names[0]).size)
    if max_size < 448:
        return None, None
    images = load_and_preprocess_images(image_names)[None].to(device)

    batch = {
        "context": {
            "image": images * 2.0 - 1,
            "image_names": image_names,
            "index": ids,
        },
        "scene": "co3d",
    }

    if use_ba:
        try:
            encoder_output = model.encoder(
                batch,
                global_step=0,
                visualization_dump={},
            )
            gaussians, pred_context_pose = (
                encoder_output.gaussians,
                encoder_output.pred_context_pose,
            )
            pred_extrinsic = pred_context_pose["extrinsic"]
            pred_intrinsic = pred_context_pose["intrinsic"]
            # rendering ba
            b, v, _, h, w = images.shape
            with (
                torch.set_grad_enabled(True),
                torch.cuda.amp.autocast(enabled=False, dtype=torch.float32),
            ):
                cam_rot_delta = nn.Parameter(
                    torch.zeros(
                        [b, v, 6],
                        requires_grad=True,
                        device=pred_extrinsic.device,
                        dtype=torch.float32,
                    )
                )
                cam_trans_delta = nn.Parameter(
                    torch.zeros(
                        [b, v, 3],
                        requires_grad=True,
                        device=pred_extrinsic.device,
                        dtype=torch.float32,
                    )
                )
                opt_params = []
                model.register_buffer(
                    "identity",
                    torch.tensor(
                        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32
                    ).to(pred_extrinsic.device),
                )
                opt_params.append(
                    {
                        "params": [cam_rot_delta],
                        "lr": 0.005,
                    }
                )
                opt_params.append(
                    {
                        "params": [cam_trans_delta],
                        "lr": 0.005,
                    }
                )
                pose_optimizer = torch.optim.Adam(opt_params)
                extrinsics = pred_extrinsic.clone().float()

                for i in range(100):
                    pose_optimizer.zero_grad()
                    dx, drot = cam_trans_delta, cam_rot_delta
                    rot = rotation_6d_to_matrix(
                        drot + model.identity.expand(b, v, -1)
                    )  # (..., 3, 3)

                    transform = torch.eye(4, device=extrinsics.device).repeat(
                        (b, v, 1, 1)
                    )
                    transform[..., :3, :3] = rot
                    transform[..., :3, 3] = dx

                    new_extrinsics = torch.matmul(extrinsics, transform)
                    # breakpoint()
                    output = model.decoder.forward(
                        gaussians,
                        new_extrinsics,
                        pred_intrinsic.float(),
                        0.1,
                        100.0,
                        (h, w),
                        # cam_rot_delta=cam_rot_delta,
                        # cam_trans_delta=cam_trans_delta,
                    )
                    # export_ply(gaussians.means[0], gaussians.scales[0], gaussians.rotations[0], gaussians.harmonics[0], gaussians.opacities[0], Path(f"gaussians_co3d.ply"))
                    rendering_loss = rendering_loss(output.color, images * 2.0 - 1)
                    torchvision.utils.save_image(
                        output.color[0], f"outputs/vis/output_co3d_{i}.png"
                    )
                    print(f"Rendering loss: {rendering_loss.item()}")
                    # print(f"Rendering loss: {rendering_loss.item()}")

                    rendering_loss.backward()
                    pose_optimizer.step()
                torchvision.utils.save_image(images[0], f"outputs/vis/gt_co3d.png")
                pred_extrinsic = new_extrinsics.inverse()[0][:, :-1, :]

        except Exception as e:
            print(
                f"BA failed with error: {e}. Falling back to standard VGGT inference."
            )
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(
                    images,
                    intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx,
                )
            with torch.cuda.amp.autocast(dtype=torch.float32):
                fp32_tokens = [token.float() for token in aggregated_tokens_list]
                pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
                pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
                    pred_all_pose_enc, images.shape[-2:]
                )
                pred_extrinsic = pred_all_extrinsic[0]
    else:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(
                images, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx
            )
        with torch.cuda.amp.autocast(dtype=torch.float32):
            fp32_tokens = [token.float() for token in aggregated_tokens_list]
            pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
            pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
                pred_all_pose_enc, images.shape[-2:]
            )
            pred_extrinsic = pred_all_extrinsic[0]

    with torch.cuda.amp.autocast(dtype=torch.float32):
        gt_extrinsic = torch.from_numpy(gt_extri).to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(
            pred_extrinsic.size(0), 1, 4
        )

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        # Set the coordinate of the first camera as the coordinate of the world
        # NOTE: DO NOT REMOVE THIS UNLESS YOU KNOW WHAT YOU ARE DOING
        # pred_se3 = align_to_first_camera(pred_se3)
        gt_se3 = align_to_first_camera(gt_se3)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
            pred_se3, gt_se3, num_frames
        )
        print(
            f"{category} sequence {seq_name} Rot Error: {rel_rangle_deg.mean().item():.4f}"
        )
        print(
            f"{category} sequence {seq_name} Trans Error: {rel_tangle_deg.mean().item():.4f}"
        )

        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()


def evaluate(args: argparse.Namespace):
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # CO3D evaluation
    SEEN_CATEGORIES = [
        "apple",
        "backpack",
        "banana",
        "baseballbat",
        "baseballglove",
        "bench",
        "bicycle",
        "bottle",
        "bowl",
        "broccoli",
        "cake",
        "car",
        "carrot",
        "cellphone",
        "chair",
        "cup",
        "donut",
        "hairdryer",
        "handbag",
        "hydrant",
        "keyboard",
        "laptop",
        "microwave",
        "motorcycle",
        "mouse",
        "orange",
        "parkingmeter",
        "pizza",
        "plant",
        "stopsign",
        "teddybear",
        "toaster",
        "toilet",
        "toybus",
        "toyplane",
        "toytrain",
        "toytruck",
        "tv",
        "umbrella",
        "vase",
        "wineglass",
    ]

    if args.debug:
        SEEN_CATEGORIES = ["apple"]

    per_category_results = {}

    for category in SEEN_CATEGORIES:
        print(f"Loading annotation for {category} test set")
        annotation_file = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        rError = []
        tError = []

        for seq_name, seq_data in annotation.items():
            print("-" * 50)

            print(f"Processing {seq_name} for {category} test set")
            if args.debug and not os.path.exists(
                os.path.join(args.co3d_dir, category, seq_name)
            ):
                print(f"Skipping {seq_name} (not found)")
                continue

            seq_rError, seq_tError = process_sequence(
                model,
                seq_name,
                seq_data,
                category,
                args.co3d_dir,
                args.min_num_images,
                args.num_frames,
                args.use_ba,
                device,
                torch.bfloat16,
            )

            print("-" * 50)

            if seq_rError is not None and seq_tError is not None:
                rError.extend(seq_rError)
                tError.extend(seq_tError)

        if not rError:
            print(f"No valid sequences found for {category}, skipping")
            continue

        rError = np.array(rError)
        tError = np.array(tError)

        thresholds = [5, 10, 20, 30]
        Aucs = {}

        for threshold in thresholds:
            Auc, _ = calculate_auc_np(rError, tError, max_threshold=threshold)
            Aucs[threshold] = Auc

        print("=" * 80)
        print(f"AUC of {category} test set: {Aucs[30]:.4f}")
        print("=" * 80)

        per_category_results[category] = {
            "rError": rError,
            "tError": tError,
            "Auc_5": Aucs[5],
            "Auc_10": Aucs[10],
            "Auc_20": Aucs[20],
            "Auc_30": Aucs[30],
        }

    # Print summary results
    print("\nSummary of AUC results:")
    print("-" * 50)
    for category in sorted(per_category_results.keys()):
        print(f"{category:<15} AUC_5: {per_category_results[category]['Auc_5']:.4f}")
        print(f"{category:<15} AUC_30: {per_category_results[category]['Auc_30']:.4f}")
        print(f"{category:<15} AUC_20: {per_category_results[category]['Auc_20']:.4f}")
        print(f"{category:<15} AUC_10: {per_category_results[category]['Auc_10']:.4f}")

    if per_category_results:
        mean_AUC_30 = np.mean(
            [
                per_category_results[category]["Auc_30"]
                for category in per_category_results
            ]
        )
        mean_AUC_20 = np.mean(
            [
                per_category_results[category]["Auc_20"]
                for category in per_category_results
            ]
        )
        mean_AUC_10 = np.mean(
            [
                per_category_results[category]["Auc_10"]
                for category in per_category_results
            ]
        )
        mean_AUC_5 = np.mean(
            [
                per_category_results[category]["Auc_5"]
                for category in per_category_results
            ]
        )
        print("-" * 50)
        print(f"Mean AUC_5: {mean_AUC_5:.4f}")
        print(f"Mean AUC_30: {mean_AUC_30:.4f}")
        print(f"Mean AUC_20: {mean_AUC_20:.4f}")
        print(f"Mean AUC_10: {mean_AUC_10:.4f}")

    # Generate a random index to avoid overwriting previous results
    # random_index = torch.randint(0, 10000, (1,)).item()
    # Use timestamp as index instead of random number
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_index = timestamp
    results_file = f"co3d_results_{random_index}.txt"

    with open(results_file, "w") as f:
        f.write("CO3D Evaluation Results\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-category results:\n")
        f.write("-" * 50 + "\n")
        for category in sorted(per_category_results.keys()):
            f.write(
                f"{category:<15} AUC_30: {per_category_results[category]['Auc_30']:.4f}\n"
            )
            f.write(
                f"{category:<15} AUC_20: {per_category_results[category]['Auc_20']:.4f}\n"
            )
            f.write(
                f"{category:<15} AUC_10: {per_category_results[category]['Auc_10']:.4f}\n"
            )
            f.write(
                f"{category:<15} AUC_5: {per_category_results[category]['Auc_5']:.4f}\n"
            )
            f.write("\n")

        if per_category_results:
            f.write("-" * 50 + "\n")
            f.write(f"Mean AUC_30: {mean_AUC_30:.4f}\n")
            f.write(f"Mean AUC_20: {mean_AUC_20:.4f}\n")
            f.write(f"Mean AUC_10: {mean_AUC_10:.4f}\n")
            f.write(f"Mean AUC_5: {mean_AUC_5:.4f}\n")
        f.write("\n" + "=" * 50 + "\n")

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    args = setup_args()
    evaluate(args)
