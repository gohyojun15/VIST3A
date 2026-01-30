import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from third_party_model.anysplat.src.misc.image_io import save_interpolated_video
from third_party_model.anysplat.src.model.model.anysplat import AnySplat
from third_party_model.anysplat.src.model.ply_export import export_ply
from third_party_model.anysplat.src.utils.image import process_image


def main():
    # Load the model from Hugging Face
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load Images
    image_folder = "examples/vrnerf/riverview"
    images = sorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    images = [process_image(img_path) for img_path in images]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape

    # Run Inference
    gaussians, pred_context_pose = model.inference((images + 1) * 0.5)

    # Save the results
    pred_all_extrinsic = pred_context_pose["extrinsic"]
    pred_all_intrinsic = pred_context_pose["intrinsic"]
    save_interpolated_video(
        pred_all_extrinsic,
        pred_all_intrinsic,
        b,
        h,
        w,
        gaussians,
        image_folder,
        model.decoder,
    )
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        Path(image_folder) / "gaussians.ply",
    )


if __name__ == "__main__":
    main()
