#!/usr/bin/env python3
import functools
import gc
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from third_party_model.anysplat.src.misc.image_io import save_interpolated_video
from third_party_model.anysplat.src.model.model.anysplat import AnySplat
from third_party_model.anysplat.src.model.ply_export import export_ply
from third_party_model.anysplat.src.utils.image import process_image


# 1) Core model inference
def get_reconstructed_scene(outdir, model, device):
    # Load Images
    image_files = sorted(
        [
            os.path.join(outdir, "images", f)
            for f in os.listdir(os.path.join(outdir, "images"))
        ]
    )
    images = [process_image(img_path) for img_path in image_files]
    images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]
    b, v, c, h, w = images.shape

    assert c == 3, "Images must have 3 channels"

    # Run Inference
    gaussians, pred_context_pose = model.inference((images + 1) * 0.5)

    # Save the results
    pred_all_extrinsic = pred_context_pose["extrinsic"]
    pred_all_intrinsic = pred_context_pose["intrinsic"]
    video, depth_colored = save_interpolated_video(
        pred_all_extrinsic,
        pred_all_intrinsic,
        b,
        h,
        w,
        gaussians,
        outdir,
        model.decoder,
    )

    plyfile = os.path.join(outdir, "gaussians.ply")
    export_ply(
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        Path(plyfile),
        save_sh_dc_only=True,
    )

    # Clean up
    torch.cuda.empty_cache()
    return plyfile, video, depth_colored


# 2) Handle uploaded video/images --> produce target_dir + images
def handle_uploads(input_video, input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images or extracted frames from video into it. Return (target_dir, image_paths).
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- Handle video ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 1 frame/sec

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(
                    target_dir_images, f"{video_frame_num:06}.png"
                )
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(
        f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds"
    )
    return target_dir, image_paths


# 3) Update gallery on upload
def update_gallery_on_upload(input_video, input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_video and not input_images:
        return None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths


# 4) Reconstruction: uses the target_dir plus any viz parameters
def gradio_demo(
    target_dir,
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = (
        sorted(os.listdir(target_dir_images))
        if os.path.isdir(target_dir_images)
        else []
    )
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]

    print("Running run_model...")
    with torch.no_grad():
        plyfile, video, depth_colored = get_reconstructed_scene(
            target_dir, model, device
        )

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")

    return plyfile, video, depth_colored


def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None, None, None


if __name__ == "__main__":
    server_name = "127.0.0.1"
    server_port = None
    share = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AnySplat.from_pretrained(
        "lhjiang/anysplat",
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )
    css = """
        .custom-log * {
            font-style: italic;
            font-size: 22px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: bold !important;
            color: transparent !important;
            text-align: center !important;
        }
        
        .example-log * {
            font-style: italic;
            font-size: 16px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent !important;
        }
        
        #my_radio .wrap {
            display: flex;
            flex-wrap: nowrap;
            justify-content: center;
            align-items: center;
        }

        #my_radio .wrap label {
            display: flex;
            width: 50%;
            justify-content: center;
            align-items: center;
            margin: 0;
            padding: 10px 0;
            box-sizing: border-box;
        }
        """
    with gr.Blocks(css=css, title="AnySplat Demo", theme=theme) as demo:
        gr.Markdown(
            """
            <h1 style='text-align: center;'>AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views</h1>
            """
        )

        with gr.Row():
            gr.Markdown(
                """
                        <p align="center">
                        <a title="Website" href="https://city-super.github.io/anysplat/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                            <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
                        </a>
                        <a title="arXiv" href="https://arxiv.org/pdf/2505.23716" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                            <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
                        </a>
                        <a title="Github" href="https://github.com/OpenRobotLab/AnySplat" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                            <img src="https://img.shields.io/badge/Github-Page-black" alt="badge-github-stars">
                        </a>
                
                        </p>
                        """
            )
        with gr.Row():
            gr.Markdown(
                """
            ### Getting Started:

            1. Upload Your Data: Use the "Upload Video" or "Upload Images" buttons on the left to provide your input. Videos will be automatically split into individual frames (one frame per second).

            2. Preview: Your uploaded images will appear in the gallery on the left.

            3. Reconstruct: Click the "Reconstruct" button to start the 3D reconstruction process.

            4. Visualize: The reconstructed 3D Gaussian Splat will appear in the viewer on the right, along with the rendered RGB and depth videos. The trajectory of the rendered video is obtained by interpolating the estimated input image poses.
            
            <strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">The generated splats are large in size, so they may not load successfully in the Hugging Face demo. You can download the .ply file and render it using other viewers, such as [SuperSplat](https://playcanvas.com/supersplat/editor).</span>
            """
            )

        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
        is_example = gr.Textbox(label="is_example", visible=False, value="None")
        num_images = gr.Textbox(label="num_images", visible=False, value="None")
        dataset_name = gr.Textbox(label="dataset_name", visible=False, value="None")
        scene_name = gr.Textbox(label="scene_name", visible=False, value="None")
        image_type = gr.Textbox(label="image_type", visible=False, value="None")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Input Data"):
                        input_video = gr.Video(label="Upload Video", interactive=True)
                        input_images = gr.File(
                            file_count="multiple",
                            label="Upload Images",
                            interactive=True,
                        )

                        image_gallery = gr.Gallery(
                            label="Preview",
                            columns=4,
                            height="300px",
                            show_download_button=True,
                            object_fit="contain",
                            preview=True,
                        )

            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("AnySplat Output"):
                        with gr.Column():
                            reconstruction_output = gr.Model3D(
                                label="3D Reconstructed Gaussian Splat",
                                height=540,
                                zoom_speed=0.5,
                                pan_speed=0.5,
                                camera_position=[20, 20, 20],
                            )

                        with gr.Row():
                            with gr.Row():
                                rgb_video = gr.Video(
                                    label="RGB Video", interactive=False, autoplay=True
                                )
                                depth_video = gr.Video(
                                    label="Depth Video",
                                    interactive=False,
                                    autoplay=True,
                                )

                        with gr.Row():
                            submit_btn = gr.Button(
                                "Reconstruct", scale=1, variant="primary"
                            )
                            clear_btn = gr.ClearButton(
                                [
                                    input_video,
                                    input_images,
                                    reconstruction_output,
                                    target_dir_output,
                                    image_gallery,
                                    rgb_video,
                                    depth_video,
                                ],
                                scale=1,
                            )

        # ---------------------- Examples section ----------------------

        examples = [
            [
                None,
                "examples/video/re10k_1eca36ec55b88fe4.mp4",
                "re10k",
                "1eca36ec55b88fe4",
                "2",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/bungeenerf_colosseum.mp4",
                "bungeenerf",
                "colosseum",
                "8",
                "Synthetic",
                "True",
            ],
            [
                None,
                "examples/video/fox.mp4",
                "InstantNGP",
                "fox",
                "14",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/matrixcity_street.mp4",
                "matrixcity",
                "street",
                "32",
                "Synthetic",
                "True",
            ],
            [
                None,
                "examples/video/vrnerf_apartment.mp4",
                "vrnerf",
                "apartment",
                "32",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/vrnerf_kitchen.mp4",
                "vrnerf",
                "kitchen",
                "17",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/vrnerf_riverview.mp4",
                "vrnerf",
                "riverview",
                "12",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/vrnerf_workshop.mp4",
                "vrnerf",
                "workshop",
                "32",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/fillerbuster_ramen.mp4",
                "fillerbuster",
                "ramen",
                "32",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/meganerf_rubble.mp4",
                "meganerf",
                "rubble",
                "10",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/llff_horns.mp4",
                "llff",
                "horns",
                "12",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/llff_fortress.mp4",
                "llff",
                "fortress",
                "7",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/dtu_scan_106.mp4",
                "dtu",
                "scan_106",
                "20",
                "Real",
                "True",
            ],
            [
                None,
                "examples/video/horizongs_hillside_summer.mp4",
                "horizongs",
                "hillside_summer",
                "55",
                "Synthetic",
                "True",
            ],
            [
                None,
                "examples/video/kitti360.mp4",
                "kitti360",
                "kitti360",
                "64",
                "Real",
                "True",
            ],
        ]

        def example_pipeline(
            input_images,
            input_video,
            dataset_name,
            scene_name,
            num_images_str,
            image_type,
            is_example,
        ):
            """
            1) Copy example images to new target_dir
            2) Reconstruct
            3) Return model3D + logs + new_dir + updated dropdown + gallery
            We do NOT return is_example. It's just an input.
            """
            target_dir, image_paths = handle_uploads(input_video, input_images)
            plyfile, video, depth_colored = gradio_demo(target_dir)
            return plyfile, video, depth_colored, target_dir, image_paths

        gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

        gr.Examples(
            examples=examples,
            inputs=[
                input_images,
                input_video,
                dataset_name,
                scene_name,
                num_images,
                image_type,
                is_example,
            ],
            outputs=[
                reconstruction_output,
                rgb_video,
                depth_video,
                target_dir_output,
                image_gallery,
            ],
            fn=example_pipeline,
            cache_examples=False,
            examples_per_page=50,
        )

        gr.Markdown(
            "<p style='text-align: center; font-style: italic; color: #666;'>We thank VGGT for their excellent gradio implementation!</p>"
        )

        submit_btn.click(
            fn=clear_fields,
            inputs=[],
            outputs=[reconstruction_output, rgb_video, depth_video],
        ).then(
            fn=gradio_demo,
            inputs=[
                target_dir_output,
            ],
            outputs=[reconstruction_output, rgb_video, depth_video],
        ).then(fn=lambda: "False", inputs=[], outputs=[is_example])

        input_video.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery],
        )
        input_images.change(
            fn=update_gallery_on_upload,
            inputs=[input_video, input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery],
        )

        # demo.launch(share=share, server_name=server_name, server_port=server_port)
        demo.launch(show_error=True, share=True, server_name="0.0.0.0")

        # We thank VGGT for their excellent gradio implementation
