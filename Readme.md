# [ICLR 2026] VIST3A: Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator 


The official code for the paper: "Text-to-3D by Stitching a Multi-view Reconstruction Network to a Video Generator".

>  [Hyojun Go](https://gohyojun15.github.io/), [Dominik Nanhofer](https://scholar.google.com/citations?user=tFx8AhkAAAAJ&hl=en), [Goutam Bhat](https://goutamgmb.github.io/), [Prune Troung](https://prunetruong.com/), [Federico Tombari](https://federicotombari.github.io/), [Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=ko).
> 
> ETH Zurich, Google

<a href="https://arxiv.org/abs/2510.13454">
  <img src="https://img.shields.io/badge/arXiv-2510.13454-b31b1b.svg">
</a>
<a href="https://gohyojun15.github.io/VIST3A/">
  <img src="https://img.shields.io/badge/Project%20Page-Website renewal-brightgreen">
</a>


https://github.com/user-attachments/assets/8610f2ac-82cf-4c37-b4e0-6d8d8ff92c6f


## 🔥 Highlights

VIST3A is a framework for text-to-3D generation that combines a multi-view reconstruction network with a video generator LDM.

- **Text → 3DGS in one LDM path**. Generates high-quality, 3D-consistent Gaussian splats directly from text prompts — even with long and detailed descriptions, maintaining both semantic fidelity and visual realism.
- **Models**. Based on Wan 2.1-14B and Wan 2.1-1.3B, we release our own VIST3A-1.3B and VIST3A-14B models.

## TODO

### Stitching
- [x] Release training code
- [x] Release evaluation code
- [x] Update Readme.md


### VDM Fine-tuning
- [ ] Release VDM fine-tuning pipeline
  - [ ] Training code
  - [ ] Inference code
  - [ ] Evaluation code
  - [ ] Demo script / noteboo


## 📦 Installation
- GPU requirements: For all experiments, we used NVIDIA 4 x NVIDIA GH200 GPUs. 
- We used Python 3.12.3 and Pytorch 2.7.0.
    ```
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt

    # Since some packages have issues, we recommend installing them separately:
    pip install --no-build-isolation  torch_scatter==2.1.2
    MAX_JOBS=4 pip install --no-deps --no-build-isolation -v "xformers==0.0.24"
    ```

<!-- ## 🚀 Quickstart
We will be back with Demo.
```
TODO:
``` -->

## Data Preparation for training and evaluation
Please see [`data/Readme.md`](data/Readme.md) for data preparation instructions.


## 🧠 Training

### 🩹 Model stitching
Model stitching connects a video diffusion model latent space with a multi-view reconstruction network (AnySplat) such that the combined VAE produces 3D Gaussian splats by leveraging pretrained reconstruction network weights.

This process has two stages:

1. Identify the optimal stitching layer in the reconstruction network
2. Fine-tune the stitched model using LoRA adapters

---

#### Step 1: Finding the Stitching Layer

The goal of this step is to find an intermediate layer in the multi-view reconstruction network whose feature distribution best matches the VAE latent space of the video diffusion model.

We perform this by:
- Extracting intermediate features from candidate layers
- Comparing their linear transferability against VAE latents
- Selecting the layer with the best alignment
  ```
  python find_layer_for_stitching.py \
      --batch_size 4 \
      --dataset dl3dv:/path/to/DL3DV-ALL-960P \
      --num_frames_per_unit_scene 50 \
      --num_images_from_unit_scene 13 \
      --resolution 512 \
      --feedforward_resolution 448 \
      --feature_save_path initialization \
      --iterations_for_feature_extraction 100 \
      --stitching_layer_config conv3d_k5x3x3_o1024_s1x2x2_p2x1x1
  ```
The output features and statistics are saved to: `initialization/`

⚠️ Important Note: The stitching layer must produce feature maps with the correct spatial and temporal resolution expected by the stitched model. Please be cautious to set the `--stitching_layer_config` argument appropriately.

---

#### Step 2: Stitching and Fine-tuning
Once the stitching layer is selected:

- AnySplat is stitched to the video VAE by chopping the network at the selected layer
- The stitched model is fine-tuned using LoRA adapters on the DL3DV dataset
  ```
  NPROC_PER_NODE=4
  MASTER_PORT=29510

  python -m torch.distributed.run \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_port ${MASTER_PORT} \
    model_stitching_training.py \
    --stitching_layer_location enc_blocks_2 \ # specify the layer location based on Step 1
    --stitching_layer_config conv3d_k5x3x3_o1024_s1x2x2_p2x1x1 \
    --resolution 512 \
    --feedforward_resolution 448 \
    --initialization_weight_path initialization/state_dict_enc_blocks_2.pt \ # path to the stitching layer weights from Step 1
    --dataset scannet:/path/to/scannet_preprocess/scans \
    --dataset dl3dv:/path/to/DL3DV-ALL-960P \
    --lora_config r64,a32,d0.0,f0 \
    --num_frames_per_unit_scene 50 \
    --num_images_from_unit_scene 21 \
    --batch_size 3 \
    --num_epochs 30 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_steps 500 \
    --save_path trained_checkpoint/wan_anysplat_stitching \
    --wandb_logging True \
    --wandb_project_name stitching_wan_anysplat \
    --exp_name wan_anysplat_stitching \
    --resume_checkpoint_path (optional): path to resume checkpoint 
  ```

### 🎯 Reward Alignment
```
TODO:
```

## 🚩 Evaluation


### Model stitching
We evaluate the stitched AnySplat models on novel view synthesis (NVS) to assess how well the stitched latent space preserves 3D consistency and rendering quality compared to the original AnySplat model.

We follow the same evaluation protocol as in the paper and benchmark on the RealEstate10K (RE10K) dataset.

#### Evaluation command:
1) Inference (render novel views):
    ```
    export PYTHONPATH=$(pwd)
    python evaluation/novel_view_synthesis_bench/nvs_eval.py \
      --feedforward_model anysplat \
      --video_model wan \
      --stitching_layer_location enc_blocks_2 \
      --stitching_layer_config conv3d_k5x3x3_o1024_s1x2x2_p2x1x1 \
      --lora_config r64,a32,d0.0,f0 \
      --resolution 512 \
      --feedforward_resolution 448 \
      --dataset re10k:/scratch2/dataset_deliver/realestate10k_test \
      --seq_id_map evaluation/datasets/re10k_indexmap.json \
      --output_dir [your output directory] \
      --checkpoint_path [your stitched_model_checkpoint.pth] 
    ```
2) Compute PSNR / SSIM / LPIPS:
    ```
    export PYTHONPATH=$(pwd)
    python evaluation/novel_view_synthesis_bench/calculate_metric.py \
      --dataset re10k:/scratch2/dataset_deliver/realestate10k_test \
      --seq_id_map evaluation/datasets/re10k_indexmap.json \
      --output_dir [your_output_dir]
    ```

#### 📦 Checkpoints
We release two stitched AnySplat checkpoints used for evaluation.

> **Note on evaluation indices**
> The evaluation index set differs from the one reported in the paper.
> Due to automatic file cleanup on our cluster, the original evaluation index files were lost.
> Therefore, we re-ran all evaluations using newly generated evaluation indices.

Checkpoint download scripts are provided in[download_checkpoints.sh](download_checkpoints.sh).

#### 📊 Quantitative Results

| Model | Description | Checkpoint | PSNR  | LPIPS | SSIM |
|-------|-------------|------------|---------|---------|--------|
| Anysplat-stitched | Stitched AnySplat model (as described in the paper) | [anysplat_stitched.pth](https://huggingface.co/HJGO/VIST3A/resolve/main/anysplat_stitched.pth?download=true) | 20.94 | 0.6944 | 0.2383 |
| Anysplat-stitched-extended | Extended training (+30 epochs, 21-frame coverage) | [anysplat_stitched_21_frame_extended.pth](https://huggingface.co/HJGO/VIST3A/resolve/main/anysplat_stitched_21_frame_extended.pth?download=true) | 21.00 | 0.7047 | 0.2310 |
|      AnySplat original     |  Original AnySplat model without stitching |   -  |       20.57 | 0.6858 | 0.2428 | 


## 🙏 Acknowledgements
We build upon open-source implementations of video LDMs (Wan, CogVideoX, HunyuanVideo, SVD), multi-view recon (AnySplat, VGGT, MVDust3R), and Gsplat. Thanks to the respective authors and communities.

I really appreciate [Yongwei Chen](https://cyw-3d.github.io/) who helped me a lot with verifying our code before the release.