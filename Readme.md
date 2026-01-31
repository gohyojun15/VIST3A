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

## Data Preparation
Please see [`data/Readme.md`](data/Readme.md) for data preparation instructions.


## 🧠 Training

### 🩹 Model stitching
```

```

### 🎯 Reward Alignment
```
TODO:
```


## 🙏 Acknowledgements
We build upon open-source implementations of video LDMs (Wan, CogVideoX, HunyuanVideo, SVD), multi-view recon (AnySplat, VGGT, MVDust3R), and Gsplat. Thanks to the respective authors and communities.

I really appreciate [Yongwei Chen](https://cyw-3d.github.io/) who helped me a lot with verifying our code before the release.

