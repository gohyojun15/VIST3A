
# Dataset Preprocessing

This document describes how to prepare the datasets used for training and evaluation.

## Datasets
We used the following datasets for training stitched models:
- [DL3DV-10K](https://dl3dv-10k.github.io/DL3DV-10K/)
- [ScanNet](https://github.com/ScanNet/ScanNet)

For evaluating stitched models, we used:
- [RealEstate10K]()

<!-- For training video diffusion models, we used:
- [DL3DV-10K](https://dl3dv-10k.github.io/DL3DV-10K/) for generative loss.
- HPSv2 dataset for reward alignment. This is saved in  -->

---

## 1) DL3DV-10K

We use the **preprocessed** DL3DV-10K release provided by the authors:
- Hugging Face dataset: https://huggingface.co/datasets/DL3DV/DL3DV-ALL-960P

### 1.1 Download
```bash
huggingface-cli login

huggingface-cli download DL3DV/DL3DV-ALL-960P \
  --repo-type dataset \
  --resume-download \
  --local-dir DL3DV-ALL-960P \
  --max-workers 32
````

### 1.2 Unzip

The dataset is distributed as many `.zip` files. The following script extracts them **in parallel**.

> Choose **ONE** unzip behavior:
>
> * `-n`: never overwrite (recommended / safe)
> * `-o`: always overwrite

```bash
# choose ONE of these:
FLAG="-n"          # never overwrite (safe)
# FLAG="-o"        # always overwrite (use with caution)

# how many parallel jobs?
JOBS=32            # e.g., 4, 8, 16, ...

export FLAG

find . -mindepth 2 -maxdepth 2 -type f -name '*.zip' -print0 \
  | xargs -0 -n1 -P "$JOBS" \
  bash -c '
    z="$1"
    dir=$(dirname "$z")
    echo "→ extracting $(basename "$z")"
    unzip -q "$FLAG" "$z" -d "$dir"
  ' _
```

### 1.3 Expected folder structure

After extraction, your directory should look like:

```text
DL3DV-ALL-960P/
├── 1K/
│   ├── <scene_id>/
│   │   └── images_4/
│   │       ├── frame_00000.png
│   │       ├── frame_00001.png
│   │       └── ...
│   ├── <scene_id>.zip   # not needed after extraction (can be deleted)
│   └── ...
├── 2K/
├── ...
└── README.md
```

(Optional) You may delete the `.zip` files after successful extraction to save disk space.

---

## 2) ScanNet

### 2.1 Download

Download ScanNet by following the official instructions:

* [https://github.com/ScanNet/ScanNet](https://github.com/ScanNet/ScanNet)

### 2.2 Preprocess (frame extraction)

After downloading, preprocess ScanNet using the official `SensReader` scripts:

* [https://github.com/ScanNet/ScanNet/tree/master/SensReader/python](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python)

### 2.3 Expected folder structure

After running the preprocessing scripts, your directory should look like:

```text
scannet_preprocess/
└── scans/
    ├── scene0000_00/
    ├── scene0000_01/
    ├── ...
    └── scene0468_02/
        └── frames/
            ├── color/         # RGB frames (.jpg)
            ├── depth/         # depth frames (typically .png)
            ├── intrinsic/     # camera intrinsics (per-sensor files)
            ├── pose/          # camera-to-world poses (per-frame .txt)
            └── completed.txt  # marker that extraction finished
```


## 3. RealEstate10K
Please download test split of RealEstate10K dataset by following the instructions from [RealEstate10K_downloader](https://github.com/cashiwamochi/RealEstate10K_Downloader/tree/master).

