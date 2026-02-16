

# Stitched AnySplat model checkpoints

## anysplat_stitched.pth: regular training
wget -O anysplat_stitched.pth https://huggingface.co/HJGO/VIST3A/resolve/main/anysplat_stitched.pth?download=true

## anysplat_stitched.pth: We trained the model for 30 more epochs
wget -O anysplat_stitched_21_frame_extended.pth https://huggingface.co/HJGO/VIST3A/resolve/main/anysplat_stitched_21_frame_extended.pth?download=true


huggingface-cli download HJGO/VIST3A --include "vist3a_1.3b_lora_ema/*" --local-dir ./checkpoints

huggingface-cli download HJGO/VIST3A --include "vist3a_14b_lora_ema/*" --local-dir ./checkpoints
