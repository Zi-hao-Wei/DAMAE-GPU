#!/bin/bash
IMAGENET_DIR="~/data/imagenet"

python main_pretrain.py \
    --batch_size 16 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.9184 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ./exp \
    --log_dir ./exp 

