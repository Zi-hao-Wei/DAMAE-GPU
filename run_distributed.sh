#!/bin/bash
IMAGENET_DIR="/data1/ImageNet"

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 distributed_train.py \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ./exp \
    --log_dir ./exp
