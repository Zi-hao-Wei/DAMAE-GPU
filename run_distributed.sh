#!/bin/bash
IMAGENET_DIR="/data1/data/ImageNet"

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 distributed_train.py \
    --batch_size 256 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ./exp \
    --log_dir ./exp
