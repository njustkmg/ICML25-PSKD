#!/bin/bash
# sh scripts/ood/logitnorm/imagenet200_train_logitnorm.sh
GPUID=$1
for kk in $(seq 1 1); do
    for seed in 0 1 2; do
        python main.py \
            --config configs/datasets/imagenet200/imagenet200.yml \
            configs/datasets/imagenet200/imagenet200_ood.yml \
            configs/networks/resnet18_224x224.yml \
            configs/pipelines/train/train_logitnorm.yml \
            configs/preprocessors/base_preprocessor.yml \
            --optimizer.num_epochs 90 \
            --seed ${seed}
    done
    # wait  
done