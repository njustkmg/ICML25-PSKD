#!/bin/bash

# python main.py \
#     --config configs/datasets/imagenet200/imagenet200.yml \
#     configs/networks/t2fnorm_net.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/pipelines/train/train_t2fnorm.yml \
#     --network.backbone.name resnet18_224x224 \
#     --optimizer.num_epochs 90 \
#     --dataset.train.batch_size 128 \
#     --num_gpus 2 --num_workers 16 \
#     --merge_option merge \
#     --seed $RANDOM \

GPUID=$1
for kk in $(seq 1 1); do
    for seed in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$GPUID python main.py \
            --config configs/datasets/imagenet200/imagenet200.yml \
            configs/datasets/imagenet200/imagenet200_ood.yml \
            configs/networks/t2fnorm_net.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/pipelines/train/train_t2fnorm.yml \
            --network.backbone.name resnet18_224x224 \
            --optimizer.num_epochs 90 \
            --seed ${seed} #&
    done
    # wait  
done
