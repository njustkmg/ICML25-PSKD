#!/bin/bash
# sh scripts/ood/logitnorm/cifar10_train_logitnorm.sh

GPUID=$1
for kk in $(seq 1 1); do
    for seed in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$GPUID python main.py \
            --config configs/datasets/cifar100/cifar100.yml \
            configs/datasets/cifar100/cifar100_ood.yml \
            configs/networks/resnet18_32x32.yml \
            configs/pipelines/train/train_logitnorm.yml \
            configs/preprocessors/base_preprocessor.yml \
            --seed ${seed} &
    done
    wait  
done
