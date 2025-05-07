#!/bin/bash
# sh scripts/ood/snn/cifar10_train_snn.sh

GPUID=$1
for kk in $(seq 1 1); do
    for seed in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$GPUID python main.py \
            --config configs/datasets/cifar10/cifar10.yml \
            configs/datasets/cifar10/cifar10_ood.yml \
            configs/networks/resnet18_32x32.yml \
            configs/pipelines/train/train_snn.yml \
            configs/preprocessors/base_preprocessor.yml \
            --seed ${seed} &
    done
    wait  
done