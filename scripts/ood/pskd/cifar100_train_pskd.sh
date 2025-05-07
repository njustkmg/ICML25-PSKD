#!/bin/bash
# sh scripts/ood/pskd/cifar100_train_pskd.sh

GPUID=$1
for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPUID python main.py \
        --config configs/datasets/cifar100/cifar100.yml \
        configs/datasets/cifar100/cifar100_ood.yml \
        configs/networks/resnet18_32x32.yml \
        configs/pipelines/train/train_pskd.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/ebo.yml \
        --seed ${seed} &
done
wait  



