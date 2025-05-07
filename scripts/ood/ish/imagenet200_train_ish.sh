#!/bin/bash
# sh scripts/ood/ish/imagenet200_train_ish.sh

GPUID=$1
for kk in $(seq 1 1); do
    for seed in 0 1 2; do
        CUDA_VISIBLE_DEVICES=$GPUID python main.py \
            --config configs/datasets/imagenet200/imagenet200.yml \
            configs/datasets/imagenet200/imagenet200_ood.yml \
            configs/networks/resnet18_224x224.yml \
            configs/pipelines/train/train_ish.yml \
            configs/preprocessors/base_preprocessor.yml \
            --optimizer.num_epochs 10 \
            --dataset.train.batch_size 128 \
            --seed ${seed} #&
    done
    # wait  
done

