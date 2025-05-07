#!/bin/bash
# sh scripts/ood/pskd/imagenet200_train_pskd.sh

GPUID=$1
for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$GPUID python main.py \
        --config configs/datasets/imagenet200/imagenet200.yml \
        configs/datasets/imagenet200/imagenet200_ood.yml \
        configs/networks/resnet18_224x224.yml \
        configs/pipelines/train/train_pskd.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/ebo.yml \
        --trainer.trainer_args.alpha 0.05 \
        --optimizer.num_epochs 90 \
        --seed ${seed}
done




