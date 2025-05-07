#!/bin/bash
# sh scripts/ood/ish/imagenet_train_ish.sh
# pretrained model: https://drive.google.com/file/d/1EQimcdbJsKdU2uw4-BrqZO6tu4kXKtbG

CUDA_VISIBLE_DEVICES=5 python main.py  \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/train_ish.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint /data/xhn/OpenOOD/results/imagenet200_resnet18_224x224_pskd_e90_lr0.1_w1_T3_a0.3_default/s0/best.ckpt \
    --trainer.trainer_args.param 0.85 \
    --optimizer.lr 0.0003 \
    --optimizer.weight_decay_fc 0.00005 \
    --optimizer.num_epochs 10 \
    --dataset.train.batch_size 128 \
    --merge_option merge \
    --seed 0
