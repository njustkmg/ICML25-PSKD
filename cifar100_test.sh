#!/bin/bash
# sh cifar100_test.sh

CUDA_VISIBLE_DEVICES=5 python scripts/eval_ood.py \
   --id-data cifar100 \
   --root './results/cifar100_resnet18_32x32_pskd_i_e100_lr0.1_w1_T3_a0.01_default' \
   --postprocessor ebo \
   --save-score --save-csv
   