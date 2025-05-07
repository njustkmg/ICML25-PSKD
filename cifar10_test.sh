#!/bin/bash
# sh cifar10_test.sh

CUDA_VISIBLE_DEVICES=5 python scripts/eval_ood.py \
   --id-data cifar10 \
   --root './results/cifar10_resnet18_32x32_pskd_i_e100_lr0.1_w1_T3_a0.01_default' \
   --postprocessor ebo \
   --save-score --save-csv
   