#!/bin/bash
# sh imagenet200_test.sh

CUDA_VISIBLE_DEVICES=4 python scripts/eval_ood.py \
   --id-data imagenet200 \
   --root './results/imagenet200_resnet18_224x224_pskd_i_e90_lr0.1_w1_T3_a0.05_default_last' \
   --postprocessor ebo \
   --save-score --save-csv
