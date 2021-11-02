#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_construction.py --dataset_root ./datasets/construction/Construction_data\
  --model trained_models/construction/pose_model_28_0.2848551799853643.pth\
  --refine_model trained_models/construction/pose_refine_model_34_0.25522998546560605.pth