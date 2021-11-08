#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_construction.py --dataset_root ./datasets/construction/Construction_data\
  --model trained_models/construction/pose_model_270_0.02581593139717976.pth\
  --refine_model trained_models/construction/pose_refine_model_199_0.05785269675155481.pth