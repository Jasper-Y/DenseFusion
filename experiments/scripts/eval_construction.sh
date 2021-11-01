#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_construction.py --dataset_root ./datasets/construction/Construction_data\
  --model trained_models/construction/pose_model_15_0.05815622769296169.pth\
  --refine_model trained_models/construction/pose_refine_model_186_0.027389534283429384.pth