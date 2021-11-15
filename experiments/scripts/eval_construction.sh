#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_construction.py --dataset_root ./datasets/construction/Construction_data\
  --model trained_models/construction/saved/pose_model_asym.pth\
  --refine_model trained_models/construction/saved/pose_refine_asym.pth