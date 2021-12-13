#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_construction.py --dataset_root ./datasets/construction/Construction_data\
  --model trained_models/construction/pose_model_62_0.049823602375302777.pth\
  --refine_model trained_models/construction/pose_refine_model_492_0.03314094381348696.pth