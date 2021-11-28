#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_airplane.py --dataset_root ./datasets/airplane/Airplane_data\
  --model trained_models/airplane/pose_model_496_2.5787107065320014.pth\