#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/e/HK1-2026/PBL6/AI_Classifier"
python_bin="$HOME/miniforge3/envs/badminton-rgb/bin/python"
dataset_root="/mnt/c/Users/ADMIN/Downloads/Fine-Badminton"
output_dir="work_dirs/r2plus1d18_fine_badminton_8class_full"
cache_dir="$HOME/datasets/fine_badminton_cache"

cd "$repo_root"
export PYTHONPATH="$repo_root/src"

exec "$python_bin" -u scripts/training/train_fine_badminton_rgb.py \
  "$dataset_root" \
  --epochs 12 \
  --batch-size 4 \
  --frames 16 \
  --learning-rate 0.0001 \
  --amp \
  --cache-dir "$cache_dir" \
  --output-dir "$output_dir"
