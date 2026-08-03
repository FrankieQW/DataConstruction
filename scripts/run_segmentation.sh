#!/usr/bin/env bash
set -euo pipefail

scene_root="${1:-data/scene}"
output_root="${2:-data/work}"
gpus="${3:-0,1,2,3,4,5,6,7}"
workers="${4:-8}"
config="${5:-configs/segmentation.json}"

scenecompose segment-scenes \
  --scene-root "$scene_root" \
  --output-root "$output_root" \
  --config "$config" \
  --gpus "$gpus" \
  --workers "$workers" \
  --resume
