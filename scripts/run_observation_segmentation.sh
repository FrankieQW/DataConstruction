#!/usr/bin/env bash
set -euo pipefail

observation_root="${1:-data/work}"
gpus="${2:-0,1,2,3,4,5,6,7}"
workers="${3:-8}"
config="${4:-configs/segmentation.json}"

scenecompose segment-observations \
  --observation-root "$observation_root" \
  --config "$config" \
  --blender "${SCENECOMPOSE_BLENDER:-blender}" \
  --gpus "$gpus" \
  --workers "$workers" \
  --resume
