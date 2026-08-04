#!/usr/bin/env bash
set -euo pipefail

scene_root="${1:-data/scene}"
output_root="${2:-data/work}"
workers="${3:-1}"
config="${4:-configs/observation_partition.json}"

scenecompose sample-observations-all \
  --scene-root "$scene_root" \
  --output-root "$output_root" \
  --config "$config" \
  --blender "${SCENECOMPOSE_BLENDER:-blender}" \
  --workers "$workers"
