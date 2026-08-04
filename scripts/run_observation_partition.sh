#!/usr/bin/env bash
set -euo pipefail

scene="${1:?usage: run_observation_partition.sh SCENE OUTPUT [CONFIG]}"
output="${2:?usage: run_observation_partition.sh SCENE OUTPUT [CONFIG]}"
config="${3:-configs/observation_partition.json}"

scenecompose sample-observations \
  --scene "$scene" \
  --output "$output" \
  --config "$config" \
  --blender "${SCENECOMPOSE_BLENDER:-blender}"
