#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SCENECOMPOSE_PYTHON:-python3}"
BLENDER_BIN="${SCENECOMPOSE_BLENDER:-blender}"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_partition.sh <scene.fbx> <output-directory> [partition-config.json]" >&2
  exit 2
fi

SCENE_PATH="$1"
OUTPUT_PATH="$2"
CONFIG_PATH="${3:-${PROJECT_ROOT}/configs/partition.json}"

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m scenecompose partition \
  --scene "${SCENE_PATH}" \
  --output "${OUTPUT_PATH}" \
  --config "${CONFIG_PATH}" \
  --blender "${BLENDER_BIN}"

