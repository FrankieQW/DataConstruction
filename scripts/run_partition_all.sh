#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SCENECOMPOSE_PYTHON:-python3}"
BLENDER_BIN="${SCENECOMPOSE_BLENDER:-blender}"
SCENE_ROOT="${1:-${PROJECT_ROOT}/data/scene}"
OUTPUT_ROOT="${2:-${PROJECT_ROOT}/data/work}"
WORKERS="${3:-1}"
CONFIG_PATH="${4:-${PROJECT_ROOT}/configs/partition.json}"

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m scenecompose partition-all \
  --scene-root "${SCENE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --blender "${BLENDER_BIN}" \
  --workers "${WORKERS}"

