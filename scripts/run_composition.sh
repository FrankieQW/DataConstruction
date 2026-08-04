#!/usr/bin/env bash
set -euo pipefail

OBSERVATIONS="${1:-data/work}"
CATALOG="${2:-data/work/object_catalog/catalog.json}"
OUTPUT="${3:-data/composed}"

scenecompose compose-observations \
  --observation-root "$OBSERVATIONS" \
  --catalog "$CATALOG" \
  --output-root "$OUTPUT" \
  --config configs/composition.json \
  --segmentation-config configs/segmentation.json \
  --blender "${SCENECOMPOSE_BLENDER:-blender}"
