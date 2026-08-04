#!/usr/bin/env bash
set -euo pipefail

OBJECT_ROOT="${1:-data/obj}"
METADATA="${2:-data/obj/metadata/annotations.json}"
OUTPUT="${3:-data/work/object_catalog}"

scenecompose build-object-catalog \
  --object-root "$OBJECT_ROOT" \
  --metadata "$METADATA" \
  --output "$OUTPUT" \
  --config configs/composition.json \
  --blender "${SCENECOMPOSE_BLENDER:-blender}"
