#!/usr/bin/env bash
set -euo pipefail

manifest="${1:-data/obj/Objaverse.md}"
output="${2:-data/obj/metadata}"

python scripts/download_objaverse_metadata.py \
  --manifest "$manifest" \
  --output "$output"
