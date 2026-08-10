#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_tokenlight.py" --config "${SCRIPT_DIR}/config.yaml" "$@"

