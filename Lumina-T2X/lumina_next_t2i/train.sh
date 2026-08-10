#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${SCRIPT_DIR}/config.yaml"
GPU_IDS="$("${PYTHON_BIN}" -c 'import sys, yaml; config = yaml.safe_load(open(sys.argv[1], encoding="utf-8")); print(",".join(str(value) for value in config["runtime"]["gpu_ids"]))' "${CONFIG_PATH}")"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"

if (( ${#GPU_ID_ARRAY[@]} > 1 )); then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  exec "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${#GPU_ID_ARRAY[@]}" \
    "${SCRIPT_DIR}/train_tokenlight.py" --config "${CONFIG_PATH}" "$@"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/train_tokenlight.py" --config "${CONFIG_PATH}" "$@"
