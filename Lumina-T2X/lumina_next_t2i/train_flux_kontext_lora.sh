#!/usr/bin/env bash
set -Eeuo pipefail

# Keep this environment separate from the existing lum environment.
export HOME="${HOME:-/mnt/afs_fangwenqi}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/afs_fangwenqi/Lumina-T2X}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/afs_fangwenqi/miniconda3}"
CONDA_ENV="${CONDA_ENV:-${CONDA_ROOT}/envs/flux-kontext}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/lumina_next_t2i/flux_kontext_lora.yaml}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MASTER_PORT="${MASTER_PORT:-29621}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV}"
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cd "${PROJECT_ROOT}"
python lumina_next_t2i/train_flux_kontext_lora.py --config "${CONFIG_FILE}" --check-config
exec accelerate launch \
  --num_machines 1 \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MASTER_PORT}" \
  lumina_next_t2i/train_flux_kontext_lora.py \
  --config "${CONFIG_FILE}"
