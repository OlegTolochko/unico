#!/usr/bin/env bash

if [ "${DEBUG:-0}" = "1" ]; then
  set -x
fi

NGPUS=$(uv run python - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)

if [ "$NGPUS" -lt 1 ]; then
  echo "No CUDA devices are visible to PyTorch." >&2
  exit 1
fi

PORT="${MASTER_PORT:-29500}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  PORT="$1"
  shift 1
fi

echo "Using $NGPUS GPUs"

UNICO_USE_HELPER_SCRIPT=1 uv run torchrun --master_port="${PORT}" --nproc_per_node="${NGPUS}" main.py "$@"

# Usage:
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh
# CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_ddp.sh 29500
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/train_ddp.sh 29500 exp_name=my_exp runtime.seed=1117
# ./scripts/train_ddp.sh  # Uses all GPUs and default port 29500
