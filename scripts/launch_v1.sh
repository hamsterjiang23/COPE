#!/usr/bin/env bash
# Invoke from the frozen source checkout; the scheduler provides the unique run ID.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/root/autodl-tmp/COPE/.cache/huggingface
export XDG_CACHE_HOME=/root/autodl-tmp/COPE/.cache
export WANDB_MODE=disabled
runtime=/root/autodl-tmp/COPE/.venv/bin/python
exec "$runtime" -u -m cope.real_experiment run \
  --config goal/v1/config.json \
  --data /root/autodl-tmp/COPE/data/v1-prepared.json \
  --run-id "$1" --source-record "$2" "${@:3}"
