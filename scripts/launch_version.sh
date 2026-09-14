#!/usr/bin/env bash
# Run from an immutable source checkout; provide version, run ID and provenance.
set -euo pipefail
version="${1:?version required}"
run_id="${2:?run ID required}"
source_record="${3:?source record required}"
[[ "$version" =~ ^v[1-9][0-9]*$ ]] || exit 2
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/root/autodl-tmp/COPE/.cache/huggingface
export XDG_CACHE_HOME=/root/autodl-tmp/COPE/.cache
export WANDB_MODE=disabled
exec /root/autodl-tmp/COPE/.venv/bin/python -u -m cope.real_experiment run \
  --config "goal/$version/config.json" \
  --data "/root/autodl-tmp/COPE/data/$version-prepared.json" \
  --run-id "$run_id" --source-record "$source_record" "${@:4}"
