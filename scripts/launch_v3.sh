#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" == "--print-command" ]]; then
  echo 'PYTHONPATH=<frozen-checkout>/src /root/autodl-tmp/COPE/.venv/bin/python -u -m cope.search_experiment run --config goal/v3/config.json --data /root/autodl-tmp/COPE/data/v3-prepared.json --run-id <unique-run-id> --source-record <source-record.json> --assets /root/autodl-tmp/COPE/outputs/preparation/assets-v3.json'
  exit 0
fi
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/root/autodl-tmp/COPE/.cache/huggingface XDG_CACHE_HOME=/root/autodl-tmp/COPE/.cache
exec /root/autodl-tmp/COPE/.venv/bin/python -u -m cope.search_experiment run \
  --config goal/v3/config.json --data /root/autodl-tmp/COPE/data/v3-prepared.json \
  --run-id "${1:?run ID required}" --source-record "${2:?source record required}" \
  --assets /root/autodl-tmp/COPE/outputs/preparation/assets-v3.json "${@:3}"
