#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT=/home/ocorzo/bigone-llm/autolearn
TRIAL_ID=001

docker run --rm --gpus all \
  -v "${REMOTE_ROOT}/trials/${TRIAL_ID}:/workspace/trial:ro" \
  -v "${REMOTE_ROOT}/trainer:/workspace/trainer:ro" \
  -v "${REMOTE_ROOT}/models:/workspace/models:ro" \
  -v "${REMOTE_ROOT}/outputs:/workspace/outputs" \
  -v "${REMOTE_ROOT}/hf-cache:/workspace/hf-cache" \
  autolearn-trainer:qwen35-4b \
  python /workspace/trainer/train_lora.py \
    --trial-dir /workspace/trial \
    --model-dir /workspace/models/Qwen3.5-4B \
    --output-dir /workspace/outputs/trial-001/lora-candidate
