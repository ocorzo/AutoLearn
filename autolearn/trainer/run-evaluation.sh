#!/usr/bin/env bash
set -euo pipefail

# Usage: run-evaluation.sh baseline|candidate
# Both modes use the same engine (Transformers BF16), prompts and scoring, so
# their sentinel results can be compared directly.
MODE="${1:?Usage: run-evaluation.sh baseline|candidate}"
REMOTE_ROOT=/home/ocorzo/bigone-llm/autolearn
TRIAL_ID=001
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

case "${MODE}" in
  baseline) ADAPTER_ARGS=() ;;
  candidate) ADAPTER_ARGS=(--adapter-dir "/workspace/outputs/trial-${TRIAL_ID}/lora-candidate/adapter") ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

docker run --rm --gpus all \
  -v "${REMOTE_ROOT}/trials/${TRIAL_ID}:/workspace/trial:ro" \
  -v "${REMOTE_ROOT}/trainer:/workspace/trainer:ro" \
  -v "${REMOTE_ROOT}/models:/workspace/models:ro" \
  -v "${REMOTE_ROOT}/outputs:/workspace/outputs" \
  -v "${REMOTE_ROOT}/hf-cache:/workspace/hf-cache" \
  autolearn-trainer:qwen35-4b \
  python /workspace/trainer/evaluate_candidate.py \
    --trial-dir /workspace/trial \
    --model-dir /workspace/models/Qwen3.5-4B \
    ${ADAPTER_ARGS[@]+"${ADAPTER_ARGS[@]}"} \
    --output-dir "/workspace/outputs/trial-${TRIAL_ID}/evaluation-${MODE}-${RUN_ID}"
