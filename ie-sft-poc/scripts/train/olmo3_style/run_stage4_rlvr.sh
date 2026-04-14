#!/usr/bin/env bash
# Stage 4 — RLVR (GRPO-lite) with verifiable IE-F1 rewards.
#
# Before running, build the RLVR prompt file:
#   python scripts/preprocess/olmo3_style/build_rlvr_prompts.py \
#     --input data/processed/splits/train.jsonl \
#     --output data/processed/olmo3_style/rlvr_prompts.jsonl
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen3}"
case "$MODEL" in
  qwen3)   CONFIG="configs/olmo3_style/qwen3/stage4_rlvr.yaml"   ;;
  qwen3.5) CONFIG="configs/olmo3_style/qwen3_5/stage4_rlvr.yaml" ;;
  *) echo "bad MODEL: $MODEL" >&2; exit 2 ;;
esac

echo "=== Stage 4 RLVR ($MODEL) ==="
echo "  CONFIG: $CONFIG"

exec python -m src.training.olmo3_style.rlvr_trainer "$CONFIG"
