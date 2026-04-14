#!/usr/bin/env bash
# Stage 3 — DPO on preference pairs built from stage-2 samples.
#
# Before running, build the pairs:
#   python scripts/preprocess/olmo3_style/build_preference_pairs.py \
#     --model-path outputs/olmo3_style/${TAG}/stage2_sft \
#     --test data/processed/splits/train.jsonl \
#     --output data/processed/olmo3_style/preference_pairs.jsonl \
#     --k 4 --limit 2000
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen3}"
case "$MODEL" in
  qwen3)   CONFIG="configs/olmo3_style/qwen3/stage3_dpo.yaml"   ;;
  qwen3.5) CONFIG="configs/olmo3_style/qwen3_5/stage3_dpo.yaml" ;;
  *) echo "bad MODEL: $MODEL" >&2; exit 2 ;;
esac

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
MASTER_PORT="${MASTER_PORT:-29503}"

echo "=== Stage 3 DPO ($MODEL) ==="
echo "  CONFIG: $CONFIG  NPROC: $NPROC"
exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  "$(which llamafactory-cli)" train "$CONFIG"
