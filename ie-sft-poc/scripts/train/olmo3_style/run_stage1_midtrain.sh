#!/usr/bin/env bash
# OLMo3-style stage 1 — mid-training (continued pretraining on IE mixture).
#
# Usage:
#   MODEL=qwen3   bash scripts/train/olmo3_style/run_stage1_midtrain.sh
#   MODEL=qwen3.5 bash scripts/train/olmo3_style/run_stage1_midtrain.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen3}"
case "$MODEL" in
  qwen3)   CONFIG="configs/olmo3_style/qwen3/stage1_midtrain.yaml"   ;;
  qwen3.5) CONFIG="configs/olmo3_style/qwen3_5/stage1_midtrain.yaml" ;;
  *) echo "bad MODEL: $MODEL" >&2; exit 2 ;;
esac

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "=== Stage 1 mid-training ($MODEL) ==="
echo "  CONFIG: $CONFIG"
echo "  NPROC:  $NPROC"

exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  "$(which llamafactory-cli)" train "$CONFIG"
