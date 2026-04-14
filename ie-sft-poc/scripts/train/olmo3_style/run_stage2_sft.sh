#!/usr/bin/env bash
# Stage 2 — SFT warm-started from stage-1 mid-training checkpoint.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen3}"
case "$MODEL" in
  qwen3)   CONFIG="configs/olmo3_style/qwen3/stage2_sft.yaml"   ;;
  qwen3.5) CONFIG="configs/olmo3_style/qwen3_5/stage2_sft.yaml" ;;
  *) echo "bad MODEL: $MODEL" >&2; exit 2 ;;
esac

NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
MASTER_PORT="${MASTER_PORT:-29502}"

echo "=== Stage 2 SFT ($MODEL) ==="
echo "  CONFIG: $CONFIG  NPROC: $NPROC"
exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  "$(which llamafactory-cli)" train "$CONFIG"
