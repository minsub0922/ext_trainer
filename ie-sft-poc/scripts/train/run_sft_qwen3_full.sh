#!/usr/bin/env bash
# Qwen3-0.6B FULL SFT + DeepSpeed — multi-GPU via torchrun.
#
# Usage:
#   bash scripts/train/run_sft_qwen3_full.sh
#   NPROC=8 bash scripts/train/run_sft_qwen3_full.sh
#   CONFIG=configs/sft/qwen3_full_sft_ds.yaml NPROC=4 \
#     bash scripts/train/run_sft_qwen3_full.sh
#
# Env:
#   CONFIG       (default: configs/sft/qwen3_full_sft_ds.yaml)
#   NPROC        (default: auto via nvidia-smi)
#   MASTER_PORT  (default: 29500)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/sft/qwen3_full_sft_ds.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${NPROC:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC=$(nvidia-smi -L | wc -l)
  else
    NPROC=1
  fi
fi

[[ ! -f "$CONFIG" ]] && { echo "missing $CONFIG" >&2; exit 1; }
command -v llamafactory-cli >/dev/null || { echo "llamafactory-cli not found" >&2; exit 2; }

echo "=== Qwen3-0.6B FULL SFT (DeepSpeed) ==="
echo "  CONFIG: $CONFIG"
echo "  NPROC:  $NPROC"
if grep -qE '^\s*deepspeed\s*:' "$CONFIG"; then
  echo "  DeepSpeed: $(grep -E '^\s*deepspeed\s*:' "$CONFIG" | awk '{print $2}')"
fi

if [[ "$NPROC" == "1" ]]; then
  exec llamafactory-cli train "$CONFIG"
fi

exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  "$(which llamafactory-cli)" train "$CONFIG"
