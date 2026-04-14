#!/usr/bin/env bash
# Qwen3-0.6B LoRA SFT — single-GPU (or DDP via CUDA_VISIBLE_DEVICES).
#
# Usage:
#   bash scripts/train/run_sft_qwen3_lora.sh
#   bash scripts/train/run_sft_qwen3_lora.sh --dry-run
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train/run_sft_qwen3_lora.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/sft/qwen3_lora_sft.yaml}"
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --config)  ;;  # handled below
  esac
done
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --dry-run) shift ;;
    *) shift ;;
  esac
done

echo "=== Qwen3-0.6B LoRA SFT ==="
echo "  CONFIG: $CONFIG"

[[ ! -f "$CONFIG" ]] && { echo "missing $CONFIG" >&2; exit 1; }
command -v llamafactory-cli >/dev/null || { echo "llamafactory-cli not found" >&2; exit 2; }

if [[ "$DRY_RUN" == true ]]; then
  echo "DRY RUN: llamafactory-cli train $CONFIG"
  exit 0
fi

exec llamafactory-cli train "$CONFIG"
