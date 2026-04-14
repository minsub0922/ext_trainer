#!/usr/bin/env bash
# One-shot: zero → trained Qwen3-0.6B.
# Runs the full data pipeline (download → normalize → unify → split → export)
# and then SFT, skipping any stage whose output already exists.
#
# Usage:
#   bash scripts/train/oneshot/run_oneshot_qwen3.sh           # LoRA (default)
#   MODE=full bash scripts/train/oneshot/run_oneshot_qwen3.sh # Full SFT
#   CONFIG=configs/sft/qwen3_lora_sft.yaml bash scripts/train/oneshot/run_oneshot_qwen3.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

MODE="${MODE:-lora}"
case "$MODE" in
  lora) DEFAULT_CONFIG="configs/sft/qwen3_lora_sft.yaml"   ;;
  full) DEFAULT_CONFIG="configs/sft/qwen3_full_sft_ds.yaml" ;;
  *) echo "bad MODE: $MODE (lora|full)" >&2; exit 2 ;;
esac
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

echo "=== Qwen3-0.6B one-shot ($MODE) ==="
echo "  CONFIG: $CONFIG"

run_data_pipeline
run_train "$CONFIG"
