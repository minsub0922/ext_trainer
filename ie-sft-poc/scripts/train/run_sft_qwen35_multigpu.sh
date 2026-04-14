#!/usr/bin/env bash
# Multi-GPU SFT launcher for Qwen3.5 models.
#
# Mirror of run_sft_qwen3_multigpu.sh — same semantics, Qwen3.5 default config.
# DDP is used by default; DeepSpeed activates automatically when the resolved
# config YAML contains a `deepspeed:` field.
#
# Usage:
#   # 4-GPU LoRA SFT (default)
#   bash scripts/train/run_sft_qwen35_multigpu.sh
#
#   # 8-GPU full SFT + DeepSpeed ZeRO-2
#   NPROC=8 CONFIG=configs/sft/qwen3_5_full_sft_ds.yaml \
#       bash scripts/train/run_sft_qwen35_multigpu.sh

set -euo pipefail

CONFIG="${CONFIG:-configs/sft/qwen3_5_lora_sft.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${NPROC:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC=$(nvidia-smi -L | wc -l)
  else
    NPROC=1
  fi
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG" >&2
  exit 1
fi

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found. Install LLaMA-Factory first." >&2
  exit 2
fi

echo "==========================================="
echo "  Multi-GPU SFT run (Qwen3.5)"
echo "-------------------------------------------"
echo "  CONFIG:              $CONFIG"
echo "  NPROC (GPUs/node):   $NPROC"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "  MASTER_PORT:         $MASTER_PORT"
if grep -qE '^\s*deepspeed\s*:' "$CONFIG"; then
  DS=$(grep -E '^\s*deepspeed\s*:' "$CONFIG" | awk '{print $2}')
  echo "  DeepSpeed config:    $DS  (ZeRO will be enabled)"
else
  echo "  DeepSpeed config:    <none>  (plain DDP)"
fi
echo "==========================================="

if [[ "$NPROC" == "1" ]]; then
  exec llamafactory-cli train "$CONFIG"
fi

exec torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  "$(which llamafactory-cli)" train "$CONFIG"
