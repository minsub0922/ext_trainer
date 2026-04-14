#!/usr/bin/env bash
# Multi-GPU SFT launcher for Qwen3 models.
#
# Defaults to DDP (torchrun). DeepSpeed is triggered automatically when the
# resolved config YAML contains a `deepspeed:` field, which is the standard
# LLaMA-Factory behavior — no separate flag needed.
#
# Usage:
#   # 4-GPU LoRA SFT (default config)
#   bash scripts/train/run_sft_qwen3_multigpu.sh
#
#   # 8-GPU full SFT + DeepSpeed ZeRO-2
#   NPROC=8 CONFIG=configs/sft/qwen3_full_sft_ds.yaml \
#       bash scripts/train/run_sft_qwen3_multigpu.sh
#
#   # Pin GPUs explicitly
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_sft_qwen3_multigpu.sh
#
# Env vars:
#   NPROC               number of GPUs per node (default: auto-detected)
#   CONFIG              path to SFT YAML (default: configs/sft/qwen3_lora_sft.yaml)
#   CUDA_VISIBLE_DEVICES  which GPUs to use
#   MASTER_PORT         torchrun rendezvous port (default: 29500)

set -euo pipefail

CONFIG="${CONFIG:-configs/sft/qwen3_lora_sft.yaml}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Auto-detect GPU count if NPROC not set.
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
echo "  Multi-GPU SFT run"
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

# Single GPU -> no torchrun overhead.
if [[ "$NPROC" == "1" ]]; then
  exec llamafactory-cli train "$CONFIG"
fi

exec torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  "$(which llamafactory-cli)" train "$CONFIG"
