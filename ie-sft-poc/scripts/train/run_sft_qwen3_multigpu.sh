#!/usr/bin/env bash
# Multi-GPU SFT for Qwen3 — delegates to unified run_sft.sh.
# Usage: NPROC=4 CONFIG=configs/sft/qwen3_full_sft_ds.yaml bash scripts/train/run_sft_qwen3_multigpu.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-qwen3}"
VARIANT="${VARIANT:-lora}"
exec bash "${SCRIPT_DIR}/run_sft.sh" --model "$MODEL" --variant "$VARIANT" "$@"
