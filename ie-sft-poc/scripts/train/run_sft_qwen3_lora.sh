#!/usr/bin/env bash
# Qwen3-0.6B LoRA SFT — delegates to unified run_sft.sh.
# Usage: bash scripts/train/run_sft_qwen3_lora.sh [--dry-run]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_sft.sh" --model qwen3 --variant lora "$@"
