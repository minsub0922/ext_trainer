#!/usr/bin/env bash
# Qwen3.5-0.8B LoRA SFT — delegates to unified run_sft.sh.
# Usage: bash scripts/train/run_sft_qwen35_lora.sh [--dry-run]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_sft.sh" --model qwen3.5 --variant lora "$@"
