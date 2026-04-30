#!/usr/bin/env bash
# Qwen3-4B Full SFT — delegates to unified run_sft.sh.
# Usage: bash scripts/train/run_sft_qwen3_4b_full.sh [--dry-run]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_sft.sh" --model qwen3-4b --variant full "$@"
