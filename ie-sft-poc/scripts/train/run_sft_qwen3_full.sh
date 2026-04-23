#!/usr/bin/env bash
# Qwen3-0.6B Full SFT + DeepSpeed — delegates to unified run_sft.sh.
# Usage: NPROC=4 bash scripts/train/run_sft_qwen3_full.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_sft.sh" --model qwen3 --variant full "$@"
