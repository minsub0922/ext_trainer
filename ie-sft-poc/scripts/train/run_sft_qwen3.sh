#!/usr/bin/env bash
# DEPRECATED shim — kept for backward compat.
# Prefer the explicit lora/full launchers:
#   scripts/train/run_sft_qwen3_lora.sh
#   scripts/train/run_sft_qwen3_full.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[deprecated] run_sft_qwen3.sh -> forwarding to run_sft_qwen3_lora.sh" >&2
exec bash "${SCRIPT_DIR}/run_sft_qwen3_lora.sh" "$@"
