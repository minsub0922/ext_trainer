#!/usr/bin/env bash
# OLMo3 POC SFT — delegates to unified run_sft.sh.
# Usage: bash scripts/train/run_sft_olmo3_poc.sh [--dry-run]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_sft.sh" --model olmo3 --variant lora "$@"
