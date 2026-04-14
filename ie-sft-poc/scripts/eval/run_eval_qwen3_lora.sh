#!/usr/bin/env bash
# Eval Qwen3-0.6B + LoRA adapter. Defaults to unified mode.
#   MODE=kv bash scripts/eval/run_eval_qwen3_lora.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-unified}"
exec bash "${SCRIPT_DIR}/run_eval_scenario.sh" --model qwen3 --variant lora --mode "$MODE" "$@"
