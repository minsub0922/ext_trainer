#!/usr/bin/env bash
# Eval Qwen3.5-0.8B OLMo3-style stage 3 DPO after stage2 3ep.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-unified}"
exec bash "${SCRIPT_DIR}/run_eval_scenario.sh" --model qwen3.5 --variant olmo3-stage3-3ep --mode "$MODE" "$@"
