#!/usr/bin/env bash
# Eval Qwen3-0.6B OLMo3-style stage 4 RLVR after stage2 3ep.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-unified}"
exec bash "${SCRIPT_DIR}/run_eval_scenario.sh" --model qwen3 --variant olmo3-stage4-3ep --mode "$MODE" "$@"
