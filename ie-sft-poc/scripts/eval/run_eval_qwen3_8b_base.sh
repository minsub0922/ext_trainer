#!/usr/bin/env bash
# Eval Qwen3-8B-Base.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-unified}"
exec bash "${SCRIPT_DIR}/run_eval_scenario.sh" --model qwen3-8b --variant base --mode "$MODE" "$@"
