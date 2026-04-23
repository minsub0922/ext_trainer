#!/usr/bin/env bash
# OLMo3-style stage 4 — RLVR. Delegates to unified run_stage.sh.
# Usage: MODEL=qwen3 bash scripts/train/olmo3_style/run_stage4_rlvr.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_stage.sh" --stage 4 "$@"
