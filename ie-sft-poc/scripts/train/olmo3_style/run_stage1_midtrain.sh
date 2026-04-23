#!/usr/bin/env bash
# OLMo3-style stage 1 — mid-training. Delegates to unified run_stage.sh.
# Usage: MODEL=qwen3 bash scripts/train/olmo3_style/run_stage1_midtrain.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_stage.sh" --stage 1 "$@"
