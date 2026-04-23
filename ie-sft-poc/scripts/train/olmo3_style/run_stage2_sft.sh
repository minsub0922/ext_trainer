#!/usr/bin/env bash
# OLMo3-style stage 2 — SFT. Delegates to unified run_stage.sh.
# Usage: MODEL=qwen3 bash scripts/train/olmo3_style/run_stage2_sft.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_stage.sh" --stage 2 "$@"
