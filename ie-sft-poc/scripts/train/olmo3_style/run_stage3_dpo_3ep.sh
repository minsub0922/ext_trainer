#!/usr/bin/env bash
# OLMo3-style stage 3 — DPO branch after the 3-epoch stage 2 SFT.
# Usage: MODEL=qwen3 bash scripts/train/olmo3_style/run_stage3_dpo_3ep.sh
#        MODEL=qwen3.5 bash scripts/train/olmo3_style/run_stage3_dpo_3ep.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_stage.sh" --stage 3-3ep "$@"
