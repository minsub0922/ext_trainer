#!/usr/bin/env bash
# OLMo3-style stage 2 — 3-epoch SFT ablation. Delegates to run_stage.sh.
# Usage: MODEL=qwen3 bash scripts/train/olmo3_style/run_stage2_sft_3ep.sh
#        MODEL=qwen3.5 bash scripts/train/olmo3_style/run_stage2_sft_3ep.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_stage.sh" --stage 2-3ep "$@"
