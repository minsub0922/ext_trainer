#!/usr/bin/env bash
# OLMo3-style 3-epoch stage-2 branch for Qwen3.5-0.8B.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_pipeline_stage2_3ep.sh" --model qwen3.5 "$@"
