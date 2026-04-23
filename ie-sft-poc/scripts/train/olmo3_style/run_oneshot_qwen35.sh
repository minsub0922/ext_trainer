#!/usr/bin/env bash
# OLMo3-style one-shot for Qwen3.5-0.8B: zero → 4-stage trained model.
# Runs data pipeline then delegates to unified run_pipeline.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_oneshot_common.sh"

echo "=== OLMo3-style one-shot: Qwen3.5-0.8B ==="
run_data_pipeline

log "hand off to run_pipeline.sh (stages 1–4 + stage-specific data prep)"
exec bash "${SCRIPT_DIR}/run_pipeline.sh" --model qwen3.5
