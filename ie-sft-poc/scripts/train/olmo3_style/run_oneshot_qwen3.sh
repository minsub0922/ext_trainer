#!/usr/bin/env bash
# OLMo3-style one-shot for Qwen3-0.6B: zero → 4-stage trained model.
#
# Runs the base data pipeline (download → normalize → unify → split → export)
# then the existing 4-stage OLMo3-style pipeline (mid-train → SFT → DPO → RLVR).
# Each phase is skipped if its output already exists, so reruns are cheap
# and resumable.
#
# Usage:
#   bash scripts/train/olmo3_style/run_oneshot_qwen3.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_oneshot_common.sh"

echo "=== OLMo3-style one-shot: Qwen3-0.6B ==="

run_data_pipeline

log "hand off to run_pipeline_qwen3.sh (stages 1–4 + stage-specific data prep)"
exec bash "${SCRIPT_DIR}/run_pipeline_qwen3.sh"
