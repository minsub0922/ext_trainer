#!/usr/bin/env bash
# One-shot: zero → OLMo3 PoC SFT.
# Runs the data pipeline then the OLMo3 single-stage PoC SFT recipe.
# For the full 4-stage OLMo3-style pipeline (midtrain → SFT → DPO → RLVR)
# use scripts/train/olmo3_style/run_pipeline_qwen3.sh instead.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

CONFIG="${CONFIG:-configs/sft/olmo3_poc_sft.yaml}"

echo "=== OLMo3 PoC one-shot ==="
echo "  CONFIG: $CONFIG"

run_data_pipeline
run_train "$CONFIG"
