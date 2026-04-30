#!/usr/bin/env bash
##############################################################################
# Unified OLMo3-style 4-stage pipeline — replaces model-specific variants.
#
# Runs: data prep → stage 1 (mid-training) → stage 2 (SFT) →
#       preference pair prep → stage 3 (DPO) →
#       RLVR prompt prep → stage 4 (RLVR)
#
# Usage:
#   bash scripts/train/olmo3_style/run_pipeline.sh --model qwen3
#   bash scripts/train/olmo3_style/run_pipeline.sh --model qwen3.5
#   MODEL=qwen3 bash scripts/train/olmo3_style/run_pipeline.sh
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen3}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

case "$MODEL" in
  qwen3)    TAG="qwen3-0.6b"   ;;
  qwen3-4b) TAG="qwen3-4b"     ;;
  qwen3.5)  TAG="qwen3.5-0.8b" ;;
  *) echo "ERROR: bad MODEL=$MODEL" >&2; exit 2 ;;
esac

export MODEL
DATA_DIR="data/processed/olmo3_style"

echo "=== OLMo3-style pipeline: ${TAG} ==="

# ----- data prep: midtrain mixture ------------------------------------------
if [[ ! -f "${DATA_DIR}/midtrain.jsonl" ]]; then
  echo "[prep] building midtrain mixture"
  python scripts/preprocess/olmo3_style/build_midtrain_mixture.py \
    --input  data/processed/splits/train.jsonl \
    --output "${DATA_DIR}/midtrain.jsonl"
else
  echo "[prep] refreshing midtrain dataset registration"
  python scripts/preprocess/olmo3_style/build_midtrain_mixture.py \
    --output "${DATA_DIR}/midtrain.jsonl" \
    --register-only
fi

# ----- stage 1 ---------------------------------------------------------------
bash "${SCRIPT_DIR}/run_stage.sh" --stage 1 --model "$MODEL"

# ----- stage 2 ---------------------------------------------------------------
bash "${SCRIPT_DIR}/run_stage.sh" --stage 2 --model "$MODEL"

# ----- prep preference pairs (needs stage-2 ckpt) ----------------------------
if [[ ! -s "${DATA_DIR}/preference_pairs.jsonl" ]]; then
  echo "[prep] building preference pairs"
  python scripts/preprocess/olmo3_style/build_preference_pairs.py \
    --model-path "outputs/olmo3_style/${TAG}/stage2_sft" \
    --test data/processed/splits/train.jsonl \
    --output "${DATA_DIR}/preference_pairs.jsonl" \
    --k 4 --limit 2000
else
  echo "[prep] found existing preference pairs: ${DATA_DIR}/preference_pairs.jsonl"
  python scripts/preprocess/olmo3_style/build_preference_pairs.py \
    --output "${DATA_DIR}/preference_pairs.jsonl" \
    --register-only
fi

# ----- stage 3 ---------------------------------------------------------------
bash "${SCRIPT_DIR}/run_stage.sh" --stage 3 --model "$MODEL"

# ----- prep RLVR prompts -----------------------------------------------------
if [[ ! -f "${DATA_DIR}/rlvr_prompts.jsonl" ]]; then
  echo "[prep] building RLVR prompts"
  python scripts/preprocess/olmo3_style/build_rlvr_prompts.py \
    --input  data/processed/splits/train.jsonl \
    --output "${DATA_DIR}/rlvr_prompts.jsonl" \
    --limit 4000
fi

# ----- stage 4 ---------------------------------------------------------------
bash "${SCRIPT_DIR}/run_stage.sh" --stage 4 --model "$MODEL"

echo "=== OLMo3-style pipeline complete for ${TAG} ==="
