#!/usr/bin/env bash
##############################################################################
# OLMo3-style 3-epoch stage-2 branch.
#
# Continues from an existing stage-1 midtrain checkpoint, then runs:
#   stage 2 SFT (3ep) -> 3ep preference pairs -> stage 3 DPO -> stage 4 RLVR
#
# Usage:
#   bash scripts/train/olmo3_style/run_pipeline_stage2_3ep.sh --model qwen3
#   bash scripts/train/olmo3_style/run_pipeline_stage2_3ep.sh --model qwen3.5
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
  qwen3)   TAG="qwen3-0.6b" ;;
  qwen3.5) TAG="qwen3.5-0.8b" ;;
  *) echo "ERROR: bad MODEL=$MODEL" >&2; exit 2 ;;
esac

export MODEL
DATA_DIR="data/processed/olmo3_style"
STAGE1_DIR="outputs/olmo3_style/${TAG}/stage1_midtrain"
PREF_3EP="${DATA_DIR}/preference_pairs_3ep.jsonl"

echo "=== OLMo3-style stage2-3ep pipeline: ${TAG} ==="

if [[ ! -d "$STAGE1_DIR" ]]; then
  echo "ERROR: stage1 checkpoint not found: ${STAGE1_DIR}" >&2
  echo "Run stage1 first, then rerun this 3ep branch pipeline." >&2
  exit 1
fi

# ----- stage 2, 3 epochs -----------------------------------------------------
bash "${SCRIPT_DIR}/run_stage.sh" --stage 2-3ep --model "$MODEL"

# ----- prep 3ep preference pairs (needs stage2_sft_3ep ckpt) -----------------
if [[ ! -s "$PREF_3EP" ]]; then
  echo "[prep] building 3ep preference pairs"
  python scripts/preprocess/olmo3_style/build_preference_pairs.py \
    --model-path "outputs/olmo3_style/${TAG}/stage2_sft_3ep" \
    --test data/processed/splits/train.jsonl \
    --output "$PREF_3EP" \
    --dataset-name ie_pref_pairs_3ep \
    --k 4 --limit 2000
else
  echo "[prep] found existing 3ep preference pairs: ${PREF_3EP}"
  python scripts/preprocess/olmo3_style/build_preference_pairs.py \
    --output "$PREF_3EP" \
    --dataset-name ie_pref_pairs_3ep \
    --register-only
fi

# ----- stage 3, separated 3ep branch ----------------------------------------
bash "${SCRIPT_DIR}/run_stage.sh" --stage 3-3ep --model "$MODEL"

# ----- prep RLVR prompts -----------------------------------------------------
if [[ ! -f "${DATA_DIR}/rlvr_prompts.jsonl" ]]; then
  echo "[prep] building RLVR prompts"
  python scripts/preprocess/olmo3_style/build_rlvr_prompts.py \
    --input  data/processed/splits/train.jsonl \
    --output "${DATA_DIR}/rlvr_prompts.jsonl" \
    --limit 4000
fi

# ----- stage 4, separated 3ep branch ----------------------------------------
bash "${SCRIPT_DIR}/run_stage.sh" --stage 4-3ep --model "$MODEL"

echo "=== OLMo3-style stage2-3ep pipeline complete for ${TAG} ==="
