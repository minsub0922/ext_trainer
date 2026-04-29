#!/usr/bin/env bash
##############################################################################
# Resume the OLMo3-style pipeline from stage 3.
#
# This assumes stage 2 is already trained, then prepares/refreshes preference
# pairs for the selected stage-2 branch, runs DPO, and continues to stage 4
# unless --stage3-only is supplied.
#
# Usage:
#   bash scripts/train/olmo3_style/run_pipeline_from_stage3.sh --model qwen3
#   bash scripts/train/olmo3_style/run_pipeline_from_stage3.sh --model qwen3 --stage2 3ep
#   MODEL=qwen3.5 NPROC=4 bash scripts/train/olmo3_style/run_pipeline_from_stage3.sh --stage2 2ep
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen3}"
STAGE2_VARIANT="2ep"
RUN_STAGE4=1
FORCE_PREFS=0

usage() {
  cat <<'USAGE'
Usage:
  run_pipeline_from_stage3.sh [--model qwen3|qwen3.5] [--stage2 2ep|3ep] [--stage3-only] [--force-preferences]

Options:
  --model              Model family. Defaults to $MODEL or qwen3.
  --stage2             Which completed stage-2 branch to use: 2ep or 3ep. Default: 2ep.
  --stage3-only        Stop after DPO instead of continuing to stage 4 RLVR.
  --force-preferences  Rebuild preference pairs even if the JSONL already exists.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --stage2) STAGE2_VARIANT="$2"; shift 2 ;;
    --stage3-only) RUN_STAGE4=0; shift ;;
    --force-preferences) FORCE_PREFS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$MODEL" in
  qwen3)   TAG="qwen3-0.6b" ;;
  qwen3.5) TAG="qwen3.5-0.8b" ;;
  *) echo "ERROR: bad MODEL=$MODEL (use qwen3 or qwen3.5)" >&2; exit 2 ;;
esac

DATA_DIR="data/processed/olmo3_style"

case "$STAGE2_VARIANT" in
  2ep|default)
    STAGE2_DIR="outputs/olmo3_style/${TAG}/stage2_sft"
    PREF_FILE="${DATA_DIR}/preference_pairs.jsonl"
    PREF_DATASET="ie_pref_pairs"
    STAGE3="3"
    STAGE4="4"
    ;;
  3ep)
    STAGE2_DIR="outputs/olmo3_style/${TAG}/stage2_sft_3ep"
    PREF_FILE="${DATA_DIR}/preference_pairs_3ep.jsonl"
    PREF_DATASET="ie_pref_pairs_3ep"
    STAGE3="3-3ep"
    STAGE4="4-3ep"
    ;;
  *) echo "ERROR: bad --stage2=$STAGE2_VARIANT (use 2ep or 3ep)" >&2; exit 2 ;;
esac

[[ -d "$STAGE2_DIR" ]] || {
  echo "ERROR: stage2 checkpoint not found: ${STAGE2_DIR}" >&2
  echo "Train the selected stage2 branch first, then rerun this script." >&2
  exit 1
}

echo "=== OLMo3-style resume from stage 3: ${TAG} (stage2=${STAGE2_VARIANT}) ==="

if [[ "$FORCE_PREFS" -eq 1 ]]; then
  rm -f "$PREF_FILE"
fi

if [[ ! -s "$PREF_FILE" ]]; then
  echo "[prep] building preference pairs from ${STAGE2_DIR}"
  python scripts/preprocess/olmo3_style/build_preference_pairs.py \
    --model-path "$STAGE2_DIR" \
    --test data/processed/splits/train.jsonl \
    --output "$PREF_FILE" \
    --dataset-name "$PREF_DATASET" \
    --k 4 --limit 2000
else
  echo "[prep] found existing preference pairs: ${PREF_FILE}"
  python scripts/preprocess/olmo3_style/build_preference_pairs.py \
    --output "$PREF_FILE" \
    --dataset-name "$PREF_DATASET" \
    --register-only
fi

bash "${SCRIPT_DIR}/run_stage.sh" --stage "$STAGE3" --model "$MODEL"

if [[ "$RUN_STAGE4" -eq 0 ]]; then
  echo "=== stopped after stage 3 (${TAG}, stage2=${STAGE2_VARIANT}) ==="
  exit 0
fi

if [[ ! -f "${DATA_DIR}/rlvr_prompts.jsonl" ]]; then
  echo "[prep] building RLVR prompts"
  python scripts/preprocess/olmo3_style/build_rlvr_prompts.py \
    --input data/processed/splits/train.jsonl \
    --output "${DATA_DIR}/rlvr_prompts.jsonl" \
    --limit 4000
fi

bash "${SCRIPT_DIR}/run_stage.sh" --stage "$STAGE4" --model "$MODEL"

echo "=== OLMo3-style stage 3+ pipeline complete for ${TAG} (stage2=${STAGE2_VARIANT}) ==="
