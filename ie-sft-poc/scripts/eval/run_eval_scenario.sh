#!/usr/bin/env bash
# Shared eval launcher. Dispatches predict + metrics for one scenario
# via evaluate_end_to_end.py.
#
# Supports:
#   - Fine-tuned models: --model qwen3 --variant lora/full
#   - Baseline (pretrained) models: --model qwen3-8b --variant base
#   - Arbitrary HF models: --model-path Qwen/Qwen3-8B --variant base --tag my-tag
#
# Usage:
#   # Fine-tuned LoRA
#   bash scripts/eval/run_eval_scenario.sh --model qwen3 --variant lora --mode unified
#
#   # Baseline (no fine-tuning) — larger model
#   bash scripts/eval/run_eval_scenario.sh --model qwen3-8b --variant base
#
#   # Arbitrary HF model path
#   bash scripts/eval/run_eval_scenario.sh \
#       --model-path Qwen/Qwen3-8B --variant base --tag qwen3-8b
#
# Env overrides:
#   TEST_FILE    (default: data/processed/splits/test.jsonl)
#   BATCH_SIZE   (default: 8)
#   MAX_NEW      (default: 512)
#   LIMIT        (default: 200 — use 0 for all)
#   DTYPE        (default: bf16)
#   OUT_ROOT     (default: outputs/eval)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
cd "$PROJECT_ROOT"

MODEL=""
MODEL_PATH_OVERRIDE=""
VARIANT=""
MODE="unified"
TAG_OVERRIDE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL="$2";              shift 2 ;;
    --model-path) MODEL_PATH_OVERRIDE="$2"; shift 2 ;;
    --variant)    VARIANT="$2";            shift 2 ;;
    --mode)       MODE="$2";               shift 2 ;;
    --tag)        TAG_OVERRIDE="$2";       shift 2 ;;
    --dry-run)    EXTRA_ARGS+=("--dry-run"); shift ;;
    --per-record) EXTRA_ARGS+=("--per-record"); shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

# ---- resolve model ----------------------------------------------------------
# If --model-path is given directly, use it as-is.
# Otherwise, resolve from --model shorthand.
if [[ -n "$MODEL_PATH_OVERRIDE" ]]; then
  [[ -z "$VARIANT" ]] && VARIANT="base"
  BASE="$MODEL_PATH_OVERRIDE"
  TAG="${TAG_OVERRIDE:-$(echo "$MODEL_PATH_OVERRIDE" | tr '/' '-' | tr '[:upper:]' '[:lower:]')}"
else
  [[ -z "$MODEL" ]]   && { echo "ERROR: --model or --model-path required" >&2; exit 2; }
  [[ -z "$VARIANT" ]] && { echo "ERROR: --variant {lora|full|base} required" >&2; exit 2; }

  case "$MODEL" in
    # --- small fine-tuned models ---
    qwen3)            BASE="Qwen/Qwen3-0.6B";           TAG="qwen3-0.6b"       ;;
    qwen3.5)          BASE="Qwen/Qwen3.5-0.8B";         TAG="qwen3.5-0.8b"     ;;
    # --- Qwen3 baseline models (post-trained / instruct) ---
    qwen3-1.7b)       BASE="Qwen/Qwen3-1.7B";           TAG="qwen3-1.7b"       ;;
    qwen3-4b)         BASE="Qwen/Qwen3-4B";             TAG="qwen3-4b"         ;;
    qwen3-8b)         BASE="Qwen/Qwen3-8B";             TAG="qwen3-8b"         ;;
    qwen3-14b)        BASE="Qwen/Qwen3-14B";            TAG="qwen3-14b"        ;;
    qwen3-30b-a3b)    BASE="Qwen/Qwen3-30B-A3B";        TAG="qwen3-30b-a3b"    ;;
    qwen3-32b)        BASE="Qwen/Qwen3-32B";            TAG="qwen3-32b"        ;;
    # --- Qwen3.5 baseline models (actual HF repo IDs) ---
    qwen3.5-4b)       BASE="Qwen/Qwen3.5-4B";           TAG="qwen3.5-4b"       ;;
    qwen3.5-9b)       BASE="Qwen/Qwen3.5-9B";           TAG="qwen3.5-9b"       ;;
    qwen3.5-27b)      BASE="Qwen/Qwen3.5-27B";          TAG="qwen3.5-27b"      ;;
    # --- Base (pretrain-only) variants ---
    qwen3-0.6b-base)  BASE="Qwen/Qwen3-0.6B";           TAG="qwen3-0.6b-base"  ;;
    qwen3-8b-base)    BASE="Qwen/Qwen3-8B-Base";        TAG="qwen3-8b-base"    ;;
    qwen3.5-0.8b-base) BASE="Qwen/Qwen3.5-0.8B";       TAG="qwen3.5-0.8b-base" ;;
    qwen3.5-9b-base)  BASE="Qwen/Qwen3.5-9B-Base";      TAG="qwen3.5-9b-base"  ;;
    *) echo "ERROR: unknown --model: $MODEL" >&2
       echo "  Qwen3 (post-trained): qwen3, qwen3-{1.7b,4b,8b,14b,30b-a3b,32b}" >&2
       echo "  Qwen3.5 (post-trained): qwen3.5, qwen3.5-{4b,9b,27b}" >&2
       echo "  Base (pretrain-only): qwen3-0.6b-base, qwen3-8b-base, qwen3.5-0.8b-base, qwen3.5-9b-base" >&2
       echo "  Or use --model-path <HF_ID> --tag <name> for arbitrary models." >&2
       exit 2 ;;
  esac
  [[ -n "$TAG_OVERRIDE" ]] && TAG="$TAG_OVERRIDE"
fi

# ---- validate variant for larger models -------------------------------------
# Only the small fine-tuned models (qwen3, qwen3.5) support lora/full variants.
case "$MODEL" in
  qwen3|qwen3.5) ;;  # fine-tuned small models: all variants OK
  *)
    if [[ "$VARIANT" != "base" ]]; then
      echo "ERROR: --model $MODEL only supports --variant base (no fine-tuned checkpoint)" >&2
      exit 2
    fi
    ;;
esac

# ---- resolve variant --------------------------------------------------------
ADAPTER_FLAG=""
case "$VARIANT" in
  lora)
    ADAPTER_DIR="outputs/${TAG}-ie-lora"
    ADAPTER_FLAG="--adapter-path ${PROJECT_ROOT}/${ADAPTER_DIR}"
    MODEL_PATH="$BASE"
    ;;
  full)
    ADAPTER_DIR="outputs/${TAG}-ie-full-ds"
    MODEL_PATH="${PROJECT_ROOT}/${ADAPTER_DIR}"
    ;;
  base)
    # No adapter, no local checkpoint — evaluate the pretrained model as-is.
    MODEL_PATH="$BASE"
    ;;
  *) echo "ERROR: --variant must be lora, full, or base" >&2; exit 2 ;;
esac

case "$MODE" in
  kv|entity|relation|unified) : ;;
  *) echo "ERROR: bad --mode: $MODE" >&2; exit 2 ;;
esac

TEST_FILE="${TEST_FILE:-data/processed/splits/test.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW="${MAX_NEW:-512}"
LIMIT="${LIMIT:-200}"
DTYPE="${DTYPE:-bf16}"
OUT_ROOT="${OUT_ROOT:-outputs/eval}"

SCENARIO="${TAG}-${VARIANT}-${MODE}"
OUT_DIR="${PROJECT_ROOT}/${OUT_ROOT}/${SCENARIO}"

echo "==========================================="
echo "  IE Eval scenario: $SCENARIO"
echo "-------------------------------------------"
echo "  MODEL_PATH:   $MODEL_PATH"
echo "  ADAPTER:      ${ADAPTER_FLAG:-<none>}"
echo "  MODE:         $MODE"
echo "  TEST_FILE:    $TEST_FILE"
echo "  OUT_DIR:      $OUT_DIR"
echo "  BATCH_SIZE:   $BATCH_SIZE"
echo "  MAX_NEW:      $MAX_NEW"
echo "  DTYPE:        $DTYPE"
echo "  LIMIT:        $LIMIT"
echo "==========================================="

# shellcheck disable=SC2086
python scripts/eval/evaluate_end_to_end.py \
  --scenario "$SCENARIO" \
  --test "$TEST_FILE" \
  --model-path "$MODEL_PATH" \
  $ADAPTER_FLAG \
  --prompt-mode "$MODE" \
  --output-dir "$OUT_DIR" \
  --max-new-tokens "$MAX_NEW" \
  --batch-size "$BATCH_SIZE" \
  --dtype "$DTYPE" \
  --limit "$LIMIT" \
  "${EXTRA_ARGS[@]}"
