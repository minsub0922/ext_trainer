#!/usr/bin/env bash
# Shared eval launcher. Dispatches predict + metrics for one (model,variant,mode)
# scenario via evaluate_end_to_end.py.
#
# Usage (direct):
#   bash scripts/eval/run_eval_scenario.sh \
#       --model qwen3   --variant lora --mode unified
#   bash scripts/eval/run_eval_scenario.sh \
#       --model qwen3.5 --variant full --mode kv
#
# Or use the thin wrappers (run_eval_qwen3_lora.sh, ...).
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
VARIANT=""
MODE="unified"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)    MODEL="$2";   shift 2 ;;
    --variant)  VARIANT="$2"; shift 2 ;;
    --mode)     MODE="$2";    shift 2 ;;
    --dry-run)  EXTRA_ARGS+=("--dry-run"); shift ;;
    --per-record) EXTRA_ARGS+=("--per-record"); shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$MODEL"   ]] && { echo "--model {qwen3|qwen3.5} required"   >&2; exit 2; }
[[ -z "$VARIANT" ]] && { echo "--variant {lora|full} required"     >&2; exit 2; }

case "$MODEL" in
  qwen3)   BASE="Qwen/Qwen3-0.6B";    TAG="qwen3-0.6b"  ;;
  qwen3.5) BASE="Qwen/Qwen3.5-0.8B";  TAG="qwen3.5-0.8b";;
  *) echo "bad --model: $MODEL" >&2; exit 2 ;;
esac

case "$VARIANT" in
  lora) ADAPTER_DIR="outputs/${TAG}-ie-lora";     ADAPTER_FLAG="--adapter-path ${PROJECT_ROOT}/${ADAPTER_DIR}" ;;
  full) ADAPTER_DIR="outputs/${TAG}-ie-full-ds";  ADAPTER_FLAG="" ;;
  *) echo "bad --variant: $VARIANT" >&2; exit 2 ;;
esac

case "$MODE" in
  kv|entity|relation|unified) : ;;
  *) echo "bad --mode: $MODE" >&2; exit 2 ;;
esac

TEST_FILE="${TEST_FILE:-data/processed/splits/test.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW="${MAX_NEW:-512}"
LIMIT="${LIMIT:-200}"
DTYPE="${DTYPE:-bf16}"
OUT_ROOT="${OUT_ROOT:-outputs/eval}"

SCENARIO="${TAG}-${VARIANT}-${MODE}"
OUT_DIR="${PROJECT_ROOT}/${OUT_ROOT}/${SCENARIO}"

# Decide model path: for full SFT use the local merged dir; for LoRA use the
# hub base + adapter.
if [[ "$VARIANT" == "full" ]]; then
  MODEL_PATH="${PROJECT_ROOT}/${ADAPTER_DIR}"
else
  MODEL_PATH="$BASE"
fi

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
