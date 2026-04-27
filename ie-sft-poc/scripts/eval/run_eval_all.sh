#!/usr/bin/env bash
# Run the full eval matrix: fine-tuned + baseline models.
#
# Skips scenarios whose checkpoint dir is missing (fine-tuned) or whose
# model can't be resolved (baselines still download on first use).
#
# Env:
#   MODES           mode sweep (default: "unified")
#   MODELS          fine-tuned model sweep (default: "qwen3 qwen3.5")
#   VARIANTS        variant sweep for fine-tuned (default: "lora full")
#   BASELINES       baseline model sweep (default: "qwen3-8b qwen3.5-7b")
#   SKIP_BASELINES  set to 1 to skip baseline evals
#   SKIP_FINETUNED  set to 1 to skip fine-tuned evals
#
# Usage:
#   # Everything (fine-tuned + baselines)
#   bash scripts/eval/run_eval_all.sh
#
#   # Only baselines, unified mode
#   SKIP_FINETUNED=1 BASELINES="qwen3-4b qwen3-8b qwen3.5-7b qwen3.5-14b" \
#     bash scripts/eval/run_eval_all.sh
#
#   # Only fine-tuned, all modes
#   SKIP_BASELINES=1 MODES="kv entity relation unified" \
#     bash scripts/eval/run_eval_all.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODELS="${MODELS:-qwen3 qwen3.5}"
VARIANTS="${VARIANTS:-lora full}"
BASELINES="${BASELINES:-qwen3-8b qwen3.5-4b qwen3.5-9b}"
MODES="${MODES:-unified}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"
SKIP_FINETUNED="${SKIP_FINETUNED:-0}"

declare -a SKIPPED=()
declare -a RAN=()
declare -a FAILED=()

# ---- fine-tuned models ------------------------------------------------------
if [[ "$SKIP_FINETUNED" != "1" ]]; then
  for model in $MODELS; do
    case "$model" in
      qwen3)   TAG="qwen3-0.6b"   ;;
      qwen3.5) TAG="qwen3.5-0.8b" ;;
      *) echo "bad model $model"; continue ;;
    esac

    for variant in $VARIANTS; do
      case "$variant" in
        lora) CKPT="${PROJECT_ROOT}/outputs/${TAG}-ie-lora" ;;
        full) CKPT="${PROJECT_ROOT}/outputs/${TAG}-ie-full-ds" ;;
        *) echo "bad variant $variant"; continue ;;
      esac

      if [[ ! -d "$CKPT" ]]; then
        echo "[skip] $model/$variant — no checkpoint at $CKPT"
        SKIPPED+=("${model}/${variant}")
        continue
      fi

      for mode in $MODES; do
        label="${model}/${variant}/${mode}"
        echo ">>> Running $label"
        if bash "${SCRIPT_DIR}/run_eval_scenario.sh" \
             --model "$model" --variant "$variant" --mode "$mode"; then
          RAN+=("$label")
        else
          FAILED+=("$label")
        fi
      done
    done
  done
fi

# ---- baseline (pretrained) models -------------------------------------------
if [[ "$SKIP_BASELINES" != "1" ]]; then
  for model in $BASELINES; do
    for mode in $MODES; do
      label="${model}/base/${mode}"
      echo ">>> Running $label"
      if bash "${SCRIPT_DIR}/run_eval_scenario.sh" \
           --model "$model" --variant base --mode "$mode"; then
        RAN+=("$label")
      else
        FAILED+=("$label")
      fi
    done
  done
fi

# ---- summary ----------------------------------------------------------------
echo
echo "===== EVAL MATRIX SUMMARY ====="
echo "ran:     ${#RAN[@]}"
printf '  - %s\n' "${RAN[@]:-}"
echo "skipped: ${#SKIPPED[@]}"
printf '  - %s\n' "${SKIPPED[@]:-}"
echo "failed:  ${#FAILED[@]}"
printf '  - %s\n' "${FAILED[@]:-}"

# ---- generate comparison table if any runs succeeded ------------------------
if [[ ${#RAN[@]} -gt 0 ]]; then
  echo
  echo ">>> Generating comparison summary..."
  python "${SCRIPT_DIR}/compare_eval_results.py" \
    --eval-dir "${PROJECT_ROOT}/outputs/eval" 2>/dev/null || true
fi

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
