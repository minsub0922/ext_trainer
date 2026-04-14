#!/usr/bin/env bash
# Run the full eval matrix: {qwen3, qwen3.5} x {lora, full} x {kv, entity, relation, unified}.
# Skips scenarios whose checkpoint dir is missing so partial matrices work.
#
# Env:
#   MODES       override the mode sweep (space-separated)
#   MODELS      override the model sweep: "qwen3 qwen3.5"
#   VARIANTS    override the variant sweep: "lora full"
#
# Usage:
#   bash scripts/eval/run_eval_all.sh
#   MODES="unified kv" bash scripts/eval/run_eval_all.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODELS="${MODELS:-qwen3 qwen3.5}"
VARIANTS="${VARIANTS:-lora full}"
MODES="${MODES:-kv entity relation unified}"

declare -a SKIPPED=()
declare -a RAN=()
declare -a FAILED=()

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

echo
echo "===== EVAL MATRIX SUMMARY ====="
echo "ran:     ${#RAN[@]}"
printf '  - %s\n' "${RAN[@]:-}"
echo "skipped: ${#SKIPPED[@]}"
printf '  - %s\n' "${SKIPPED[@]:-}"
echo "failed:  ${#FAILED[@]}"
printf '  - %s\n' "${FAILED[@]:-}"

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
