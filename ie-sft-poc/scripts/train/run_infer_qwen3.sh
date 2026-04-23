#!/usr/bin/env bash
##############################################################################
# Run inference with a trained model via LLaMA-Factory CLI.
#
# Usage:
#   bash scripts/train/run_infer_qwen3.sh \
#     --model-path Qwen/Qwen3-0.6B \
#     --input "Extract entities from: John works at Google" \
#     [--adapter-path outputs/qwen3-0.6b-ie-lora] \
#     [--max-length 2048]
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
cd "$PROJECT_ROOT"

MODEL_PATH=""
INPUT_TEXT=""
ADAPTER_PATH=""
MAX_LENGTH=2048

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)   MODEL_PATH="$2";   shift 2 ;;
    --input)        INPUT_TEXT="$2";    shift 2 ;;
    --adapter-path) ADAPTER_PATH="$2"; shift 2 ;;
    --max-length)   MAX_LENGTH="$2";   shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]] || [[ -z "$INPUT_TEXT" ]]; then
  echo "ERROR: --model-path and --input are required" >&2
  exit 1
fi

echo "==========================================="
echo "  Model Inference"
echo "-------------------------------------------"
echo "  Model:      $MODEL_PATH"
echo "  Max length: $MAX_LENGTH"
[[ -n "$ADAPTER_PATH" ]] && echo "  Adapter:    $ADAPTER_PATH"
echo "  Input:      ${INPUT_TEXT:0:80}..."
echo "==========================================="

command -v llamafactory-cli >/dev/null || { echo "ERROR: llamafactory-cli not found" >&2; exit 2; }

# Build the inference config
INFER_CONFIG_DIR=$(mktemp -d)
INFER_CONFIG="$INFER_CONFIG_DIR/infer_config.yaml"
trap 'rm -rf "$INFER_CONFIG_DIR"' EXIT

FA_MODE="$(detect_flash_attn)"

cat > "$INFER_CONFIG" << EOF
model_name_or_path: $MODEL_PATH
template: qwen
cutoff_len: $MAX_LENGTH
flash_attn: $FA_MODE
EOF

if [[ -n "$ADAPTER_PATH" ]]; then
  echo "adapter_name_or_path: $ADAPTER_PATH" >> "$INFER_CONFIG"
fi

echo ""
echo "Note: Use 'llamafactory-cli chat $INFER_CONFIG' for interactive inference."
echo "Config generated at: $INFER_CONFIG"
echo ""
cat "$INFER_CONFIG"

# For a non-interactive single-query, use the Python eval script instead:
echo ""
echo "For batch inference, use:"
echo "  python scripts/eval/run_predict.py --model-path $MODEL_PATH --input <test.jsonl> --output <out.jsonl>"
