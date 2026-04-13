#!/usr/bin/env bash
##############################################################################
# Run inference with Qwen3 model
#
# Usage:
#   bash scripts/train/run_infer_qwen3.sh --model-path MODEL_PATH --input "text" [--adapter-path ADAPTER_PATH] [--max-length LENGTH]
#
# Examples:
#   bash scripts/train/run_infer_qwen3.sh --model-path Qwen/Qwen3-0.6B --input "Extract entities from: John works at Google"
#   bash scripts/train/run_infer_qwen3.sh --model-path Qwen/Qwen3-0.6B --input "..." --adapter-path outputs/qwen3-0.6b-ie-lora/adapter_model.bin
#   bash scripts/train/run_infer_qwen3.sh --model-path Qwen/Qwen3-0.6B --input "..." --max-length 1024
##############################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH=""
INPUT_TEXT=""
ADAPTER_PATH=""
MAX_LENGTH=2048

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --input)
            INPUT_TEXT="$2"
            shift 2
            ;;
        --adapter-path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model-path MODEL_PATH --input TEXT [--adapter-path ADAPTER_PATH] [--max-length LENGTH]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]] || [[ -z "$INPUT_TEXT" ]]; then
    echo "ERROR: --model-path and --input are required"
    echo "Usage: $0 --model-path MODEL_PATH --input TEXT [--adapter-path ADAPTER_PATH] [--max-length LENGTH]"
    exit 1
fi

echo "============================================================"
echo "Qwen3 Model Inference"
echo "============================================================"
echo ""
echo "Model path: $MODEL_PATH"
echo "Max length: $MAX_LENGTH"
if [[ -n "$ADAPTER_PATH" ]]; then
    echo "Adapter path: $ADAPTER_PATH"
fi
echo ""
echo "Input:"
echo "  $INPUT_TEXT"
echo ""

# Check llamafactory-cli is installed
if ! command -v llamafactory-cli &> /dev/null; then
    echo "ERROR: llamafactory-cli not found in PATH"
    echo "Install with: pip install llamafactory"
    exit 1
fi

echo "Running inference..."
echo "============================================================"
echo ""

# Build command
CMD="llamafactory-cli api \
  --model_name_or_path \"$MODEL_PATH\" \
  --template qwen \
  --max_length $MAX_LENGTH"

if [[ -n "$ADAPTER_PATH" ]]; then
    CMD="$CMD --adapter_name_or_path \"$ADAPTER_PATH\""
fi

echo "Note: This is a placeholder. Use LLaMA-Factory API or CLI directly for actual inference."
echo ""
echo "Example command:"
echo "  $CMD"
echo ""
