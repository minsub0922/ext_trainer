#!/usr/bin/env bash
##############################################################################
# Merge LoRA adapter back into base model for Qwen3
#
# Usage:
#   bash scripts/train/merge_lora_qwen3.sh --model-path MODEL_PATH --adapter-path ADAPTER_PATH --output-dir OUTPUT_DIR
#
# This exports the fine-tuned model by merging LoRA weights with the base model.
#
# Examples:
#   bash scripts/train/merge_lora_qwen3.sh \
#     --model-path Qwen/Qwen3-0.6B \
#     --adapter-path outputs/qwen3-0.6b-ie-lora \
#     --output-dir outputs/qwen3-0.6b-ie-merged
##############################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH=""
ADAPTER_PATH=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --adapter-path)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model-path MODEL_PATH --adapter-path ADAPTER_PATH --output-dir OUTPUT_DIR"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]] || [[ -z "$ADAPTER_PATH" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --model-path, --adapter-path, and --output-dir are required"
    echo "Usage: $0 --model-path MODEL_PATH --adapter-path ADAPTER_PATH --output-dir OUTPUT_DIR"
    exit 1
fi

echo "============================================================"
echo "Merge LoRA Adapter into Base Model"
echo "============================================================"
echo ""
echo "Base model: $MODEL_PATH"
echo "LoRA adapter: $ADAPTER_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Resolve adapter path to absolute if needed
if [[ ! "$ADAPTER_PATH" = /* ]]; then
    ADAPTER_PATH="${PROJECT_ROOT}/${ADAPTER_PATH}"
fi

# Check adapter directory exists
if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "ERROR: Adapter directory not found: $ADAPTER_PATH"
    exit 1
fi

# Check llamafactory-cli is installed
if ! command -v llamafactory-cli &> /dev/null; then
    echo "ERROR: llamafactory-cli not found in PATH"
    echo "Install with: pip install llamafactory"
    exit 1
fi

echo "llamafactory-cli found: $(which llamafactory-cli)"
echo ""

# Create a temporary merge config
MERGE_CONFIG_DIR=$(mktemp -d)
MERGE_CONFIG="$MERGE_CONFIG_DIR/merge_config.yaml"

cat > "$MERGE_CONFIG" << EOF
### Temporary merge configuration
model_name_or_path: $MODEL_PATH
adapter_name_or_path: $ADAPTER_PATH
export_dir: $OUTPUT_DIR
export_quantization_bit: null
export_quantization_dataset: null
export_device: cuda
export_legacy_format: false
EOF

echo "Merge configuration:"
echo "============================================================"
cat "$MERGE_CONFIG"
echo "============================================================"
echo ""

echo "Starting LoRA merge..."
echo ""

llamafactory-cli export "$MERGE_CONFIG"

EXIT_CODE=$?

# Clean up temporary config
rm -rf "$MERGE_CONFIG_DIR"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo "LoRA merge completed successfully!"
    echo "Merged model saved to: $OUTPUT_DIR"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "LoRA merge failed with exit code $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
