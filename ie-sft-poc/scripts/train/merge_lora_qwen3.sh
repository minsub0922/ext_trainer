#!/usr/bin/env bash
##############################################################################
# Merge LoRA adapter back into base model.
#
# Usage:
#   bash scripts/train/merge_lora_qwen3.sh \
#     --model-path Qwen/Qwen3-0.6B \
#     --adapter-path outputs/qwen3-0.6b-ie-lora \
#     --output-dir outputs/qwen3-0.6b-ie-merged
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
cd "$PROJECT_ROOT"

MODEL_PATH=""
ADAPTER_PATH=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)   MODEL_PATH="$2";   shift 2 ;;
    --adapter-path) ADAPTER_PATH="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$MODEL_PATH" ]] || [[ -z "$ADAPTER_PATH" ]] || [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: --model-path, --adapter-path, and --output-dir are required" >&2
  exit 1
fi

# Resolve adapter path
[[ ! "$ADAPTER_PATH" = /* ]] && ADAPTER_PATH="${PROJECT_ROOT}/${ADAPTER_PATH}"
[[ ! -d "$ADAPTER_PATH" ]] && { echo "ERROR: Adapter dir not found: $ADAPTER_PATH" >&2; exit 1; }

command -v llamafactory-cli >/dev/null || { echo "ERROR: llamafactory-cli not found" >&2; exit 2; }

echo "==========================================="
echo "  Merge LoRA Adapter"
echo "-------------------------------------------"
echo "  Base model:  $MODEL_PATH"
echo "  Adapter:     $ADAPTER_PATH"
echo "  Output:      $OUTPUT_DIR"
echo "==========================================="

MERGE_CONFIG_DIR=$(mktemp -d)
MERGE_CONFIG="$MERGE_CONFIG_DIR/merge_config.yaml"
trap 'rm -rf "$MERGE_CONFIG_DIR"' EXIT

cat > "$MERGE_CONFIG" << EOF
model_name_or_path: $MODEL_PATH
adapter_name_or_path: $ADAPTER_PATH
export_dir: $OUTPUT_DIR
export_quantization_bit: null
export_quantization_dataset: null
export_device: cuda
export_legacy_format: false
EOF

exec llamafactory-cli export "$MERGE_CONFIG"
