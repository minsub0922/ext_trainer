#!/usr/bin/env bash
##############################################################################
# Run SFT training for OLMo3 POC model with LoRA
#
# WARNING: This is a POC configuration. The model path may need updating
# once the actual OLMo3 1B model is available.
#
# Usage:
#   bash scripts/train/run_sft_olmo3_poc.sh [--config path/to/config.yaml] [--dry-run]
#
# Environment variables:
#   CONFIG_PATH: Path to training config (default: configs/sft/olmo3_poc_sft.yaml)
#
# Examples:
#   bash scripts/train/run_sft_olmo3_poc.sh
#   bash scripts/train/run_sft_olmo3_poc.sh --config configs/sft/custom_olmo3.yaml
#   bash scripts/train/run_sft_olmo3_poc.sh --dry-run
##############################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default config path
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/sft/olmo3_poc_sft.yaml}"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config path/to/config.yaml] [--dry-run]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "OLMo3 POC LoRA SFT Training"
echo "============================================================"
echo ""
echo "WARNING: This is a POC configuration!"
echo "  - Model path may need updating for actual OLMo3 model"
echo "  - Template compatibility should be verified"
echo "  - Test carefully before production use"
echo ""

# Resolve config path to absolute
if [[ ! "$CONFIG_PATH" = /* ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi

# Check config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "Config path: $CONFIG_PATH"
echo "Project root: $PROJECT_ROOT"
echo "Dry run: $DRY_RUN"
echo ""

# Check llamafactory-cli is installed
if ! command -v llamafactory-cli &> /dev/null; then
    echo "ERROR: llamafactory-cli not found in PATH"
    echo "Install with: pip install llamafactory"
    exit 1
fi

echo "llamafactory-cli found: $(which llamafactory-cli)"
echo ""

# Print resolved config
echo "Resolved training configuration:"
echo "============================================================"
cat "$CONFIG_PATH"
echo "============================================================"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: Would execute:"
    echo "  llamafactory-cli train $CONFIG_PATH"
    exit 0
fi

# Run training
echo "Starting OLMo3 POC LoRA SFT training..."
echo ""

llamafactory-cli train "$CONFIG_PATH"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo "Training completed successfully!"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Training failed with exit code $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
