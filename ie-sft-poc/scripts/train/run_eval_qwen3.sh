#!/usr/bin/env bash
##############################################################################
# Run evaluation for Qwen3 models
#
# Usage:
#   bash scripts/train/run_eval_qwen3.sh --config path/to/eval_config.yaml [--dry-run]
#
# Examples:
#   bash scripts/train/run_eval_qwen3.sh --config configs/eval/qwen3_eval.yaml
#   bash scripts/train/run_eval_qwen3.sh --config configs/eval/qwen3_eval.yaml --dry-run
##############################################################################

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_PATH=""
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
            echo "Usage: $0 --config path/to/eval_config.yaml [--dry-run]"
            exit 1
            ;;
    esac
done

# Validate config path provided
if [[ -z "$CONFIG_PATH" ]]; then
    echo "ERROR: --config argument required"
    echo "Usage: $0 --config path/to/eval_config.yaml [--dry-run]"
    exit 1
fi

echo "============================================================"
echo "Qwen3 Model Evaluation"
echo "============================================================"
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
echo "Resolved evaluation configuration:"
echo "============================================================"
cat "$CONFIG_PATH"
echo "============================================================"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: Would execute:"
    echo "  llamafactory-cli eval $CONFIG_PATH"
    exit 0
fi

# Run evaluation
echo "Starting Qwen3 evaluation..."
echo ""

llamafactory-cli eval "$CONFIG_PATH"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo "Evaluation completed successfully!"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Evaluation failed with exit code $EXIT_CODE"
    echo "============================================================"
    exit $EXIT_CODE
fi
