#!/usr/bin/env bash
##############################################################################
# Run LLaMA-Factory eval for a given config.
#
# Usage:
#   bash scripts/train/run_eval_qwen3.sh --config configs/eval/qwen3_eval.yaml
#   bash scripts/train/run_eval_qwen3.sh --config configs/eval/qwen3_eval.yaml --dry-run
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
cd "$PROJECT_ROOT"

CONFIG_PATH=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$CONFIG_PATH" ]] && { echo "ERROR: --config required" >&2; exit 1; }
[[ ! "$CONFIG_PATH" = /* ]] && CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
[[ ! -f "$CONFIG_PATH" ]] && { echo "ERROR: config not found: $CONFIG_PATH" >&2; exit 1; }
command -v llamafactory-cli >/dev/null || { echo "ERROR: llamafactory-cli not found" >&2; exit 2; }

# Flash attention check
CONFIG_PATH="$(patch_flash_attn_in_yaml "$CONFIG_PATH")"

echo "=== LLaMA-Factory Eval ==="
echo "  CONFIG: $CONFIG_PATH"

if [[ "$DRY_RUN" == true ]]; then
  echo "DRY RUN: llamafactory-cli eval $CONFIG_PATH"
  exit 0
fi

exec llamafactory-cli eval "$CONFIG_PATH"
