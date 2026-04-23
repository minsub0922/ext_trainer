#!/usr/bin/env bash
# Compare arbitrary base models or local checkpoints with the IE eval pipeline.
#
# Examples:
#   bash scripts/eval/run_eval_compare_models.sh \
#     --candidate qwen35-large=<hf-model-id> \
#     --candidate tuned=outputs/qwen3.5-0.8b-ie-full-ds
#
#   bash scripts/eval/run_eval_compare_models.sh \
#     --candidate base=Qwen/Qwen3.5-0.8B \
#     --candidate lora=Qwen/Qwen3.5-0.8B::outputs/qwen3.5-0.8b-ie-lora \
#     --limit 50

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
cd "$PROJECT_ROOT"

exec python scripts/eval/run_eval_compare_models.py "$@"
