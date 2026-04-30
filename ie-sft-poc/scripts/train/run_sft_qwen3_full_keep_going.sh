#!/usr/bin/env bash
##############################################################################
# Resume Qwen3-0.6B Full SFT from the highest-step checkpoint.
#
# The base full-SFT config has overwrite_output_dir: true, which is right for
# fresh runs but dangerous for resuming. This launcher creates a temporary copy
# of the config with:
#   overwrite_output_dir: false
#   output_dir: <selected output dir>
#   resume_from_checkpoint: <latest checkpoint-*>
#
# Usage:
#   NPROC=4 bash scripts/train/run_sft_qwen3_full_keep_going.sh
#   NPROC=4 bash scripts/train/run_sft_qwen3_full_keep_going.sh --dry-run
#   NPROC=4 bash scripts/train/run_sft_qwen3_full_keep_going.sh --checkpoint outputs/qwen3-0.6b-ie-full-ds/checkpoint-8000
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
cd "$PROJECT_ROOT"

BASE_CONFIG="configs/sft/qwen3_full_sft_ds.yaml"
OUTPUT_DIR="outputs/qwen3-0.6b-ie-full-ds"
CHECKPOINT=""
DRY_RUN=false

usage() {
  cat <<'USAGE'
Usage:
  run_sft_qwen3_full_keep_going.sh [--config PATH] [--output-dir PATH] [--checkpoint PATH] [--dry-run]

Options:
  --config PATH      Base LLaMA-Factory YAML. Default: configs/sft/qwen3_full_sft_ds.yaml
  --output-dir PATH  Directory containing checkpoint-* dirs. Default: outputs/qwen3-0.6b-ie-full-ds
  --checkpoint PATH  Resume from this checkpoint instead of auto-selecting the highest step.
  --dry-run          Print the selected checkpoint and generated config, then exit.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) BASE_CONFIG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$BASE_CONFIG" ]] || { echo "ERROR: config not found: $BASE_CONFIG" >&2; exit 1; }
[[ -d "$OUTPUT_DIR" ]] || { echo "ERROR: output dir not found: $OUTPUT_DIR" >&2; exit 1; }

find_latest_checkpoint() {
  local output_dir="$1"
  find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print \
    | awk -F'checkpoint-' '
        NF > 1 && $NF ~ /^[0-9]+$/ { print $NF "\t" $0 }
      ' \
    | sort -n -k1,1 \
    | tail -n 1 \
    | cut -f2-
}

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(find_latest_checkpoint "$OUTPUT_DIR")"
fi

[[ -n "$CHECKPOINT" ]] || {
  echo "ERROR: no checkpoint-* directories found under $OUTPUT_DIR" >&2
  exit 1
}
[[ -d "$CHECKPOINT" ]] || { echo "ERROR: checkpoint dir not found: $CHECKPOINT" >&2; exit 1; }

TMP_CONFIG="$(mktemp -t qwen3_full_resume_XXXXXX.yaml)"
PATCHED_CONFIG=""
cleanup() {
  rm -f "$TMP_CONFIG"
  if [[ -n "$PATCHED_CONFIG" && "$PATCHED_CONFIG" != "$TMP_CONFIG" && "$PATCHED_CONFIG" == /tmp/* ]]; then
    rm -f "$PATCHED_CONFIG"
  fi
}
trap cleanup EXIT

python3 - "$BASE_CONFIG" "$TMP_CONFIG" "$OUTPUT_DIR" "$CHECKPOINT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
output_dir = sys.argv[3]
checkpoint = sys.argv[4]

lines = base_path.read_text(encoding="utf-8").splitlines()
rendered: list[str] = []
seen_output_dir = False
seen_overwrite = False

for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("resume_from_checkpoint:"):
        continue
    if stripped.startswith("output_dir:"):
        indent = line[: len(line) - len(stripped)]
        rendered.append(f"{indent}output_dir: {json.dumps(output_dir)}")
        seen_output_dir = True
        continue
    if stripped.startswith("overwrite_output_dir:"):
        indent = line[: len(line) - len(stripped)]
        rendered.append(f"{indent}overwrite_output_dir: false")
        seen_overwrite = True
        continue
    rendered.append(line)

if not seen_output_dir:
    rendered.append(f"output_dir: {json.dumps(output_dir)}")
if not seen_overwrite:
    rendered.append("overwrite_output_dir: false")

rendered.append("")
rendered.append("# Added by run_sft_qwen3_full_keep_going.sh")
rendered.append(f"resume_from_checkpoint: {json.dumps(checkpoint)}")

out_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
PY

PATCHED_CONFIG="$(patch_flash_attn_in_yaml "$TMP_CONFIG")"

setup_multigpu

echo "==========================================="
echo "  Qwen3 Full SFT Keep Going"
echo "-------------------------------------------"
echo "  BASE_CONFIG:          $BASE_CONFIG"
echo "  RESUME_CHECKPOINT:    $CHECKPOINT"
echo "  RESUME_CONFIG:        $PATCHED_CONFIG"
echo "  NPROC (GPUs):         $NPROC"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "==========================================="

if [[ "$DRY_RUN" == true ]]; then
  echo "DRY RUN - generated config:"
  sed -n '1,220p' "$PATCHED_CONFIG"
  exit 0
fi

command -v llamafactory-cli >/dev/null || { echo "ERROR: llamafactory-cli not found" >&2; exit 2; }

setup_distributed_env
llamafactory-cli train "$PATCHED_CONFIG"
