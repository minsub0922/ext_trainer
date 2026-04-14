#!/usr/bin/env bash
# Shared helpers for one-shot train launchers.
#
# Source this file; it defines:
#   - PROJECT_ROOT       (absolute)
#   - DATA_RAW / DATA_SPLITS / DATA_LF   (conventional locations)
#   - run_data_pipeline  (idempotent: download → normalize → unify → split → export)
#   - run_train          (exec llamafactory-cli train "$CONFIG")
#
# Each phase is skipped if its expected output already exists.

set -euo pipefail

SCRIPT_DIR_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR_COMMON}/../../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_RAW="data/raw/instructie"
DATA_CANONICAL="data/processed/canonical"
DATA_UNIFIED="data/processed/unified.jsonl"
DATA_SPLITS="data/processed/splits"
DATA_LF="data/processed/llamafactory"

log()  { printf '\n[oneshot] %s\n' "$*"; }
have() { [[ -e "$1" ]]; }

run_data_pipeline() {
  # 1. download
  if ! have "${DATA_RAW}/train.jsonl" && ! have "${DATA_RAW}"/*.jsonl 2>/dev/null; then
    log "download InstructIE"
    python scripts/download/download_instructie.py --output-dir "${DATA_RAW}"
  else
    log "skip download (raw data present)"
  fi

  # 2. normalize → canonical
  if ! have "${DATA_CANONICAL}/instructie.jsonl"; then
    log "normalize InstructIE → canonical"
    mkdir -p "${DATA_CANONICAL}"
    python scripts/preprocess/normalize_instructie.py \
      --input  "${DATA_RAW}" \
      --output "${DATA_CANONICAL}/instructie.jsonl"
  else
    log "skip normalize (canonical present)"
  fi

  # 3. unify
  if ! have "${DATA_UNIFIED}"; then
    log "unify canonical datasets"
    python scripts/preprocess/unify_ie_datasets.py \
      --input  "${DATA_CANONICAL}"/*.jsonl \
      --output "${DATA_UNIFIED}"
  else
    log "skip unify (${DATA_UNIFIED} present)"
  fi

  # 4. split
  if ! have "${DATA_SPLITS}/train.jsonl"; then
    log "split into train/dev/test"
    mkdir -p "${DATA_SPLITS}"
    python scripts/export/export_train_dev_test.py \
      --input  "${DATA_UNIFIED}" \
      --output "${DATA_SPLITS}/"
  else
    log "skip split (splits present)"
  fi

  # 5. export to LLaMA-Factory
  if ! have "${DATA_LF}/train.jsonl" || [[ ! -s "${DATA_LF}/train.jsonl" ]]; then
    log "export to LLaMA-Factory format"
    mkdir -p "${DATA_LF}"
    python scripts/export/export_to_llamafactory.py \
      --input  "${DATA_SPLITS}" \
      --output "${DATA_LF}" \
      --dataset-name ie_sft_unified
  else
    log "skip export (${DATA_LF}/train.jsonl non-empty)"
  fi
}

run_train() {
  local config="$1"
  [[ -f "$config" ]] || { echo "missing $config" >&2; exit 1; }
  command -v llamafactory-cli >/dev/null || {
    echo "llamafactory-cli not found in PATH" >&2; exit 2;
  }
  log "train with $config"
  exec llamafactory-cli train "$config"
}
