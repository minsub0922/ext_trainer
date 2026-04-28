#!/usr/bin/env bash
# Shared helpers for OLMo3-style one-shot launchers.
#
# Source this file; it defines:
#   - PROJECT_ROOT (absolute)
#   - DATA_RAW / DATA_CANONICAL / DATA_UNIFIED / DATA_SPLITS / DATA_LF
#   - run_data_pipeline  (idempotent: download → normalize → unify → split → export)
#
# Each phase is skipped if its expected output already exists. Training is
# invoked by the individual olmo3_style stage scripts, not by this file —
# this only handles the pre-training data prep shared by all model-specific
# oneshots.

set -euo pipefail

_ONESHOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_ONESHOT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"
export IESFT_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_RAW="data/raw/instructie"
DATA_CANONICAL="data/processed/canonical"
DATA_CANONICAL_PARTS="${DATA_CANONICAL}/instructie_parts"
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
  if [[ ! -s "${DATA_CANONICAL}/instructie.jsonl" ]]; then
    log "normalize InstructIE → canonical"
    mkdir -p "${DATA_CANONICAL}" "${DATA_CANONICAL_PARTS}"
    python scripts/preprocess/normalize_instructie.py \
      --input-dir  "${DATA_RAW}" \
      --output-dir "${DATA_CANONICAL_PARTS}" \
      --overwrite
    find "${DATA_CANONICAL_PARTS}" -maxdepth 1 -name '*.jsonl' -type f -print0 \
      | sort -z \
      | xargs -0 cat \
      > "${DATA_CANONICAL}/instructie.jsonl"
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

  # 5. export to LLaMA-Factory (stage 2 SFT reads this)
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
