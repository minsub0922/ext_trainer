#!/usr/bin/env bash
# Run all 4 OLMo3-style stages back-to-back for Qwen3-0.6B.
#
# Stops early on failure. Each stage's checkpoint feeds the next.
# Data prep steps (midtrain mixture, preference pairs, RLVR prompts)
# are invoked here so a cold box just needs the base splits to exist.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

export MODEL=qwen3
TAG="qwen3-0.6b"
DATA_DIR="data/processed/olmo3_style"

# ----- data prep -------------------------------------------------------------
if [[ ! -f "${DATA_DIR}/midtrain.jsonl" ]]; then
  echo "[prep] building midtrain mixture"
  python scripts/preprocess/olmo3_style/build_midtrain_mixture.py \
    --input  data/processed/splits/train.jsonl \
    --output ${DATA_DIR}/midtrain.jsonl
fi

# ----- stage 1 ---------------------------------------------------------------
bash scripts/train/olmo3_style/run_stage1_midtrain.sh

# ----- stage 2 ---------------------------------------------------------------
bash scripts/train/olmo3_style/run_stage2_sft.sh

# ----- prep preference pairs (needs stage-2 ckpt) ----------------------------
if [[ ! -f "${DATA_DIR}/preference_pairs.jsonl" ]]; then
  echo "[prep] building preference pairs"
  python scripts/preprocess/olmo3_style/build_preference_pairs.py \
    --model-path outputs/olmo3_style/${TAG}/stage2_sft \
    --test data/processed/splits/train.jsonl \
    --output ${DATA_DIR}/preference_pairs.jsonl \
    --k 4 --limit 2000
fi

# ----- stage 3 ---------------------------------------------------------------
bash scripts/train/olmo3_style/run_stage3_dpo.sh

# ----- prep RLVR prompts -----------------------------------------------------
if [[ ! -f "${DATA_DIR}/rlvr_prompts.jsonl" ]]; then
  echo "[prep] building RLVR prompts"
  python scripts/preprocess/olmo3_style/build_rlvr_prompts.py \
    --input  data/processed/splits/train.jsonl \
    --output ${DATA_DIR}/rlvr_prompts.jsonl \
    --limit 4000
fi

# ----- stage 4 ---------------------------------------------------------------
bash scripts/train/olmo3_style/run_stage4_rlvr.sh

echo "=== OLMo3-style pipeline complete for ${TAG} ==="
