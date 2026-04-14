#!/usr/bin/env bash
# Run all 4 OLMo3-style stages back-to-back for Qwen3.5-0.8B.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

export MODEL=qwen3.5
TAG="qwen3.5-0.8b"
DATA_DIR="data/processed/olmo3_style"

if [[ ! -f "${DATA_DIR}/midtrain.jsonl" ]]; then
  python scripts/preprocess/olmo3_style/build_midtrain_mixture.py \
    --input  data/processed/splits/train.jsonl \
    --output ${DATA_DIR}/midtrain.jsonl
fi

bash scripts/train/olmo3_style/run_stage1_midtrain.sh
bash scripts/train/olmo3_style/run_stage2_sft.sh

if [[ ! -f "${DATA_DIR}/preference_pairs.jsonl" ]]; then
  python scripts/preprocess/olmo3_style/build_preference_pairs.py \
    --model-path outputs/olmo3_style/${TAG}/stage2_sft \
    --test data/processed/splits/train.jsonl \
    --output ${DATA_DIR}/preference_pairs.jsonl \
    --k 4 --limit 2000
fi

bash scripts/train/olmo3_style/run_stage3_dpo.sh

if [[ ! -f "${DATA_DIR}/rlvr_prompts.jsonl" ]]; then
  python scripts/preprocess/olmo3_style/build_rlvr_prompts.py \
    --input  data/processed/splits/train.jsonl \
    --output ${DATA_DIR}/rlvr_prompts.jsonl \
    --limit 4000
fi

bash scripts/train/olmo3_style/run_stage4_rlvr.sh

echo "=== OLMo3-style pipeline complete for ${TAG} ==="
