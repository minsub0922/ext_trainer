#!/usr/bin/env bash
##############################################################################
# Unified OLMo3-style stage launcher — replaces run_stage{1,2,3,4}_*.sh
#
# Usage:
#   MODEL=qwen3 bash scripts/train/olmo3_style/run_stage.sh --stage 1
#   MODEL=qwen3.5 bash scripts/train/olmo3_style/run_stage.sh --stage 2
#   MODEL=qwen3 bash scripts/train/olmo3_style/run_stage.sh --stage 2-3ep
#   MODEL=qwen3 bash scripts/train/olmo3_style/run_stage.sh --stage 3-3ep
#   bash scripts/train/olmo3_style/run_stage.sh --stage 3 --model qwen3
#   bash scripts/train/olmo3_style/run_stage.sh --stage 4 --model qwen3
#
# Environment:
#   MODEL        qwen3 or qwen3.5 (default: qwen3)
#   NPROC        number of GPUs (default: auto)
#   MASTER_PORT  rendezvous port (default: auto-detected free port)
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../_common.sh"
cd "$PROJECT_ROOT"

STAGE=""
MODEL="${MODEL:-qwen3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$STAGE" ]] && { echo "ERROR: --stage {1|2|2-3ep|3|3-3ep|4|4-3ep} required" >&2; exit 2; }

# ---- resolve config ---------------------------------------------------------
case "$MODEL" in
  qwen3)   MODEL_DIR="qwen3" ;;
  qwen3.5) MODEL_DIR="qwen3_5" ;;
  *) echo "ERROR: unknown MODEL=$MODEL (use qwen3 or qwen3.5)" >&2; exit 2 ;;
esac

case "$STAGE" in
  1) CONFIG="configs/olmo3_style/${MODEL_DIR}/stage1_midtrain.yaml" ; STAGE_NAME="mid-training" ;;
  2) CONFIG="configs/olmo3_style/${MODEL_DIR}/stage2_sft.yaml"      ; STAGE_NAME="SFT" ;;
  2-3ep) CONFIG="configs/olmo3_style/${MODEL_DIR}/stage2_sft_3ep.yaml" ; STAGE_NAME="SFT (3 epochs)" ;;
  3) CONFIG="configs/olmo3_style/${MODEL_DIR}/stage3_dpo.yaml"      ; STAGE_NAME="DPO" ;;
  3-3ep) CONFIG="configs/olmo3_style/${MODEL_DIR}/stage3_dpo_3ep.yaml" ; STAGE_NAME="DPO (stage2 3 epochs)" ;;
  4) CONFIG="configs/olmo3_style/${MODEL_DIR}/stage4_rlvr.yaml"     ; STAGE_NAME="RLVR" ;;
  4-3ep) CONFIG="configs/olmo3_style/${MODEL_DIR}/stage4_rlvr_3ep.yaml" ; STAGE_NAME="RLVR (stage2 3 epochs)" ;;
  *) echo "ERROR: invalid stage $STAGE (use 1, 2, 2-3ep, 3, 3-3ep, 4, or 4-3ep)" >&2; exit 2 ;;
esac

[[ ! -f "$CONFIG" ]] && { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }

echo "=== Stage $STAGE ${STAGE_NAME} ($MODEL) ==="
echo "  CONFIG: $CONFIG"

# ---- stage 4 uses custom RLVR trainer (not llamafactory-cli) ----------------
if [[ "$STAGE" == "4" || "$STAGE" == "4-3ep" ]]; then
  setup_multigpu
  echo "  NPROC: $NPROC"

  if [[ "${NPROC:-1}" -gt 1 ]]; then
    MASTER_PORT="${MASTER_PORT:-$(find_free_port 29500)}"
    echo "  MASTER_PORT: $MASTER_PORT"
    exec torchrun \
      --nproc_per_node="$NPROC" \
      --master_port="$MASTER_PORT" \
      -m src.training.olmo3_style.rlvr_trainer "$CONFIG"
  else
    exec python -m src.training.olmo3_style.rlvr_trainer "$CONFIG"
  fi
fi

# ---- stages 1-3 use llamafactory-cli ----------------------------------------
command -v llamafactory-cli >/dev/null || { echo "ERROR: llamafactory-cli not found" >&2; exit 2; }

# Flash attention check
CONFIG="$(patch_flash_attn_in_yaml "$CONFIG")"

# GPU setup
setup_multigpu
echo "  NPROC: $NPROC"
setup_distributed_env

exec llamafactory-cli train "$CONFIG"
