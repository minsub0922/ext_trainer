#!/usr/bin/env bash
##############################################################################
# Unified SFT training launcher — replaces all model-specific variants.
#
# Supports: Qwen3 LoRA/Full, Qwen3.5 LoRA/Full, OLMo3 POC, custom configs.
# Handles: single-GPU, multi-GPU (via llamafactory-cli), flash-attn fallback,
#          dynamic port selection.
#
# Usage:
#   # Qwen3 LoRA (default)
#   bash scripts/train/run_sft.sh
#
#   # Qwen3.5 Full SFT with DeepSpeed
#   bash scripts/train/run_sft.sh --model qwen3.5 --variant full
#
#   # OLMo3 POC
#   bash scripts/train/run_sft.sh --model olmo3
#
#   # Custom config
#   bash scripts/train/run_sft.sh --config configs/sft/my_custom.yaml
#
#   # Multi-GPU explicit
#   NPROC=4 bash scripts/train/run_sft.sh --model qwen3 --variant full
#
# Environment variables:
#   CONFIG        override config path directly
#   NPROC         number of GPUs (default: auto-detected)
#   MASTER_PORT   rendezvous port (default: auto-detected free port)
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
cd "$PROJECT_ROOT"

# ---- defaults ---------------------------------------------------------------
MODEL="${MODEL:-qwen3}"
VARIANT="${VARIANT:-lora}"
CONFIG="${CONFIG:-}"
DRY_RUN=false
EXTRA_ARGS=()

# ---- parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)    MODEL="$2";   shift 2 ;;
    --variant)  VARIANT="$2"; shift 2 ;;
    --config)   CONFIG="$2";  shift 2 ;;
    --dry-run)  DRY_RUN=true; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ---- resolve config ---------------------------------------------------------
if [[ -z "$CONFIG" ]]; then
  case "${MODEL}/${VARIANT}" in
    qwen3/lora)     CONFIG="configs/sft/qwen3_lora_sft.yaml" ;;
    qwen3/full)     CONFIG="configs/sft/qwen3_full_sft_ds.yaml" ;;
    qwen3-4b/lora)  CONFIG="configs/sft/qwen3_4b_lora_sft.yaml" ;;
    qwen3-4b/full)  CONFIG="configs/sft/qwen3_4b_full_sft_ds.yaml" ;;
    qwen3.5/lora)   CONFIG="configs/sft/qwen3_5_lora_sft.yaml" ;;
    qwen3.5/full)   CONFIG="configs/sft/qwen3_5_full_sft_ds.yaml" ;;
    olmo3/lora|olmo3/*)  CONFIG="configs/sft/olmo3_poc_sft.yaml" ;;
    *)
      echo "ERROR: unknown model/variant: ${MODEL}/${VARIANT}" >&2
      echo "  Supported: qwen3/lora, qwen3/full, qwen3.5/lora, qwen3.5/full, olmo3/lora" >&2
      echo "  Or use --config to specify a custom YAML." >&2
      exit 2
      ;;
  esac
fi

[[ ! -f "$CONFIG" ]] && { echo "ERROR: config not found: $CONFIG" >&2; exit 1; }
command -v llamafactory-cli >/dev/null || { echo "ERROR: llamafactory-cli not found" >&2; exit 2; }

# ---- flash attention check --------------------------------------------------
CONFIG="$(patch_flash_attn_in_yaml "$CONFIG")"

# ---- GPU setup --------------------------------------------------------------
setup_multigpu

echo "==========================================="
echo "  IE SFT Training"
echo "-------------------------------------------"
echo "  MODEL/VARIANT:       ${MODEL}/${VARIANT}"
echo "  CONFIG:              $CONFIG"
echo "  NPROC (GPUs):        $NPROC"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all>}"
if grep -qE '^\s*deepspeed\s*:' "$CONFIG" 2>/dev/null; then
  DS=$(grep -E '^\s*deepspeed\s*:' "$CONFIG" | awk '{print $2}')
  echo "  DeepSpeed config:    $DS"
fi
FA=$(grep -E '^\s*flash_attn\s*:' "$CONFIG" 2>/dev/null | awk '{print $2}' || echo "n/a")
echo "  Flash Attention:     $FA"
echo "==========================================="

if [[ "$DRY_RUN" == true ]]; then
  echo "DRY RUN — would execute:"
  echo "  FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=$NPROC llamafactory-cli train $CONFIG"
  exit 0
fi

# ---- launch -----------------------------------------------------------------
# llamafactory-cli supports multi-GPU natively via environment variables.
# No need for torchrun wrapper.
setup_distributed_env

exec llamafactory-cli train "$CONFIG"
