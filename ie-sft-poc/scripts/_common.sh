#!/usr/bin/env bash
# Shared helpers for all shell scripts in this project.
#
# Source this file from any launcher script:
#   source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
#
# Provides:
#   PROJECT_ROOT        absolute project root
#   find_free_port      find an available TCP port
#   detect_flash_attn   check if flash-attn works and patch YAML if needed
#   setup_multigpu      detect GPU count and configure environment

set -euo pipefail

# ---- project root ----------------------------------------------------------
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Works whether sourced from scripts/ or scripts/train/ or scripts/train/olmo3_style/
PROJECT_ROOT="$(cd "${_COMMON_DIR}" && while [[ ! -f "src/common/io.py" ]] && [[ "$(pwd)" != "/" ]]; do cd ..; done; pwd)"
export IESFT_PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- find free port --------------------------------------------------------
find_free_port() {
  # Find an available TCP port. Pure-Python for reliability across environments.
  # Bind-tests each port so there's no TOCTOU gap with ss/netstat.
  python3 -c "
import socket, sys
base = int(sys.argv[1]) if len(sys.argv) > 1 else 29500
for port in range(base, base + 200):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', port))
        s.close()
        print(port)
        sys.exit(0)
    except OSError:
        try: s.close()
        except: pass
# fallback: let OS pick
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
" "${1:-29500}"
}

# ---- flash attention detection ---------------------------------------------
detect_flash_attn() {
  # Returns "fa2" if flash-attn is actually usable, "auto" otherwise.
  # Uses the SAME check that transformers runs inside from_pretrained(),
  # so there's zero chance of a mismatch at model-load time.
  python3 -c "
import sys
try:
    from transformers.utils import is_flash_attn_2_available
    if is_flash_attn_2_available():
        print('fa2')
    else:
        print('auto')
except Exception:
    print('auto')
" 2>/dev/null || echo "auto"
}

patch_flash_attn_in_yaml() {
  # If flash_attn: fa2 is in the config but FA2 is not available,
  # create a temporary patched copy. Returns the (possibly new) config path.
  local config="$1"
  local fa_status
  fa_status="$(detect_flash_attn)"

  if [[ "$fa_status" == "fa2" ]]; then
    echo "$config"
    return 0
  fi

  # Check if config even has flash_attn setting that needs patching
  if ! grep -qE '^\s*flash_attn\s*:\s*fa2' "$config" 2>/dev/null; then
    echo "$config"
    return 0
  fi

  echo "[flash_attn] FA2 not available, patching config to use 'auto'" >&2
  local tmp_config
  tmp_config="$(mktemp -t ie_sft_config_XXXXXX.yaml)"
  sed 's/^\(\s*flash_attn\s*:\s*\)fa2/\1auto/' "$config" > "$tmp_config"
  echo "$tmp_config"
}

# ---- GPU setup -------------------------------------------------------------
setup_multigpu() {
  # Sets NPROC if not already set. Detects GPU count from nvidia-smi.
  if [[ -z "${NPROC:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
      NPROC=1
    fi
  fi
  export NPROC
}

# ---- multi-GPU launch helper -----------------------------------------------
setup_distributed_env() {
  # Export all env vars that llamafactory-cli / torchrun need for multi-GPU.
  # Call this AFTER setup_multigpu.
  if [[ "${NPROC:-1}" -gt 1 ]]; then
    MASTER_PORT="${MASTER_PORT:-$(find_free_port 29500)}"
    export FORCE_TORCHRUN=1
    export NNODES=1
    export NPROC_PER_NODE="$NPROC"
    export MASTER_PORT
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    echo "[distributed] NPROC=$NPROC MASTER_PORT=$MASTER_PORT MASTER_ADDR=$MASTER_ADDR"
  fi
}
