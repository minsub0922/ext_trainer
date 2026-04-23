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

# ---- find free port --------------------------------------------------------
find_free_port() {
  # Find an available port, starting from a base and incrementing.
  local base="${1:-29500}"
  local max_tries="${2:-100}"
  local port="$base"
  for (( i=0; i<max_tries; i++ )); do
    if ! ss -tln 2>/dev/null | grep -q ":${port} " && \
       ! netstat -tln 2>/dev/null | grep -q ":${port} "; then
      # Double-check with python as a fallback
      if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(('', ${port}))
    s.close()
    sys.exit(0)
except OSError:
    s.close()
    sys.exit(1)
" 2>/dev/null; then
        echo "$port"
        return 0
      fi
    fi
    port=$((port + 1))
  done
  # If all else fails, let the OS pick
  python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
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

  # Check if config even has flash_attn setting
  if ! grep -qE '^\s*flash_attn\s*:' "$config" 2>/dev/null; then
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
