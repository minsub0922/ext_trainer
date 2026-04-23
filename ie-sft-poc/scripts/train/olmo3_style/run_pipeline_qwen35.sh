#!/usr/bin/env bash
# OLMo3-style pipeline for Qwen3.5-0.8B — delegates to unified run_pipeline.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_pipeline.sh" --model qwen3.5 "$@"
