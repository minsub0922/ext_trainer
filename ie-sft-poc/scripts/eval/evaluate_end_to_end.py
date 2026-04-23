#!/usr/bin/env python3
"""End-to-end evaluation orchestrator: predict + score in one call.

Runs ``run_predict.py`` then ``compute_metrics.py`` on the result, so
you don't need to wire them together manually. Good for CI / nightly
eval / sweeps.

Usage:
    python scripts/eval/evaluate_end_to_end.py \\
        --scenario qwen3_lora_unified \\
        --test data/processed/splits/test.jsonl \\
        --model-path Qwen/Qwen3-0.6B \\
        --adapter-path outputs/qwen3-0.6b-ie-lora \\
        --output-dir outputs/eval/qwen3-lora-unified

The ``--scenario`` tag only affects the default output subdirectory so
multiple runs don't overwrite each other.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", required=True, help="Name tag for this eval run")
    p.add_argument("--test", required=True, help="Canonical JSONL test file")
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--prompt-mode",
        default="unified",
        choices=["kv", "entity", "relation", "unified"],
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=200, help="Max samples (0=all, default=200)")
    p.add_argument(
        "--task-types",
        nargs="+",
        choices=["kv", "entity", "relation"],
        default=None,
    )
    p.add_argument("--per-record", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "test_predictions.jsonl"
    repo_root = Path(__file__).resolve().parents[2]

    predict_cmd = [
        sys.executable,
        str(repo_root / "scripts/eval/run_predict.py"),
        "--input", args.test,
        "--output", str(pred_path),
        "--model-path", args.model_path,
        "--prompt-mode", args.prompt_mode,
        "--max-new-tokens", str(args.max_new_tokens),
        "--batch-size", str(args.batch_size),
        "--dtype", args.dtype,
        "--temperature", str(args.temperature),
        "--limit", str(args.limit),
    ]
    if args.adapter_path:
        predict_cmd += ["--adapter-path", args.adapter_path]
    if args.dry_run:
        predict_cmd.append("--dry-run")

    logger.info("[%s] Running: %s", args.scenario, " ".join(predict_cmd))
    rc = subprocess.call(predict_cmd)
    if rc != 0:
        logger.error("Prediction step failed (rc=%d)", rc)
        return rc
    if args.dry_run:
        return 0

    metrics_cmd = [
        sys.executable,
        str(repo_root / "scripts/eval/compute_metrics.py"),
        "--predictions", str(pred_path),
    ]
    if args.task_types:
        metrics_cmd += ["--task-types", *args.task_types]
    if args.per_record:
        metrics_cmd.append("--per-record")

    logger.info("[%s] Running: %s", args.scenario, " ".join(metrics_cmd))
    rc = subprocess.call(metrics_cmd)
    if rc != 0:
        logger.error("Metrics step failed (rc=%d)", rc)
        return rc

    logger.info(
        "[%s] DONE. predictions=%s metrics=%s",
        args.scenario,
        pred_path,
        out_dir / "metrics.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
