#!/usr/bin/env python3
"""Compute IE metrics (KV / entity / relation F1) on a predictions file.

Input: a JSONL produced by ``scripts/eval/run_predict.py`` where each
line contains ``{id, prompt, prediction, gold}``.

Output: a human-readable summary on stdout plus an optional JSON file
with full details (per-task PRF1, parse failures, and optional
per-record breakdown).

Usage:
    python scripts/eval/compute_metrics.py \\
        --predictions outputs/eval/qwen3-lora/test_predictions.jsonl \\
        --output outputs/eval/qwen3-lora/metrics.json \\
        --task-types kv entity relation \\
        --per-record
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.io import read_jsonl, write_json  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.training.ie_metrics import evaluate  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True, help="Predictions JSONL")
    p.add_argument(
        "--output",
        default=None,
        help="Optional JSON summary output path",
    )
    p.add_argument(
        "--task-types",
        nargs="+",
        choices=["kv", "entity", "relation"],
        default=None,
        help="Tasks to score. Default: union of tasks present in gold.",
    )
    p.add_argument(
        "--per-record",
        action="store_true",
        help="Include per-record breakdown in the JSON output.",
    )
    args = p.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        logger.error(f"Predictions file not found: {pred_path}")
        return 1

    rows = list(read_jsonl(pred_path))
    logger.info(f"Loaded {len(rows)} prediction rows from {pred_path}")

    predictions = [r.get("prediction", "") for r in rows]
    golds = [r.get("gold") or {} for r in rows]

    metrics = evaluate(predictions, golds, task_types=args.task_types)

    summary = metrics.to_dict(include_per_record=args.per_record)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, summary)
        logger.info(f"Wrote metrics summary -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
