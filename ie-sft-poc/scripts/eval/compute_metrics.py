#!/usr/bin/env python3
"""Compute IE metrics (KV / entity / relation F1) on one or more predictions files.

Input: a JSONL produced by ``scripts/eval/run_predict.py`` where each
line contains ``{id, prompt, prediction, gold}``.

Output: a human-readable summary on stdout plus a ``metrics.json`` file
written next to each predictions file.

Usage:
    python scripts/eval/compute_metrics.py \\
        --predictions outputs/eval/qwen3-lora/test_predictions.jsonl \\
        --task-types kv entity relation \\
        --per-record

    python scripts/eval/compute_metrics.py \\
        --predictions \\
            outputs/eval/qwen3-lora/test_predictions.jsonl \\
            outputs/eval/qwen35-full/test_predictions.jsonl \\
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


def derive_output_path(pred_path: Path) -> Path:
    """Write ``metrics.json`` next to a predictions file."""
    return pred_path.parent / "metrics.json"


def compute_metrics_for_file(
    pred_path: Path,
    task_types: list[str] | None,
    per_record: bool,
) -> tuple[dict, Path]:
    """Load predictions, score them, and write the derived metrics path."""
    rows = list(read_jsonl(pred_path))
    logger.info(f"Loaded {len(rows)} prediction rows from {pred_path}")

    predictions = [r.get("prediction", "") for r in rows]
    golds = [r.get("gold") or {} for r in rows]
    summary = evaluate(predictions, golds, task_types=task_types).to_dict(
        include_per_record=per_record
    )

    out_path = derive_output_path(pred_path)
    write_json(summary, out_path)
    logger.info(f"Wrote metrics summary -> {out_path}")
    return summary, out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="One or more predictions JSONL files",
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

    pred_paths = [Path(raw) for raw in args.predictions]
    derived_outputs: dict[Path, Path] = {}
    for pred_path in pred_paths:
        out_path = derive_output_path(pred_path)
        if out_path in derived_outputs and derived_outputs[out_path] != pred_path:
            logger.error(
                "Multiple predictions files map to the same metrics path: %s <- %s, %s",
                out_path,
                derived_outputs[out_path],
                pred_path,
            )
            return 2
        derived_outputs[out_path] = pred_path

    exit_code = 0
    multi = len(pred_paths) > 1
    for idx, pred_path in enumerate(pred_paths):
        if not pred_path.exists():
            logger.error(f"Predictions file not found: {pred_path}")
            exit_code = 1
            continue

        summary, out_path = compute_metrics_for_file(
            pred_path,
            task_types=args.task_types,
            per_record=args.per_record,
        )

        if multi:
            if idx > 0:
                print()
            print(f"===== {pred_path} =====")
            print(f"metrics_path: {out_path}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
