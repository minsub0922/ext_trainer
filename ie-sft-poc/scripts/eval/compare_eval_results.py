#!/usr/bin/env python3
"""Compare evaluation results across multiple model/variant/mode scenarios.

Scans ``outputs/eval/*/metrics.json`` and produces a summary table sorted
by entity F1 (descending).  Also writes ``outputs/eval/comparison.json``
for programmatic consumption.

Usage:
    python scripts/eval/compare_eval_results.py
    python scripts/eval/compare_eval_results.py --eval-dir outputs/eval
    python scripts/eval/compare_eval_results.py --eval-dir outputs/eval --output outputs/eval/comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_metrics(metrics_path: Path) -> dict | None:
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _extract_row(scenario_name: str, m: dict) -> dict:
    """Flatten a metrics dict into one row for the comparison table."""
    row: dict = {"scenario": scenario_name, "n_records": m.get("n_records", 0)}

    parse_fail = m.get("n_parse_failures", 0)
    n = m.get("n_records", 1) or 1
    row["parse_fail_rate"] = round(parse_fail / n, 4)

    for task in ("kv", "entity", "relation", "relation_typed"):
        t = m.get(task)
        if t is not None:
            row[f"{task}_f1"] = t.get("f1", 0.0)
            row[f"{task}_prec"] = t.get("precision", 0.0)
            row[f"{task}_rec"] = t.get("recall", 0.0)

    return row


def _sort_key(row: dict) -> str:
    """Sort by scenario name in ascending alphabetical order."""
    return row.get("scenario", "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-dir",
        default="outputs/eval",
        help="Root directory containing scenario subdirs with metrics.json",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write comparison JSON here (default: <eval-dir>/comparison.json)",
    )
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"Eval directory not found: {eval_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for metrics_path in sorted(eval_dir.glob("*/metrics.json")):
        scenario = metrics_path.parent.name
        m = _load_metrics(metrics_path)
        if m is None:
            continue
        rows.append(_extract_row(scenario, m))

    if not rows:
        print("No metrics.json files found under", eval_dir, file=sys.stderr)
        return 1

    rows.sort(key=_sort_key)

    # ---- print table --------------------------------------------------------
    # Determine which task columns exist
    has_kv = any("kv_f1" in r for r in rows)
    has_ent = any("entity_f1" in r for r in rows)
    has_rel = any("relation_f1" in r for r in rows)

    header_parts = [f"{'scenario':<45s}", f"{'parse_fail':>10s}"]
    if has_kv:
        header_parts.append(f"{'kv_F1':>8s}")
    if has_ent:
        header_parts += [f"{'ent_P':>8s}", f"{'ent_R':>8s}", f"{'ent_F1':>8s}"]
    if has_rel:
        header_parts += [f"{'rel_P':>8s}", f"{'rel_R':>8s}", f"{'rel_F1':>8s}"]

    header = " | ".join(header_parts)
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  EVALUATION COMPARISON")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in rows:
        parts = [
            f"{r['scenario']:<45s}",
            f"{r['parse_fail_rate']:>9.1%} ",
        ]
        if has_kv:
            parts.append(f"{r.get('kv_f1', -1):>8.4f}")
        if has_ent:
            parts += [
                f"{r.get('entity_prec', -1):>8.4f}",
                f"{r.get('entity_rec', -1):>8.4f}",
                f"{r.get('entity_f1', -1):>8.4f}",
            ]
        if has_rel:
            parts += [
                f"{r.get('relation_prec', -1):>8.4f}",
                f"{r.get('relation_rec', -1):>8.4f}",
                f"{r.get('relation_f1', -1):>8.4f}",
            ]
        print(" | ".join(parts))

    print(sep)
    print(f"  {len(rows)} scenario(s) compared")
    print()

    # ---- write JSON ---------------------------------------------------------
    out_path = Path(args.output) if args.output else eval_dir / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Comparison JSON -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
