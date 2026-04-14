#!/usr/bin/env python
"""Build the RLVR prompt file consumed by src/training/olmo3_style/rlvr_trainer.

Each line is a JSON object with:
    {
      "prompt":    <formatted prompt string>,
      "gold":      <canonical gold record>,
      "task_type": "kv" | "entity" | "relation",
    }

Usage:
    python scripts/preprocess/olmo3_style/build_rlvr_prompts.py \
        --input data/processed/splits/train.jsonl \
        --output data/processed/olmo3_style/rlvr_prompts.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_prompt(rec: dict) -> str:
    schema = rec.get("schema") or rec.get("task_schema") or {}
    doc = rec.get("text") or rec.get("input") or ""
    return (f"Task schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Document:\n{doc}\n\nExtract as JSON:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as src, \
         out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            if args.limit and n >= args.limit:
                break
            rec = json.loads(line)
            row = {
                "prompt":    build_prompt(rec),
                "gold":      rec.get("records") or rec.get("output") or {},
                "task_type": rec.get("task_type") or rec.get("task") or "kv",
            }
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"wrote {n} RLVR prompts → {out_path}")


if __name__ == "__main__":
    main()
