#!/usr/bin/env python
"""Build a packed pretraining-style mixture for OLMo3 stage-1 mid-training.

Takes canonical IE records and renders them as flat text lines suitable
for `stage: pt` LLaMA-Factory training. Each canonical record becomes
one plaintext example alternating schema + rendered record:

    <SCHEMA>
    fields: ...
    </SCHEMA>

    INPUT: <doc text>
    OUTPUT:
    {"records": [...]}

Writes:
    data/processed/olmo3_style/midtrain.jsonl
    data/processed/olmo3_style/dataset_info.json  (adds `ie_midtrain`)

Usage:
    python scripts/preprocess/olmo3_style/build_midtrain_mixture.py \
        --input data/processed/splits/train.jsonl \
        --output data/processed/olmo3_style/midtrain.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_record(rec: dict) -> str:
    """Flatten a canonical record into a single text block."""
    schema = rec.get("schema") or rec.get("task_schema") or {}
    doc = rec.get("text") or rec.get("input") or ""
    target = rec.get("records") or rec.get("output") or rec.get("target") or {}

    parts = []
    parts.append("<SCHEMA>")
    parts.append(json.dumps(schema, ensure_ascii=False))
    parts.append("</SCHEMA>")
    parts.append("")
    parts.append("INPUT:")
    parts.append(doc)
    parts.append("")
    parts.append("OUTPUT:")
    parts.append(json.dumps(target, ensure_ascii=False))
    return "\n".join(parts)


def update_dataset_info(info_path: Path, mixture_file: str) -> None:
    info = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    info["ie_midtrain"] = {
        "file_name": mixture_file,
        "columns": {"prompt": "text"},
        "formatting": "plain",
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                         encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="canonical JSONL")
    ap.add_argument("--output", required=True, help="midtrain JSONL out path")
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
            rec = json.loads(line)
            text = render_record(rec)
            dst.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1

    info_path = out_path.parent / "dataset_info.json"
    update_dataset_info(info_path, out_path.name)
    print(f"wrote {n} midtrain rows → {out_path}")
    print(f"registered `ie_midtrain` in {info_path}")


if __name__ == "__main__":
    main()
