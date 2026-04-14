#!/usr/bin/env python
"""Generate (or merge into) LLaMA-Factory dataset_info.json.

LLaMA-Factory requires every dataset referenced by `dataset:` in a YAML
to be declared in `<dataset_dir>/dataset_info.json` with a FLAT layout:

    {
      "ie_sft_unified":      {"file_name": "train.jsonl", "columns": {...}},
      "ie_sft_unified_dev":  {"file_name": "dev.jsonl",   "columns": {...}},
      "ie_sft_unified_test": {"file_name": "test.jsonl",  "columns": {...}}
    }

If the JSON already exists, new keys are merged in (existing keys are
overwritten by default; use --no-overwrite to keep them).

Usage:
    # After export_to_llamafactory.py has written train/dev/test jsonl:
    python scripts/export/register_llamafactory_datasets.py \
        --dataset-dir data/processed/llamafactory \
        --name ie_sft_unified

    # Multiple datasets at once
    python scripts/export/register_llamafactory_datasets.py \
        --dataset-dir data/processed/llamafactory \
        --name ie_sft_unified --name ie_sft_kv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_COLUMNS = {
    "prompt":   "instruction",
    "query":    "input",
    "response": "output",
}

DEFAULT_FILES = [
    ("train.jsonl", ""),      # empty suffix = bare dataset name
    ("dev.jsonl",   "_dev"),
    ("test.jsonl",  "_test"),
]


def build_entries(dataset_dir: Path, name: str,
                   files: list[tuple[str, str]] = DEFAULT_FILES
                   ) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for fname, suffix in files:
        path = dataset_dir / fname
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        key = f"{name}{suffix}"
        entries[key] = {
            "file_name": fname,
            "columns": dict(DEFAULT_COLUMNS),
        }
        print(f"  [ok]   {key} -> {fname}")
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True,
                    help="directory containing train/dev/test.jsonl and "
                         "dataset_info.json (will be created)")
    ap.add_argument("--name", action="append", required=True,
                    help="dataset name (can repeat to register multiple)")
    ap.add_argument("--no-overwrite", action="store_true",
                    help="don't overwrite existing keys in dataset_info.json")
    args = ap.parse_args()

    ddir = Path(args.dataset_dir)
    ddir.mkdir(parents=True, exist_ok=True)
    info_path = ddir / "dataset_info.json"

    existing: dict = {}
    if info_path.exists():
        existing = json.loads(info_path.read_text(encoding="utf-8"))
        print(f"loaded existing registry with {len(existing)} keys")

    added = 0
    for name in args.name:
        print(f"registering `{name}` ...")
        entries = build_entries(ddir, name)
        for k, v in entries.items():
            if k in existing and args.no_overwrite:
                print(f"  [keep] {k} (already present)")
                continue
            existing[k] = v
            added += 1

    info_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {info_path}  (+{added} new/updated keys, {len(existing)} total)")


if __name__ == "__main__":
    main()
