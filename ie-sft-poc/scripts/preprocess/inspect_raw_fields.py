"""Quick diagnostic: print field names of the first N records of a JSONL file.

Usage:
    python scripts/preprocess/inspect_raw_fields.py data/raw/instructie/test_zh.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: inspect_raw_fields.py <jsonl> [limit]", file=sys.stderr)
        sys.exit(2)

    path = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    key_counter: Counter[str] = Counter()
    samples: list[dict] = []

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {i}: bad json: {e}", file=sys.stderr)
                continue
            if isinstance(rec, dict):
                key_counter.update(rec.keys())
                if len(samples) < limit:
                    samples.append(rec)

    print(f"File: {path}  ({sum(key_counter.values())} field occurrences)")
    print("Top-level keys (count):")
    for k, n in key_counter.most_common():
        print(f"  {k}: {n}")

    print(f"\nFirst {len(samples)} record(s):")
    for rec in samples:
        preview = {k: (v if not isinstance(v, str) else v[:120]) for k, v in rec.items()}
        print(json.dumps(preview, ensure_ascii=False, indent=2)[:800])
        print("---")


if __name__ == "__main__":
    main()
