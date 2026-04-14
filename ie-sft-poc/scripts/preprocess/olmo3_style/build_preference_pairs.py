#!/usr/bin/env python
"""Generate DPO preference pairs from the stage-2 SFT model.

For each prompt in the train split we:
    1. Sample K completions (high temperature).
    2. Score each with src/training/ie_metrics.
    3. Emit (chosen, rejected) pairs with margin >= min_margin.

Outputs:
    data/processed/olmo3_style/preference_pairs.jsonl
    data/processed/olmo3_style/dataset_info.json  (adds `ie_pref_pairs`)

Usage:
    python scripts/preprocess/olmo3_style/build_preference_pairs.py \
        --model-path outputs/olmo3_style/qwen3-0.6b/stage2_sft \
        --test data/processed/splits/train.jsonl \
        --output data/processed/olmo3_style/preference_pairs.jsonl \
        --k 4 --temperature 1.0 --limit 2000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.olmo3_style.preference_builder import (  # noqa: E402
    PairBuilderConfig, build_preference_pairs, write_pairs_jsonl,
)


def _iter_canonical(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_prompt(rec: dict) -> str:
    schema = rec.get("schema") or rec.get("task_schema") or {}
    doc = rec.get("text") or rec.get("input") or ""
    return (f"Task schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Document:\n{doc}\n\nExtract as JSON:")


def _sample_k(model, tok, prompt: str, k: int, temperature: float,
              max_new_tokens: int, device) -> list[str]:
    import torch
    enc = tok(prompt, return_tensors="pt").to(device)
    batch = enc["input_ids"].expand(k, -1)
    attn = enc["attention_mask"].expand(k, -1)
    with torch.no_grad():
        gen = model.generate(
            input_ids=batch,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen_only = gen[:, batch.shape[1]:]
    return tok.batch_decode(gen_only, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--test", required=True,
                    help="canonical train/dev JSONL to draw prompts from")
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-margin", type=float, default=0.15)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype_map[args.dtype],
        attn_implementation="flash_attention_2", trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    def gen_group():
        n = 0
        for rec in _iter_canonical(Path(args.test)):
            if args.limit and n >= args.limit:
                break
            prompt = _build_prompt(rec)
            samples = _sample_k(model, tok, prompt, args.k, args.temperature,
                                args.max_new_tokens, device)
            yield {
                "prompt": prompt,
                "instruction": prompt,
                "input": "",
                "gold": rec.get("records") or rec.get("output") or {},
                "task_type": rec.get("task_type") or rec.get("task") or "kv",
                "samples": samples,
            }
            n += 1

    cfg = PairBuilderConfig(min_margin=args.min_margin)
    pairs = build_preference_pairs(gen_group(), cfg)
    out = write_pairs_jsonl(pairs, args.output)

    # Register in dataset_info.json
    info_path = out.parent / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info["ie_pref_pairs"] = {
        "file_name": out.name,
        "ranking": True,
        "columns": {"prompt": "instruction", "query": "input",
                    "response": "output"},
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"wrote {len(pairs)} pairs → {out}")


if __name__ == "__main__":
    main()
