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
sys.path.insert(0, str(ROOT))


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


def _normalize_answer(answer: dict | None) -> dict:
    answer = answer or {}
    return {
        "kv": answer.get("kv") or {},
        "entity": answer.get("entity") or [],
        "relation": answer.get("relation") or [],
    }


def _infer_task_types(answer: dict) -> list[str]:
    tasks = []
    if answer.get("kv"):
        tasks.append("kv")
    if answer.get("entity"):
        tasks.append("entity")
    if answer.get("relation"):
        tasks.append("relation")
    return tasks


def _canonical_gold(rec: dict) -> dict:
    """Return a canonical-shaped gold record for ie_metrics.evaluate."""
    if rec.get("answer"):
        gold = dict(rec)
        gold["answer"] = _normalize_answer(gold.get("answer"))
        if not gold.get("task_types"):
            gold["task_types"] = _infer_task_types(gold["answer"])
        return gold

    # Legacy fallback for older generated/intermediate files.
    answer = _normalize_answer(
        rec.get("records") or rec.get("output") or rec.get("target") or {}
    )
    gold = dict(rec)
    gold["answer"] = answer
    gold["task_types"] = rec.get("task_types") or _infer_task_types(answer)
    return gold


def _primary_task_type(rec: dict) -> str:
    """Pick one task to score for pair ranking.

    Relation extraction is the most specific signal when present; entity is
    next; kv is used for KV-only records.
    """
    answer = _normalize_answer(rec.get("answer"))
    if answer["relation"]:
        return "relation"
    if answer["entity"]:
        return "entity"
    if answer["kv"]:
        return "kv"

    task_types = rec.get("task_types") or []
    for task in ("relation", "entity", "kv"):
        if task in task_types:
            return task
    return "kv"


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
    ap.add_argument("--model-path")
    ap.add_argument("--test",
                    help="canonical train/dev JSONL to draw prompts from")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--dataset-name",
        default="ie_pref_pairs",
        help="dataset_info.json key to register (default: ie_pref_pairs)",
    )
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-margin", type=float, default=0.15)
    ap.add_argument(
        "--no-gold-fallback",
        action="store_true",
        help=(
            "disable fallback pairs that use the canonical gold answer as "
            "chosen when model samples do not produce a margin"
        ),
    )
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument(
        "--register-only",
        action="store_true",
        help="only refresh dataset_info.json for an existing pairs file",
    )
    args = ap.parse_args()

    out = Path(args.output)
    info_path = out.parent / "dataset_info.json"

    def register_dataset() -> None:
        info = (
            json.loads(info_path.read_text(encoding="utf-8"))
            if info_path.exists()
            else {}
        )
        info[args.dataset_name] = {
            "file_name": out.name,
            "ranking": True,
            "columns": {"prompt": "instruction", "query": "input",
                        "response": "output"},
        }
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"registered `{args.dataset_name}` in {info_path}")

    if args.register_only:
        if not out.exists() or out.stat().st_size == 0:
            raise FileNotFoundError(
                f"non-empty preference pairs file not found: {out}"
            )
        register_dataset()
        return

    if not args.model_path or not args.test:
        ap.error("--model-path and --test are required unless --register-only is used")

    from src.training.olmo3_style.preference_builder import (  # noqa: E402
        PairBuilderConfig,
        build_preference_pairs,
        write_pairs_jsonl,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Detect best available attention implementation via transformers' own check
    attn_impl = "sdpa"
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            attn_impl = "flash_attention_2"
    except Exception:
        pass

    # Load with fallback: if chosen attn_impl still fails, retry with sdpa
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype_map[args.dtype],
            attn_implementation=attn_impl, device_map="auto",
            trust_remote_code=True,
        )
    except (ImportError, ValueError):
        if attn_impl == "sdpa":
            raise
        print(f"[WARN] attn_implementation={attn_impl} failed; retrying with sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype_map[args.dtype],
            attn_implementation="sdpa", device_map="auto",
            trust_remote_code=True,
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
            gold = _canonical_gold(rec)
            raw_task = rec.get("task_type") or rec.get("task")
            task_type = (
                raw_task
                if raw_task in {"kv", "entity", "relation"}
                else _primary_task_type(gold)
            )
            yield {
                "prompt": prompt,
                "instruction": prompt,
                "input": "",
                "gold": gold,
                "task_type": task_type,
                "samples": samples,
            }
            n += 1

    cfg = PairBuilderConfig(
        min_margin=args.min_margin,
        allow_gold_fallback=not args.no_gold_fallback,
    )
    pairs = build_preference_pairs(gen_group(), cfg)
    if not pairs:
        if out.exists():
            out.unlink()
        raise RuntimeError(
            "No preference pairs were generated. Check that the input split is "
            "canonical with non-empty answer/task_types, retry with a lower "
            "--min-margin, or omit --no-gold-fallback after inspecting model "
            "samples."
        )
    out = write_pairs_jsonl(pairs, args.output)

    # Register in dataset_info.json
    register_dataset()
    fallback_pairs = sum(
        1 for pair in pairs
        if (pair.get("metadata") or {}).get("fallback") == "gold_answer"
    )
    if fallback_pairs:
        print(f"wrote {len(pairs)} pairs → {out} ({fallback_pairs} gold-fallback)")
    else:
        print(f"wrote {len(pairs)} pairs → {out}")


if __name__ == "__main__":
    main()
