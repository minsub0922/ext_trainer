#!/usr/bin/env python3
"""Generate predictions from a trained (base + optional LoRA) checkpoint
on a canonical IE test set.

Writes a JSONL file with fields: ``id``, ``prompt``, ``prediction``,
``gold``. ``gold`` is the original canonical record so a separate metrics
script can score without round-tripping.

Usage:
    python scripts/eval/run_predict.py \\
        --input data/processed/splits/test.jsonl \\
        --output outputs/eval/qwen3-lora/test_predictions.jsonl \\
        --model-path Qwen/Qwen3-0.6B \\
        --adapter-path outputs/qwen3-0.6b-ie-lora \\
        --prompt-mode unified \\
        --max-new-tokens 512

The prompt builders from ``src.datasets.gollie_reference.schema_patterns``
are reused so train/eval prompts stay in sync.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the project root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.io import read_jsonl, write_jsonl  # noqa: E402
from src.common.logging_utils import get_logger  # noqa: E402
from src.common.schema import CanonicalIERecord, SchemaDefinition  # noqa: E402
from src.datasets.gollie_reference.schema_patterns import (  # noqa: E402
    build_entity_extraction_prompt,
    build_kv_extraction_prompt,
    build_relation_extraction_prompt,
    build_unified_extraction_prompt,
)

logger = get_logger(__name__)


def build_prompt(record: dict, mode: str) -> str:
    """Build the input prompt for a single canonical record."""
    text = record.get("text", "")
    schema = record.get("schema", {}) or {}
    sd = SchemaDefinition(
        kv=schema.get("kv", []) or [],
        entity=schema.get("entity", []) or [],
        relation=schema.get("relation", []) or [],
    )
    if mode == "kv":
        return build_kv_extraction_prompt(text, sd.kv)
    if mode == "entity":
        return build_entity_extraction_prompt(text, sd.entity)
    if mode == "relation":
        return build_relation_extraction_prompt(text, sd.relation, sd.entity)
    return build_unified_extraction_prompt(text, sd)


def _detect_attn_implementation() -> str:
    """Detect the best available attention implementation.

    Uses transformers' own ``is_flash_attn_2_available()`` so the check
    is identical to the one that happens inside ``from_pretrained``.
    Falls back to ``"sdpa"`` (PyTorch ≥2.0 scaled-dot-product) which
    works everywhere without extra packages.
    """
    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            logger.info("Flash Attention 2 available (transformers check passed)")
            return "flash_attention_2"
        else:
            logger.info("Flash Attention 2 NOT available (transformers check); using sdpa")
    except Exception as exc:
        logger.info("Could not check flash-attn availability (%s); using sdpa", exc)
    return "sdpa"


def _load_model_with_fallback(model_path: str, torch_dtype, attn_impl: str):
    """Try loading with *attn_impl*; on failure retry with ``"sdpa"``."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except (ImportError, ValueError) as exc:
        if attn_impl == "sdpa":
            raise  # already the safest option; nothing to fall back to
        logger.warning(
            "Loading with attn_implementation=%s failed (%s); retrying with sdpa",
            attn_impl,
            exc,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )


def load_model(model_path: str, adapter_path: str | None, dtype: str):
    """Load the base model (and optional LoRA adapter) via transformers/peft."""
    import torch
    from transformers import AutoTokenizer

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    attn_impl = _detect_attn_implementation()
    model = _load_model_with_fallback(model_path, torch_dtype, attn_impl)

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"Loaded LoRA adapter from {adapter_path}")

    model.eval()
    return model, tok


def generate_batch(
    model,
    tok,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Run generation on a list of prompts and return decoded completions."""
    import torch

    device = next(model.parameters()).device
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tok.model_max_length or 4096,
    ).to(device)

    do_sample = temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tok.pad_token_id,
        )

    # Decode only the newly generated tokens.
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    completions: list[str] = []
    for i, full_ids in enumerate(out):
        gen_ids = full_ids[input_lens[i] :]
        completions.append(tok.decode(gen_ids, skip_special_tokens=True))
    return completions


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Canonical JSONL test file")
    p.add_argument("--output", required=True, help="Predictions JSONL output path")
    p.add_argument("--model-path", required=True, help="Base model (HF repo or local)")
    p.add_argument("--adapter-path", default=None, help="Optional LoRA adapter dir")
    p.add_argument(
        "--prompt-mode",
        default="unified",
        choices=["kv", "entity", "relation", "unified"],
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Computation dtype. H100/A100 -> bf16, V100 -> fp16.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Only run on the first N records (0 = all). Default 200 for speed.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(read_jsonl(in_path))
    if args.limit > 0:
        records = records[: args.limit]
    logger.info(f"Loaded {len(records)} records from {in_path}")

    if args.dry_run:
        logger.info(
            "DRY RUN — model=%s adapter=%s mode=%s n=%d",
            args.model_path,
            args.adapter_path,
            args.prompt_mode,
            len(records),
        )
        # Still emit one example prompt so the user can eyeball it.
        example = build_prompt(records[0], args.prompt_mode) if records else ""
        logger.info("Example prompt:\n%s", example[:1200])
        return 0

    model, tok = load_model(args.model_path, args.adapter_path, args.dtype)

    prompts: list[str] = [build_prompt(r, args.prompt_mode) for r in records]
    outputs: list[dict] = []
    for start in range(0, len(records), args.batch_size):
        end = min(start + args.batch_size, len(records))
        batch_prompts = prompts[start:end]
        completions = generate_batch(
            model,
            tok,
            batch_prompts,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
        )
        for rec, prompt, pred in zip(records[start:end], batch_prompts, completions):
            outputs.append(
                {
                    "id": rec.get("id"),
                    "prompt": prompt,
                    "prediction": pred,
                    "gold": rec,
                }
            )
        logger.info(f"  generated {end}/{len(records)}")

    write_jsonl(out_path, outputs)
    logger.info(f"Wrote {len(outputs)} predictions -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
