"""Build DPO preference pairs from SFT-model samples scored by ie_metrics.

Pipeline:
    1. For each prompt in the train split, sample K completions from the
       stage-2 SFT model (high temperature for diversity).
    2. Score each sample with src/training/ie_metrics.evaluate against the
       canonical gold record. Use the per-record F1 of the active task
       (kv / entity / relation) as the scalar reward.
    3. Keep a pair (chosen, rejected) iff margin = r_best - r_worst >=
       `min_margin`. This filters uninformative pairs.
    4. Emit LLaMA-Factory `pairwise`-style records:
         {"instruction": str, "input": str, "output": [chosen, rejected]}

The expensive generation step is kept in a separate CLI script
(scripts/preprocess/olmo3_style/build_preference_pairs.py) so unit tests
can mock it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ..ie_metrics import evaluate


@dataclass
class PairBuilderConfig:
    min_margin: float = 0.15
    max_pairs_per_prompt: int = 1
    task_types: tuple[str, ...] = ("kv", "entity", "relation")
    allow_gold_fallback: bool = True


def _primary_score(metrics_obj, task_type: str) -> float:
    """Pick the scalar F1 for this record's active task."""
    if task_type == "kv":
        return float(metrics_obj.kv.f1)
    if task_type == "entity":
        return float(metrics_obj.entity.f1)
    if task_type == "relation":
        return float(metrics_obj.relation.f1)
    # fallback: macro over whatever scored > 0
    scores = [
        task.f1
        for task in (metrics_obj.kv, metrics_obj.entity, metrics_obj.relation)
        if task is not None
    ]
    non_zero = [s for s in scores if s > 0]
    return float(sum(non_zero) / len(non_zero)) if non_zero else 0.0


def _score_sample(prediction: str, gold: dict, task_type: str) -> float:
    """Score a single (prediction, gold) pair and return the primary F1."""
    m = evaluate([prediction], [gold], task_types=[task_type], log_summary=False)
    return _primary_score(m, task_type)


def _gold_answer_text(gold: dict) -> str:
    """Serialize the canonical answer as a strict JSON completion."""
    answer = gold.get("answer") or {}
    normalized = {
        "kv": answer.get("kv") or {},
        "entity": answer.get("entity") or [],
        "relation": answer.get("relation") or [],
    }
    return json.dumps(normalized, ensure_ascii=False)


def build_preference_pairs(
    samples_iter: Iterable[dict],
    cfg: PairBuilderConfig | None = None,
    scorer: Callable[[str, dict, str], float] | None = None,
) -> list[dict]:
    """Convert per-prompt sample groups into DPO pairs.

    Each element of `samples_iter` is expected to be:
        {
          "prompt": str,
          "instruction": str,
          "input": str,
          "gold": <canonical record dict>,
          "task_type": "kv" | "entity" | "relation",
          "samples": [str, str, ...],
        }
    """
    cfg = cfg or PairBuilderConfig()
    scorer = scorer or _score_sample
    pairs: list[dict] = []

    for group in samples_iter:
        samples = group["samples"]
        if len(samples) < 2:
            continue
        gold = group["gold"]
        task_type = group.get("task_type") or "kv"
        start_len = len(pairs)
        scored = [(s, scorer(s, gold, task_type)) for s in samples]
        scored.sort(key=lambda t: t[1], reverse=True)

        n = min(cfg.max_pairs_per_prompt, len(scored) // 2)
        for i in range(n):
            chosen, r_c = scored[i]
            rejected, r_r = scored[-(i + 1)]
            if r_c - r_r < cfg.min_margin:
                continue
            pairs.append({
                "instruction": group["instruction"],
                "input": group.get("input", ""),
                "output": [chosen, rejected],
                "metadata": {
                    "task_type": task_type,
                    "reward_chosen": r_c,
                    "reward_rejected": r_r,
                    "margin": r_c - r_r,
                },
            })

        if len(pairs) > start_len:
            continue

        if not cfg.allow_gold_fallback:
            continue

        gold_answer = _gold_answer_text(gold)
        gold_score = scorer(gold_answer, gold, task_type)
        rejected, rejected_score = scored[-1]
        margin = gold_score - rejected_score
        if margin < cfg.min_margin:
            continue
        pairs.append({
            "instruction": group["instruction"],
            "input": group.get("input", ""),
            "output": [gold_answer, rejected],
            "metadata": {
                "task_type": task_type,
                "reward_chosen": gold_score,
                "reward_rejected": rejected_score,
                "margin": margin,
                "fallback": "gold_answer",
            },
        })
    return pairs


def write_pairs_jsonl(pairs: list[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return out_path
