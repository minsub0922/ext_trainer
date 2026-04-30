"""Tests for OLMo3-style preference pair preparation."""

import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPT_PATH = _REPO_ROOT / "scripts/preprocess/olmo3_style/build_preference_pairs.py"
_SPEC = importlib.util.spec_from_file_location("build_preference_pairs_cli", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_canonical_gold = _MODULE._canonical_gold
_primary_task_type = _MODULE._primary_task_type

from src.training.olmo3_style.preference_builder import (  # noqa: E402
    PairBuilderConfig,
    build_preference_pairs,
)


def test_canonical_gold_preserves_answer_and_task_types():
    rec = {
        "id": "rel-1",
        "text": "Alice works at Acme.",
        "task_types": ["entity", "relation"],
        "answer": {
            "entity": [{"text": "Alice", "type": "PERSON"}],
            "relation": [
                {
                    "head": "Alice",
                    "head_type": "PERSON",
                    "relation": "works_for",
                    "tail": "Acme",
                    "tail_type": "ORG",
                }
            ],
        },
    }

    gold = _canonical_gold(rec)

    assert gold["answer"]["kv"] == {}
    assert gold["answer"]["entity"] == [{"text": "Alice", "type": "PERSON"}]
    assert gold["answer"]["relation"][0]["relation"] == "works_for"
    assert gold["task_types"] == ["entity", "relation"]


def test_primary_task_prefers_relation_over_entity():
    rec = {
        "answer": {
            "entity": [{"text": "Alice", "type": "PERSON"}],
            "relation": [{"head": "Alice", "relation": "works_for", "tail": "Acme"}],
        },
        "task_types": ["entity", "relation"],
    }

    assert _primary_task_type(rec) == "relation"


def test_canonical_gold_infers_legacy_answer_shape():
    rec = {
        "output": {
            "entity": [{"text": "Acme", "type": "ORG"}],
        },
    }

    gold = _canonical_gold(rec)

    assert gold["answer"] == {
        "kv": {},
        "entity": [{"text": "Acme", "type": "ORG"}],
        "relation": [],
    }
    assert gold["task_types"] == ["entity"]


def test_build_preference_pairs_uses_gold_fallback_when_samples_tie():
    gold = {
        "answer": {
            "kv": {"name": "Alice"},
            "entity": [],
            "relation": [],
        },
        "task_types": ["kv"],
    }
    groups = [{
        "instruction": "Extract as JSON",
        "input": "",
        "gold": gold,
        "task_type": "kv",
        "samples": ["not json", "{}"],
    }]

    def scorer(prediction, _gold, _task_type):
        return 1.0 if '"name": "Alice"' in prediction else 0.0

    pairs = build_preference_pairs(
        groups,
        PairBuilderConfig(min_margin=0.15, allow_gold_fallback=True),
        scorer=scorer,
    )

    assert len(pairs) == 1
    assert pairs[0]["output"][0] == (
        '{"kv": {"name": "Alice"}, "entity": [], "relation": []}'
    )
    assert pairs[0]["metadata"]["fallback"] == "gold_answer"


def test_build_preference_pairs_can_disable_gold_fallback():
    gold = {
        "answer": {
            "kv": {"name": "Alice"},
            "entity": [],
            "relation": [],
        },
        "task_types": ["kv"],
    }
    groups = [{
        "instruction": "Extract as JSON",
        "input": "",
        "gold": gold,
        "task_type": "kv",
        "samples": ["not json", "{}"],
    }]

    pairs = build_preference_pairs(
        groups,
        PairBuilderConfig(min_margin=0.15, allow_gold_fallback=False),
        scorer=lambda *_args: 0.0,
    )

    assert pairs == []
