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
