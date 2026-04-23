"""Tests for IE metric parsing robustness."""

import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MODULE_PATH = _REPO_ROOT / "src/training/ie_metrics.py"
_SPEC = importlib.util.spec_from_file_location("ie_metrics", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
parse_prediction = _MODULE.parse_prediction


def test_parse_prediction_recovers_first_valid_output_json():
    raw = """
Input text:
foo

Output: {"kv": {}, "entity": [{"text": "foo", "type": "X", "start": null, "end": null}], "relation": []}

```json
{"kv": {}, "entity": [{"head": "bad", "relation": "oops"}]}
""".strip()

    parsed = parse_prediction(raw)

    assert parsed["kv"] == {}
    assert parsed["relation"] == []
    assert parsed["entity"] == [{"text": "foo", "type": "X", "start": None, "end": None}]


def test_parse_prediction_skips_prompt_echo_before_output_marker():
    raw = """
Output format:
{
  "kv": {
    "field_name": "value_or_null",
    ...
  }
}

Input text:
bar

Output: {"kv": {}, "entity": [{"text": "bar", "type": "Y", "start": null, "end": null}], "relation": []}]}]}]}
""".strip()

    parsed = parse_prediction(raw)

    assert parsed["kv"] == {}
    assert parsed["relation"] == []
    assert parsed["entity"] == [{"text": "bar", "type": "Y", "start": None, "end": None}]


def test_parse_prediction_accepts_valid_empty_answer():
    raw = '{"kv": {}, "entity": [], "relation": []}'

    parsed = parse_prediction(raw)

    assert parsed == {"kv": {}, "entity": [], "relation": []}
