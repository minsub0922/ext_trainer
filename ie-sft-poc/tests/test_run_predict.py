"""Tests for evaluation prediction generation helpers."""

import importlib.util
import sys
import types
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# `run_predict.py` imports the project IO helpers at module load time; this
# test only exercises generation slicing, so keep optional YAML dependency out.
sys.modules.setdefault(
    "yaml",
    types.SimpleNamespace(safe_load=lambda *_args, **_kwargs: None),
)

_MODULE_PATH = _REPO_ROOT / "scripts/eval/run_predict.py"
_SPEC = importlib.util.spec_from_file_location("run_predict", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
generate_batch = _MODULE.generate_batch


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeParam:
    device = "cpu"


class _FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    model_max_length = 4096
    pad_token_id = 0

    def __call__(self, prompts, **kwargs):
        # Simulate a left-padded batch with width 6. The first prompt has
        # only 3 real tokens, which is the regression case for sum(mask).
        return _FakeBatch(
            {
                "input_ids": _FakeTensor(
                    [
                        ["<pad>", "<pad>", "<pad>", "p1", "p2", "p3"],
                        ["q1", "q2", "q3", "q4", "q5", "q6"],
                    ]
                ),
                "attention_mask": _FakeTensor(
                    [
                        [0, 0, 0, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                    ]
                ),
            }
        )

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


class _FakeModel:
    def parameters(self):
        return iter([_FakeParam()])

    def generate(self, **kwargs):
        # Transformers returns the padded input prefix followed by newly
        # generated tokens. Only tokens after padded input width are completion.
        return [
            ["<pad>", "<pad>", "<pad>", "p1", "p2", "p3", "answer-a"],
            ["q1", "q2", "q3", "q4", "q5", "q6", "answer-b"],
        ]


def test_generate_batch_slices_after_padded_input_width(monkeypatch):
    fake_torch = types.SimpleNamespace(no_grad=lambda: _NoGrad())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    completions = generate_batch(
        _FakeModel(),
        _FakeTokenizer(),
        ["short prompt", "long prompt"],
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
    )

    assert completions == ["answer-a", "answer-b"]
