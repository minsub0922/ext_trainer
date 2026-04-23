"""Tests for model comparison eval helpers."""

from pathlib import Path

from src.eval_compare import (
    EvalCandidate,
    EvalResult,
    build_summary,
    ensure_unique_candidate_slugs,
    make_run_name,
    parse_candidate_spec,
    render_summary_markdown,
)


def test_parse_candidate_spec_with_explicit_name_and_adapter():
    candidate = parse_candidate_spec(
        "qwen35-large=Qwen/Qwen3.5-0.8B::outputs/qwen3.5-0.8b-ie-lora"
    )

    assert candidate.name == "qwen35-large"
    assert candidate.model_path == "Qwen/Qwen3.5-0.8B"
    assert candidate.adapter_path == "outputs/qwen3.5-0.8b-ie-lora"
    assert candidate.slug == "qwen35-large"


def test_parse_candidate_spec_infers_name_from_model_path():
    candidate = parse_candidate_spec("Qwen/Qwen3.5-0.8B")

    assert candidate.name == "Qwen3.5-0.8B"
    assert candidate.model_path == "Qwen/Qwen3.5-0.8B"
    assert candidate.adapter_path is None


def test_make_run_name_uses_mode_and_candidate_count():
    candidates = [
        EvalCandidate(name="A", model_path="model/a"),
        EvalCandidate(name="B", model_path="model/b"),
    ]

    assert make_run_name("unified", candidates) == "compare-unified-2models"


def test_ensure_unique_candidate_slugs_rejects_collisions():
    candidates = [
        EvalCandidate(name="Qwen3.5-8B", model_path="org-a/model"),
        EvalCandidate(name="qwen3.5 8b", model_path="org-b/model"),
    ]

    try:
        ensure_unique_candidate_slugs(candidates)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for duplicate candidate slugs")


def test_render_summary_markdown_contains_candidate_rows():
    results = [
        EvalResult(
            candidate=EvalCandidate(name="baseline", model_path="model/base"),
            output_dir=Path("outputs/eval/comparisons/demo/baseline"),
            scenario="demo-baseline-unified",
            status="success",
            metrics={
                "n_records": 10,
                "n_parse_failures": 1,
                "kv": {"f1": 0.5},
                "entity": {"f1": 0.6},
                "relation": {"f1": 0.7},
            },
        ),
        EvalResult(
            candidate=EvalCandidate(
                name="adapter",
                model_path="Qwen/Qwen3.5-0.8B",
                adapter_path="outputs/qwen3.5-0.8b-ie-lora",
            ),
            output_dir=Path("outputs/eval/comparisons/demo/adapter"),
            scenario="demo-adapter-unified",
            status="failed",
            error="rc=1",
        ),
    ]
    summary = build_summary(
        run_name="demo",
        mode="unified",
        test_file=Path("data/processed/splits/test.jsonl"),
        summary_dir=Path("outputs/eval/comparisons/demo"),
        results=results,
    )

    markdown = render_summary_markdown(summary)

    assert "# IE Eval Comparison: demo" in markdown
    assert "| baseline | success | 0.5000 | 0.6000 | 0.7000 | 1 |" in markdown
    assert "| adapter | failed (rc=1) | - | - | - | - |" in markdown
