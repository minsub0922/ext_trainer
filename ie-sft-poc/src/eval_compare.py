"""Helpers for comparing arbitrary IE evaluation candidates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASK_TYPES = ("kv", "entity", "relation", "relation_typed")


@dataclass(frozen=True)
class EvalCandidate:
    """One model candidate in a comparison run."""

    name: str
    model_path: str
    adapter_path: str | None = None

    @property
    def slug(self) -> str:
        return slugify(self.name)


@dataclass
class EvalResult:
    """Result bundle for one evaluated candidate."""

    candidate: EvalCandidate
    output_dir: Path
    scenario: str
    status: str
    metrics: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.candidate.name,
            "slug": self.candidate.slug,
            "model_path": self.candidate.model_path,
            "adapter_path": self.candidate.adapter_path,
            "scenario": self.scenario,
            "status": self.status,
            "output_dir": str(self.output_dir),
        }
        if self.metrics is not None:
            payload["metrics"] = self.metrics
        if self.error:
            payload["error"] = self.error
        return payload


def slugify(value: str) -> str:
    """Convert a free-form label into a stable filesystem slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "candidate"


def infer_candidate_name(model_path: str) -> str:
    """Infer a readable candidate name from a model path or repo id."""
    cleaned = model_path.strip().rstrip("/")
    if not cleaned:
        return "candidate"
    return cleaned.split("/")[-1] or "candidate"


def parse_candidate_spec(spec: str) -> EvalCandidate:
    """Parse ``NAME=MODEL_PATH[::ADAPTER_PATH]`` or ``MODEL_PATH[::ADAPTER_PATH]``."""
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("Empty candidate spec")

    adapter_path: str | None = None
    model_spec = raw
    if "::" in raw:
        model_spec, adapter_spec = raw.split("::", 1)
        adapter_path = adapter_spec.strip() or None
        if adapter_path is None:
            raise ValueError(
                "Adapter path is empty. Expected MODEL_PATH::ADAPTER_PATH or "
                "NAME=MODEL_PATH::ADAPTER_PATH"
            )

    if "=" in model_spec:
        name, model_path = model_spec.split("=", 1)
        name = name.strip()
        model_path = model_path.strip()
        if not name or not model_path:
            raise ValueError(
                "Candidate spec must look like NAME=MODEL_PATH[::ADAPTER_PATH]"
            )
    else:
        model_path = model_spec.strip()
        if not model_path:
            raise ValueError("Model path is empty in candidate spec")
        name = infer_candidate_name(model_path)

    return EvalCandidate(name=name, model_path=model_path, adapter_path=adapter_path)


def make_run_name(mode: str, candidates: list[EvalCandidate]) -> str:
    """Build a short default run name that is stable across reruns."""
    if not candidates:
        return f"compare-{mode}"
    if len(candidates) == 1:
        return f"compare-{mode}-{candidates[0].slug}"
    return f"compare-{mode}-{len(candidates)}models"


def ensure_unique_candidate_slugs(candidates: list[EvalCandidate]) -> None:
    """Fail fast when multiple candidates would write into the same directory."""
    seen: dict[str, str] = {}
    for candidate in candidates:
        slug = candidate.slug
        previous = seen.get(slug)
        if previous is not None:
            raise ValueError(
                "Duplicate candidate slug '{slug}' from '{previous}' and '{current}'. "
                "Use explicit NAME=MODEL_PATH values to disambiguate.".format(
                    slug=slug,
                    previous=previous,
                    current=candidate.name,
                )
            )
        seen[slug] = candidate.name


def _metric_value(metrics: dict[str, Any] | None, task_type: str, key: str = "f1") -> float | None:
    if not metrics:
        return None
    task_metrics = metrics.get(task_type)
    if not isinstance(task_metrics, dict):
        return None
    value = task_metrics.get(key)
    return value if isinstance(value, (int, float)) else None


def build_summary(
    *,
    run_name: str,
    mode: str,
    test_file: Path,
    summary_dir: Path,
    results: list[EvalResult],
) -> dict[str, Any]:
    """Build a serializable comparison summary."""
    rankings: dict[str, list[dict[str, Any]]] = {}
    for task_type in TASK_TYPES:
        task_rows: list[dict[str, Any]] = []
        for result in results:
            f1 = _metric_value(result.metrics, task_type)
            if f1 is None:
                continue
            task_rows.append(
                {
                    "name": result.candidate.name,
                    "f1": round(float(f1), 4),
                    "status": result.status,
                    "output_dir": str(result.output_dir),
                }
            )
        if task_rows:
            rankings[task_type] = sorted(task_rows, key=lambda row: row["f1"], reverse=True)

    return {
        "run_name": run_name,
        "mode": mode,
        "test_file": str(test_file),
        "summary_dir": str(summary_dir),
        "n_candidates": len(results),
        "n_success": sum(1 for result in results if result.status == "success"),
        "n_failed": sum(1 for result in results if result.status != "success"),
        "results": [result.to_dict() for result in results],
        "rankings": rankings,
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    """Render a compact markdown summary for quick manual inspection."""

    def fmt_f1(result: dict[str, Any], task_type: str) -> str:
        metrics = result.get("metrics") or {}
        task_metrics = metrics.get(task_type) or {}
        value = task_metrics.get("f1")
        return f"{value:.4f}" if isinstance(value, (int, float)) else "-"

    def fmt_parse_failures(result: dict[str, Any]) -> str:
        metrics = result.get("metrics") or {}
        value = metrics.get("n_parse_failures")
        return str(value) if value is not None else "-"

    lines = [
        f"# IE Eval Comparison: {summary['run_name']}",
        "",
        f"- mode: `{summary['mode']}`",
        f"- test_file: `{summary['test_file']}`",
        f"- success: {summary['n_success']} / {summary['n_candidates']}",
        "",
        "| candidate | status | kv_f1 | entity_f1 | relation_f1 | parse_failures | output_dir |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for result in summary.get("results", []):
        status = result.get("status", "-")
        if result.get("error"):
            status = f"{status} ({result['error']})"
        lines.append(
            "| {name} | {status} | {kv} | {entity} | {relation} | {parse_failures} | {output_dir} |".format(
                name=result.get("name", "-"),
                status=status,
                kv=fmt_f1(result, "kv"),
                entity=fmt_f1(result, "entity"),
                relation=fmt_f1(result, "relation"),
                parse_failures=fmt_parse_failures(result),
                output_dir=result.get("output_dir", "-"),
            )
        )

    lines.extend(
        [
            "",
            "| candidate | model_path | adapter_path |",
            "|---|---|---|",
        ]
    )
    for result in summary.get("results", []):
        lines.append(
            "| {name} | {model_path} | {adapter_path} |".format(
                name=result.get("name", "-"),
                model_path=result.get("model_path", "-"),
                adapter_path=result.get("adapter_path") or "-",
            )
        )

    return "\n".join(lines) + "\n"


def write_summary(summary: dict[str, Any], summary_dir: Path) -> tuple[Path, Path]:
    """Write both JSON and markdown summaries to disk."""
    summary_dir.mkdir(parents=True, exist_ok=True)
    json_path = summary_dir / "summary.json"
    md_path = summary_dir / "summary.md"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(render_summary_markdown(summary), encoding="utf-8")
    return json_path, md_path


def load_metrics(metrics_path: Path) -> dict[str, Any]:
    """Load a metrics JSON file from a completed scenario."""
    return json.loads(metrics_path.read_text(encoding="utf-8"))
