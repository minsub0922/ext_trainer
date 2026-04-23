#!/usr/bin/env python3
"""Compare arbitrary IE evaluation candidates in one sweep.

Each candidate can be a HuggingFace model id or a local model directory.
Optional LoRA adapters are supported via ``::adapter_path``.

Examples:
    python scripts/eval/run_eval_compare_models.py \
        --candidate qwen35-8b=my-hf-org/my-model \
        --candidate my-full=outputs/qwen3.5-0.8b-ie-full-ds \
        --mode unified

    python scripts/eval/run_eval_compare_models.py \
        --candidate Qwen/Qwen3.5-0.8B \
        --candidate tuned=Qwen/Qwen3.5-0.8B::outputs/qwen3.5-0.8b-ie-lora \
        --limit 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.logging_utils import get_logger  # noqa: E402
from src.eval_compare import (  # noqa: E402
    EvalResult,
    build_summary,
    ensure_unique_candidate_slugs,
    load_metrics,
    make_run_name,
    parse_candidate_spec,
    write_summary,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help=(
            "Candidate spec: NAME=MODEL_PATH[::ADAPTER_PATH] or "
            "MODEL_PATH[::ADAPTER_PATH]. Repeat this flag for multiple models."
        ),
    )
    parser.add_argument(
        "--test",
        default="data/processed/splits/test.jsonl",
        help="Canonical JSONL test file (repo-relative by default)",
    )
    parser.add_argument(
        "--mode",
        default="unified",
        choices=["kv", "entity", "relation", "unified"],
    )
    parser.add_argument("--run-name", default=None, help="Output directory label")
    parser.add_argument(
        "--out-root",
        default="outputs/eval/comparisons",
        help="Comparison output root (repo-relative by default)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=200, help="0 = all")
    parser.add_argument("--per-record", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining candidates even if one fails",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else repo_root / path


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    test_file = _resolve_repo_path(repo_root, args.test)

    try:
        candidates = [parse_candidate_spec(spec) for spec in args.candidate]
        ensure_unique_candidate_slugs(candidates)
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    run_name = args.run_name or make_run_name(args.mode, candidates)
    summary_dir = _resolve_repo_path(repo_root, args.out_root) / run_name

    if args.dry_run:
        logger.info("DRY RUN: run_name=%s mode=%s test=%s", run_name, args.mode, test_file)
        for candidate in candidates:
            logger.info(
                "candidate=%s model=%s adapter=%s output=%s",
                candidate.name,
                candidate.model_path,
                candidate.adapter_path or "<none>",
                summary_dir / candidate.slug,
            )
        logger.info("summary -> %s/{summary.json,summary.md}", summary_dir)
        return 0

    if not test_file.exists():
        logger.error("Test file not found: %s", test_file)
        return 1

    evaluate_script = repo_root / "scripts/eval/evaluate_end_to_end.py"
    results: list[EvalResult] = []
    exit_code = 0

    for candidate in candidates:
        scenario = f"{run_name}-{candidate.slug}-{args.mode}"
        output_dir = summary_dir / candidate.slug
        cmd = [
            sys.executable,
            str(evaluate_script),
            "--scenario",
            scenario,
            "--test",
            str(test_file),
            "--model-path",
            candidate.model_path,
            "--prompt-mode",
            args.mode,
            "--output-dir",
            str(output_dir),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--batch-size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
            "--temperature",
            str(args.temperature),
            "--limit",
            str(args.limit),
        ]
        if candidate.adapter_path:
            cmd += ["--adapter-path", candidate.adapter_path]
        if args.per_record:
            cmd.append("--per-record")

        logger.info("Running comparison candidate '%s'", candidate.name)
        logger.info("Command: %s", " ".join(cmd))

        rc = subprocess.call(cmd, cwd=repo_root)
        if rc != 0:
            exit_code = rc
            results.append(
                EvalResult(
                    candidate=candidate,
                    output_dir=output_dir,
                    scenario=scenario,
                    status="failed",
                    error=f"rc={rc}",
                )
            )
            if not args.continue_on_error:
                break
            continue

        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            exit_code = 1
            results.append(
                EvalResult(
                    candidate=candidate,
                    output_dir=output_dir,
                    scenario=scenario,
                    status="failed",
                    error=f"missing metrics at {metrics_path}",
                )
            )
            if not args.continue_on_error:
                break
            continue

        results.append(
            EvalResult(
                candidate=candidate,
                output_dir=output_dir,
                scenario=scenario,
                status="success",
                metrics=load_metrics(metrics_path),
            )
        )

    summary = build_summary(
        run_name=run_name,
        mode=args.mode,
        test_file=test_file,
        summary_dir=summary_dir,
        results=results,
    )
    json_path, md_path = write_summary(summary, summary_dir)
    logger.info("Wrote comparison summary -> %s", json_path)
    logger.info("Wrote comparison markdown -> %s", md_path)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
