# IE evaluation

End-to-end scoring pipeline for IE SFT runs. Unlike `llamafactory-cli eval`
(which measures loss / MC accuracy), this pipeline scores extraction
quality: KV field F1, entity (type, text) set F1, and relation triple F1.

## Pieces

| File | Role |
|---|---|
| `run_predict.py` | Batch generation from a base model (+ optional LoRA) on a canonical test JSONL. |
| `compute_metrics.py` | Scores a predictions JSONL with `src/training/ie_metrics.py`. |
| `evaluate_end_to_end.py` | Runs predict + metrics back-to-back. |
| `run_eval_scenario.sh` | Thin bash dispatcher. Parametrized by `--model / --variant / --mode`. |
| `run_eval_{qwen3,qwen35}_{lora,full}.sh` | Wrappers for each model x variant. |
| `run_eval_all.sh` | Sweeps the full matrix and skips missing checkpoints. |
| `run_eval_compare_models.sh` | Compares arbitrary HF/local models in one sweep and writes a summary table. |

## Scenarios

The 4-wrapper grid covers:

- `qwen3 / qwen3.5` (0.6B / 0.8B base)
- `lora / full` (LoRA adapter on top of hub base, or a merged full-SFT dir)
- `kv / entity / relation / unified` prompt modes

```bash
# LoRA, unified prompt (default)
bash scripts/eval/run_eval_qwen3_lora.sh

# LoRA, KV only
MODE=kv bash scripts/eval/run_eval_qwen3_lora.sh

# Full SFT, relation only, 50-record smoke test
LIMIT=50 MODE=relation bash scripts/eval/run_eval_qwen35_full.sh

# Whole matrix
bash scripts/eval/run_eval_all.sh

# Compare arbitrary models / checkpoints
bash scripts/eval/run_eval_compare_models.sh \
  --candidate baseline=Qwen/Qwen3.5-0.8B \
  --candidate tuned=Qwen/Qwen3.5-0.8B::outputs/qwen3.5-0.8b-ie-lora
```

## Outputs

Each scenario writes to `outputs/eval/<tag>-<variant>-<mode>/`:

- `test_predictions.jsonl` — one line per record: `id / prompt / prediction / gold`
- `metrics.json` — PRF1 per task, plus parse-failure count

Use `--per-record` to append a per-row breakdown to `metrics.json` for
error analysis.

`run_eval_compare_models.sh` writes a separate run directory under
`outputs/eval/comparisons/<run-name>/`:

- `<candidate>/test_predictions.jsonl` — raw predictions per candidate
- `<candidate>/metrics.json` — task metrics for that candidate
- `summary.json` — aggregated comparison summary
- `summary.md` — compact markdown table for quick inspection

## Compare Candidates

Use repeated `--candidate` flags to compare stronger base models, local
merged checkpoints, or LoRA adapters.

Candidate spec format:

- `NAME=MODEL_PATH`
- `NAME=MODEL_PATH::ADAPTER_PATH`
- `MODEL_PATH`

Examples:

```bash
# Larger base models or external checkpoints
bash scripts/eval/run_eval_compare_models.sh \
  --candidate qwen35-large=<hf-model-id> \
  --candidate local-full=outputs/qwen3.5-0.8b-ie-full-ds \
  --mode unified

# Base vs LoRA on the same foundation model
bash scripts/eval/run_eval_compare_models.sh \
  --candidate base=Qwen/Qwen3.5-0.8B \
  --candidate lora=Qwen/Qwen3.5-0.8B::outputs/qwen3.5-0.8b-ie-lora \
  --limit 50
```

Useful flags:

- `--run-name` to keep multiple comparison experiments side by side
- `--continue-on-error` to keep running if one large model fails
- `--dry-run` to print the planned sweep without loading models

## Env knobs

| Var | Default | Notes |
|---|---|---|
| `TEST_FILE` | `data/processed/splits/test.jsonl` | Canonical JSONL test set |
| `BATCH_SIZE` | 8 | Generation batch size |
| `MAX_NEW` | 512 | `--max-new-tokens` |
| `LIMIT` | 0 | Run on first N records; 0 = all |
| `DTYPE` | `bf16` | `bf16` (A100/H100) / `fp16` (V100) / `fp32` |
