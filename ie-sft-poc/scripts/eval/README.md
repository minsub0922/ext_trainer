# IE evaluation

End-to-end scoring pipeline for IE SFT runs. Unlike `llamafactory-cli eval`
(which measures loss / MC accuracy), this pipeline scores extraction
quality: KV field F1, entity (type, text) set F1, and relation triple F1.

## Pieces

| File | Role |
|---|---|
| `run_predict.py` | Batch generation from a base model (+ optional LoRA) on a canonical test JSONL. |
| `compute_metrics.py` | Scores one or more predictions JSONLs with `src/training/ie_metrics.py`. |
| `evaluate_end_to_end.py` | Runs predict + metrics back-to-back. |
| `run_eval_scenario.sh` | Thin bash dispatcher. Parametrized by `--model / --variant / --mode`. |
| `run_eval_{qwen3,qwen35}_{lora,full}.sh` | Wrappers for each model x variant. |
| `run_eval_{qwen3,qwen35}_olmo3_stage{2,3,4}_3ep.sh` | Wrappers for the separated OLMo3-style branch after stage-2 3ep. |
| `run_eval_qwen3_8b_base.sh` | Wrapper for `Qwen/Qwen3-8B-Base`. |
| `run_eval_qwen35_9b_base.sh` | Wrapper for `Qwen/Qwen3.5-9B-Base`. |
| `run_eval_all.sh` | Sweeps the full matrix and skips missing checkpoints. |

## Scenarios

The common wrappers cover:

- `qwen3 / qwen3.5` (0.6B / 0.8B base) with `lora / full`
- `qwen3-8b / qwen3.5-9b` base-only wrappers
- `kv / entity / relation / unified` prompt modes

```bash
# LoRA, unified prompt (default)
bash scripts/eval/run_eval_qwen3_lora.sh

# LoRA, KV only
MODE=kv bash scripts/eval/run_eval_qwen3_lora.sh

# Full SFT, relation only, 50-record smoke test
LIMIT=50 MODE=relation bash scripts/eval/run_eval_qwen35_full.sh

# OLMo3-style stage 2, 3-epoch ablation
bash scripts/eval/run_eval_qwen3_olmo3_stage2_3ep.sh
bash scripts/eval/run_eval_qwen35_olmo3_stage2_3ep.sh
bash scripts/eval/run_eval_qwen3_olmo3_stage3_3ep.sh
bash scripts/eval/run_eval_qwen35_olmo3_stage3_3ep.sh
bash scripts/eval/run_eval_qwen3_olmo3_stage4_3ep.sh
bash scripts/eval/run_eval_qwen35_olmo3_stage4_3ep.sh

# Base models
bash scripts/eval/run_eval_qwen3_8b_base.sh
bash scripts/eval/run_eval_qwen35_9b_base.sh

# Whole matrix
bash scripts/eval/run_eval_all.sh
```

## Outputs

Each scenario writes to `outputs/eval/<tag>-<variant>-<mode>/`:

- `test_predictions.jsonl` — one line per record: `id / prompt / prediction / gold`
- `metrics.json` — PRF1 per task, plus parse-failure count

Use `--per-record` to append a per-row breakdown to `metrics.json` for
error analysis.

You can recompute metrics later from existing predictions files without
specifying an output path. `compute_metrics.py` writes `metrics.json`
next to each input predictions file:

```bash
python scripts/eval/compute_metrics.py \
  --predictions outputs/eval/qwen3-8b-base-unified/test_predictions.jsonl

python scripts/eval/compute_metrics.py \
  --predictions \
    outputs/eval/qwen3-8b-base-unified/test_predictions.jsonl \
    outputs/eval/qwen3.5-9b-base-unified/test_predictions.jsonl
```

## Env knobs

| Var | Default | Notes |
|---|---|---|
| `TEST_FILE` | `data/processed/splits/test.jsonl` | Canonical JSONL test set |
| `BATCH_SIZE` | 8 | Generation batch size |
| `MAX_NEW` | 512 | `--max-new-tokens` |
| `LIMIT` | 0 | Run on first N records; 0 = all |
| `DTYPE` | `bf16` | `bf16` (A100/H100) / `fp16` (V100) / `fp32` |
