# OLMo3-Style 4-Stage Training Scripts

Launchers for the OLMo3-style training recipe (mid-train → SFT → DPO → RLVR) applied to Qwen3 / Qwen3.5 base models. The recipe is described in detail in [`docs/olmo3_style_pipeline.md`](../../../docs/olmo3_style_pipeline.md); this README covers the scripts in this directory.

## Contents

| Script | Stage | What it does |
| --- | --- | --- |
| `run_stage.sh` | 1–4 | **Unified stage launcher.** `--stage {1\|2\|2-3ep\|3\|3-3ep\|4\|4-3ep} --model {qwen3\|qwen3.5}`. Handles flash-attn detection, dynamic port, multi-GPU via `llamafactory-cli`. |
| `run_stage1_midtrain.sh` | 1 | Thin wrapper → `run_stage.sh --stage 1`. |
| `run_stage2_sft.sh` | 2 | Thin wrapper → `run_stage.sh --stage 2`. |
| `run_stage2_sft_3ep.sh` | 2 | Thin wrapper → `run_stage.sh --stage 2-3ep`; keeps stage 1 fixed and trains stage 2 for 3 epochs. |
| `run_stage3_dpo.sh` | 3 | Thin wrapper → `run_stage.sh --stage 3`. |
| `run_stage3_dpo_3ep.sh` | 3 | Thin wrapper → `run_stage.sh --stage 3-3ep`; DPO branch after stage2 3ep. |
| `run_stage4_rlvr.sh` | 4 | Thin wrapper → `run_stage.sh --stage 4`. |
| `run_stage4_rlvr_3ep.sh` | 4 | Thin wrapper → `run_stage.sh --stage 4-3ep`; RLVR branch after stage2 3ep. |
| `run_pipeline.sh` | all | **Unified pipeline.** `--model {qwen3\|qwen3.5}`. Runs data prep + stages 1–4. |
| `run_pipeline_stage2_3ep.sh` | 2–4 | Continues from an existing stage-1 checkpoint, then runs the separated 3ep branch through stage 4. |
| `run_pipeline_from_stage3.sh` | 3–4 | Resumes from an existing stage-2 checkpoint. `--stage2 {2ep\|3ep}` chooses `stage2_sft` vs `stage2_sft_3ep`; add `--stage3-only` to stop after DPO. |
| `run_pipeline_qwen3.sh` | all | Thin wrapper → `run_pipeline.sh --model qwen3`. |
| `run_pipeline_qwen35.sh` | all | Thin wrapper → `run_pipeline.sh --model qwen3.5`. |
| `run_oneshot_qwen3.sh` | all + base prep | Runs base data pipeline then `run_pipeline.sh --model qwen3`. |
| `run_oneshot_qwen35.sh` | all + base prep | Same, for Qwen3.5-0.8B. |
| `_oneshot_common.sh` | — | Shared `run_data_pipeline` helper sourced by the oneshots. |

Each stage launcher reads a YAML config under `configs/olmo3_style/<model>/stageN_*.yaml`.

## Model selection

Every single-stage script honors the `MODEL` env var:

```bash
MODEL=qwen3    bash scripts/train/olmo3_style/run_stage1_midtrain.sh   # default
MODEL=qwen3.5  bash scripts/train/olmo3_style/run_stage1_midtrain.sh
```

`qwen3` maps to `configs/olmo3_style/qwen3/…`, `qwen3.5` to `configs/olmo3_style/qwen3_5/…`. Any other value aborts with `bad MODEL`.

The pipeline wrappers (`run_pipeline_qwen3.sh`, `run_pipeline_qwen35.sh`) set `MODEL` for you and also drive the data-prep steps.

## Distributed training knobs

Stages 1–3 launch via `llamafactory-cli` with `FORCE_TORCHRUN=1` for multi-GPU. Two env vars are respected:

- `NPROC` — number of GPUs per node. Defaults to `nvidia-smi -L | wc -l` (falls back to `1`).
- `MASTER_PORT` — rendezvous port. **Auto-detected** to an available port (starting from 29500) so concurrent runs don't collide.

Stage 4 (RLVR) runs as a plain `python -m` process because the custom GRPO-lite trainer manages its own sampler/evaluator workers.

## Data prerequisites

The stage scripts assume the following files exist. `run_pipeline_*.sh` builds any missing ones automatically; if you run stages individually you may need to prep them yourself.

| File | Produced by | Consumed by |
| --- | --- | --- |
| `data/processed/splits/train.jsonl` | core preprocessing (see main README) | all prep steps |
| `data/processed/llamafactory/*.jsonl` + `dataset_info.json` | `scripts/export/export_to_llamafactory.py` | stage 2 (via `dataset: ie_sft_unified`) |
| `data/processed/olmo3_style/midtrain.jsonl` | `scripts/preprocess/olmo3_style/build_midtrain_mixture.py` | stage 1 |
| `data/processed/olmo3_style/preference_pairs.jsonl` | `scripts/preprocess/olmo3_style/build_preference_pairs.py` (needs stage-2 ckpt) | stage 3 |
| `data/processed/olmo3_style/preference_pairs_3ep.jsonl` | `scripts/preprocess/olmo3_style/build_preference_pairs.py` (needs stage2_sft_3ep ckpt) | stage 3 3ep branch |
| `data/processed/olmo3_style/rlvr_prompts.jsonl` | `scripts/preprocess/olmo3_style/build_rlvr_prompts.py` | stage 4 |

## Output layout

All checkpoints land under `outputs/olmo3_style/<tag>/stageN_*/`, where `<tag>` is e.g. `qwen3-0.6b` or `qwen3.5-…`. The 3ep branch writes to `stage2_sft_3ep`, `stage3_dpo_3ep`, and `stage4_rlvr_3ep` so it can be compared with the default branch. Each stage reads the previous stage's output dir from its config's `model_name_or_path` field — edit the YAML if you want to skip a stage or point at a different starting model.

## Typical runbook

Cold box, single model, full pipeline:

```bash
# one-time: preprocess + export unified SFT data (see repo README §§ 4–6)
bash scripts/train/olmo3_style/run_pipeline_qwen3.sh
```

Iterating on a single stage (e.g. tweaking DPO hyperparams):

```bash
MODEL=qwen3 NPROC=4 bash scripts/train/olmo3_style/run_stage3_dpo.sh
```

Running the 3-epoch stage-2 SFT ablation:

```bash
MODEL=qwen3   NPROC=4 bash scripts/train/olmo3_style/run_stage2_sft_3ep.sh
MODEL=qwen3.5 NPROC=4 bash scripts/train/olmo3_style/run_stage2_sft_3ep.sh
```

Continuing the separated 3ep branch from an already-trained stage 1:

```bash
bash scripts/train/olmo3_style/run_pipeline_stage2_3ep_qwen3.sh
bash scripts/train/olmo3_style/run_pipeline_stage2_3ep_qwen35.sh
```

Starting from stage 3 after stage 2 is already trained:

```bash
# default stage2 branch (2 epochs): stage2_sft -> stage3_dpo -> stage4_rlvr
MODEL=qwen3 NPROC=4 bash scripts/train/olmo3_style/run_pipeline_from_stage3.sh --stage2 2ep

# 3-epoch stage2 branch: stage2_sft_3ep -> stage3_dpo_3ep -> stage4_rlvr_3ep
MODEL=qwen3 NPROC=4 bash scripts/train/olmo3_style/run_pipeline_from_stage3.sh --stage2 3ep

# only rebuild/register preference pairs and run DPO
MODEL=qwen3 NPROC=4 bash scripts/train/olmo3_style/run_pipeline_from_stage3.sh --stage2 3ep --stage3-only
```

Rebuilding preference pairs after a stage-2 retrain:

```bash
rm data/processed/olmo3_style/preference_pairs.jsonl
bash scripts/train/olmo3_style/run_pipeline_qwen3.sh   # resumes from the missing artifact
```

## Skipping stages

Each stage config's `model_name_or_path` points at the previous stage's `output_dir`. To bypass stage 1, open `configs/olmo3_style/<model>/stage2_sft.yaml` and set `model_name_or_path` back to the raw HF id (e.g. `Qwen/Qwen3-0.6B`); stage 2 then behaves like the plain full-SFT recipe in `configs/sft/`.

## Troubleshooting

- **`bad MODEL: …`** — the launcher only knows `qwen3` and `qwen3.5`. Add a new case block if you're introducing a new family.
- **`llamafactory-cli` not found** — stages 1–3 invoke it via `$(which llamafactory-cli)`. Install LLaMA-Factory into the active venv (`pip install -e LLaMA-Factory`).
- **Stage 4 OOM with small GPUs** — RLVR batches live rollouts; reduce `rollout_batch_size` / `num_generations` in `stage4_rlvr.yaml` before touching micro-batch size.
- **Resume from a mid-stage crash** — LLaMA-Factory writes `save_steps` checkpoints under each stage's `output_dir`; rerunning the stage picks up the latest one automatically.

## See also

- [`docs/olmo3_style_pipeline.md`](../../../docs/olmo3_style_pipeline.md) — recipe rationale, stage-by-stage design notes.
- [`configs/olmo3_style/`](../../../configs/olmo3_style/) — per-stage YAMLs for each model family.
- [`scripts/preprocess/olmo3_style/`](../../preprocess/olmo3_style/) — the three data-prep scripts.
- [`src/training/olmo3_style/rlvr_trainer.py`](../../../src/training/olmo3_style/rlvr_trainer.py) — custom RLVR trainer entry point for stage 4.
