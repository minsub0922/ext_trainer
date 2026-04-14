# OLMo3-Style 4-Stage Training Scripts

Launchers for the OLMo3-style training recipe (mid-train → SFT → DPO → RLVR) applied to Qwen3 / Qwen3.5 base models. The recipe is described in detail in [`docs/olmo3_style_pipeline.md`](../../../docs/olmo3_style_pipeline.md); this README covers the scripts in this directory.

## Contents

| Script | Stage | What it does |
| --- | --- | --- |
| `run_stage1_midtrain.sh` | 1. Mid-training | Continued pretraining on the IE-heavy midtrain mixture. Output warm-starts stage 2. |
| `run_stage2_sft.sh` | 2. SFT | Full-parameter supervised fine-tuning on `ie_sft_unified`, starting from the stage-1 checkpoint. |
| `run_stage3_dpo.sh` | 3. DPO | Direct Preference Optimization on pairs sampled from the stage-2 model. |
| `run_stage4_rlvr.sh` | 4. RLVR | GRPO-lite with verifiable IE-F1 rewards; uses the custom trainer in `src/training/olmo3_style/rlvr_trainer.py`. |
| `run_pipeline_qwen3.sh` | all | Runs stages 1–4 back-to-back for Qwen3-0.6B, invoking the **stage-specific** data-prep scripts as needed. Assumes base splits already exist. |
| `run_pipeline_qwen35.sh` | all | Same, for Qwen3.5-0.8B. |
| `run_oneshot_qwen3.sh` | all + base prep | **Zero-to-trained** for Qwen3-0.6B: runs the base data pipeline (download → normalize → unify → split → export) then hands off to `run_pipeline_qwen3.sh`. Idempotent; skips any phase whose output already exists. |
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

Stages 1–3 launch via `torchrun`. Two env vars are respected:

- `NPROC` — number of GPUs per node. Defaults to `nvidia-smi -L | wc -l` (falls back to `1`).
- `MASTER_PORT` — torchrun rendezvous port. Defaults are staggered per stage (29501 / 29502 / 29503) so concurrent runs don't collide.

Stage 4 (RLVR) runs as a plain `python -m` process because the custom GRPO-lite trainer manages its own sampler/evaluator workers.

## Data prerequisites

The stage scripts assume the following files exist. `run_pipeline_*.sh` builds any missing ones automatically; if you run stages individually you may need to prep them yourself.

| File | Produced by | Consumed by |
| --- | --- | --- |
| `data/processed/splits/train.jsonl` | core preprocessing (see main README) | all prep steps |
| `data/processed/llamafactory/*.jsonl` + `dataset_info.json` | `scripts/export/export_to_llamafactory.py` | stage 2 (via `dataset: ie_sft_unified`) |
| `data/processed/olmo3_style/midtrain.jsonl` | `scripts/preprocess/olmo3_style/build_midtrain_mixture.py` | stage 1 |
| `data/processed/olmo3_style/preference_pairs.jsonl` | `scripts/preprocess/olmo3_style/build_preference_pairs.py` (needs stage-2 ckpt) | stage 3 |
| `data/processed/olmo3_style/rlvr_prompts.jsonl` | `scripts/preprocess/olmo3_style/build_rlvr_prompts.py` | stage 4 |

## Output layout

All checkpoints land under `outputs/olmo3_style/<tag>/stageN_*/`, where `<tag>` is e.g. `qwen3-0.6b` or `qwen3.5-…`. Each stage reads the previous stage's output dir from its config's `model_name_or_path` field — edit the YAML if you want to skip a stage or point at a different starting model.

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
