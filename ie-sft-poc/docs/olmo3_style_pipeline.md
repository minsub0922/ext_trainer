# OLMo3-style training pipeline (applied to Qwen3 / Qwen3.5)

This is **not** OLMo3 the model — it is the OLMo3 *training recipe*
applied on top of Qwen3-0.6B and Qwen3.5-0.8B. Four stages, each
starting from the previous stage's checkpoint:

```
  Qwen3(.5) base
        │
        ▼
┌─────────────────────┐
│ Stage 1: mid-train  │  continued pretraining on IE mixture (pt loss)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Stage 2: SFT        │  instruction tuning on canonical IE tasks
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Stage 3: DPO        │  preference pairs from stage-2 samples, scored
│                     │  by ie_metrics (chosen = higher F1)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ Stage 4: RLVR       │  GRPO-lite with verifiable rewards from
│                     │  ie_metrics — no reward model needed
└─────────────────────┘
```

## Why "verifiable" matters

For IE tasks the gold record is a dict of fields/entities/triples, and
`src/training/ie_metrics.py` turns a model completion into a scalar F1.
That F1 is the **verifiable reward**: deterministic, no trained reward
model, no annotator preference noise. Same reason RLVR works for
math/code — the IE-F1 is a provable scoring function.

## Files added by this recipe

```
configs/olmo3_style/
├── qwen3/
│   ├── stage1_midtrain.yaml     # pt loss, LR 5e-6, 1 epoch
│   ├── stage2_sft.yaml          # full SFT on mid-trained base, LR 2e-5
│   ├── stage3_dpo.yaml          # DPO, beta 0.1, LR 5e-7, ZeRO-3
│   └── stage4_rlvr.yaml         # GRPO-lite, K=4, IE-F1 reward
└── qwen3_5/                     # same 4 files, Qwen3.5-0.8B twin

src/training/olmo3_style/
├── __init__.py
├── preference_builder.py        # DPO pair builder (F1 margin filter)
└── rlvr_trainer.py              # reference GRPO-lite implementation

scripts/preprocess/olmo3_style/
├── build_midtrain_mixture.py    # canonical → packed text JSONL
├── build_preference_pairs.py    # stage-2 model samples → chosen/rejected
└── build_rlvr_prompts.py        # canonical → prompt + gold + task_type

scripts/train/olmo3_style/
├── run_stage1_midtrain.sh       # MODEL={qwen3|qwen3.5} dispatch
├── run_stage2_sft.sh
├── run_stage3_dpo.sh
├── run_stage4_rlvr.sh
├── run_pipeline_qwen3.sh        # all 4 stages + data prep
└── run_pipeline_qwen35.sh
```

## Quick start — one-shot pipeline

```bash
# Qwen3-0.6B full 4-stage run (handles data prep between stages)
bash scripts/train/olmo3_style/run_pipeline_qwen3.sh

# Qwen3.5-0.8B twin
bash scripts/train/olmo3_style/run_pipeline_qwen35.sh
```

## Per-stage manual run

Each stage has its own launcher — picks up the previous stage's output
directory automatically via the YAML `model_name_or_path`:

```bash
MODEL=qwen3 bash scripts/train/olmo3_style/run_stage1_midtrain.sh
MODEL=qwen3 bash scripts/train/olmo3_style/run_stage2_sft.sh

# DPO needs preference pairs first
python scripts/preprocess/olmo3_style/build_preference_pairs.py \
  --model-path outputs/olmo3_style/qwen3-0.6b/stage2_sft \
  --test data/processed/splits/train.jsonl \
  --output data/processed/olmo3_style/preference_pairs.jsonl \
  --k 4 --limit 2000
MODEL=qwen3 bash scripts/train/olmo3_style/run_stage3_dpo.sh

# RLVR needs the prompt file
python scripts/preprocess/olmo3_style/build_rlvr_prompts.py \
  --input data/processed/splits/train.jsonl \
  --output data/processed/olmo3_style/rlvr_prompts.jsonl
MODEL=qwen3 bash scripts/train/olmo3_style/run_stage4_rlvr.sh
```

## Stage details

### Stage 1 — mid-training

Goal: bias the base model toward IE distribution before instruction
tuning. Uses `stage: pt` (pretrain / LM loss on raw packed text, no
SFT mask), LR 5e-6, 1 epoch, cosine decay. Packed text is produced by
`build_midtrain_mixture.py` which flattens canonical records into a
schema-then-input-then-output block.

Output: `outputs/olmo3_style/<tag>/stage1_midtrain/`.

### Stage 2 — SFT

Standard full SFT but warm-started from the stage-1 checkpoint and with
slightly higher LR (2e-5 vs 1e-5 for vanilla) since the base has already
seen IE distribution. Same dataset as `configs/sft/qwen3_full_sft_ds.yaml`.

Output: `outputs/olmo3_style/<tag>/stage2_sft/`.

### Stage 3 — DPO

`build_preference_pairs.py` samples K=4 completions per prompt from the
stage-2 model, scores each with `ie_metrics`, and keeps (chosen,rejected)
pairs where `margin = r_best - r_worst >= 0.15`. LLaMA-Factory runs
native DPO (`stage: dpo`, `pref_loss: sigmoid`, `pref_beta: 0.1`) using
the stage-2 checkpoint as both policy and reference.

Output: `outputs/olmo3_style/<tag>/stage3_dpo/`.

### Stage 4 — RLVR (GRPO-lite)

`rlvr_trainer.py` is a ~250-line reference GRPO implementation:

1. Sample K completions per prompt from policy π_θ.
2. Reward = `ie_metrics.evaluate([comp], [gold], [task_type]).<task>.f1`.
3. Advantages = (r − group_mean) / (group_std + ε).
4. PPO-style clip objective + KL(π_θ ‖ π_ref).

For serious runs swap this for `trl.GRPOTrainer` or verl — the RLVR
YAML schema is intentionally a superset of `trl.GRPOConfig` common
fields.

Output: `outputs/olmo3_style/<tag>/stage4_rlvr/final/`.

## Evaluation

Every stage output is scoreable by the existing eval matrix. Point
`--variant full` and `--model-path` at the stage directory:

```bash
# Eval the final RLVR checkpoint
python scripts/eval/evaluate_end_to_end.py \
  --scenario qwen3-0.6b-rlvr-unified \
  --test data/processed/splits/test.jsonl \
  --model-path outputs/olmo3_style/qwen3-0.6b/stage4_rlvr/final \
  --prompt-mode unified \
  --output-dir outputs/eval/qwen3-0.6b-rlvr-unified
```

Or add a 5th variant tag to `run_eval_scenario.sh` if you want full
matrix coverage of the RLVR stage.

## Caveats

- The RLVR trainer in this repo is a **reference implementation** meant
  to show the loop end-to-end. For production scale (>1B params, long
  rollouts, mixed rewards) use `trl.GRPOTrainer` or verl.
- Preference pair generation requires the stage-2 checkpoint — stage 3
  cannot run without completing stage 2 first.
- Stage 1 with `packing: true` and `cutoff_len: 4096` may need ≥40 GB
  per GPU at batch size 2. Drop `cutoff_len` to 2048 if OOM.
- `deepspeed_z3.json` is used for stage 3 DPO because DPO holds a
  reference model in memory too — ZeRO-2 may OOM on smaller GPUs.
