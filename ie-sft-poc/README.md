# IE-SFT-PoC: Information Extraction SFT Research Repository

An internal proof-of-concept repository for fine-tuning large language models on information extraction (IE) tasks using supervised fine-tuning (SFT). This repository implements a canonical schema supporting key-value (KV), entity, and relation extraction, with LLaMA-Factory as the training backend.

## Overview

**IE-SFT-PoC** is a research-focused repository designed to explore how to effectively teach language models to perform information extraction through instruction-tuning. The project uses:

- **Canonical Schema**: A unified, task-agnostic data format supporting KV, entity, and relation extraction in a single record structure
- **Primary Models**: Qwen3-0.6B and Qwen3.5-0.8B (lightweight, research-friendly models)
- **Training Backend**: LLaMA-Factory for efficient supervised fine-tuning with LoRA support
- **Design Reference**: GoLLIE framework (not a training dependency—used for schema inspiration only)
- **Default Dataset**: InstructIE (public, license-compliant)
- **Future Extension**: OLMo3 (currently in PoC stage)

### Why a Canonical Schema?

Traditional IE datasets and models are fragmented across different task definitions and formats. This project unifies three core IE tasks into a **single canonical format**:

- **Key-Value Extraction**: Structured field-value pairs (e.g., product specifications, document metadata)
- **Entity Extraction**: Named entity recognition with span-level annotations (e.g., person, organization, location)
- **Relation Extraction**: Structured relations between entities (e.g., "founded_by", "located_in")

This unified format enables:
1. **Single model training** on heterogeneous IE tasks
2. **Flexible evaluation** across multiple task types
3. **Easy composition** of multi-task IE datasets
4. **Consistent preprocessing** and validation pipelines

## Repository Structure

```
ie-sft-poc/
├── README.md                           # This file
├── .env.example                        # Environment configuration template
│
├── configs/                            # Configuration files
│   ├── models/                         # Model definitions
│   │   ├── qwen3_0_6b.yaml
│   │   ├── qwen3_5_0_8b.yaml
│   │   └── olmo3_poc.yaml
│   ├── datasets/                       # Dataset metadata configs
│   │   ├── instructie.yaml
│   │   ├── unified_ie_example.yaml
│   │   └── internal_kv_example.yaml
│   └── sft/                            # LLaMA-Factory training configs
│       ├── qwen3_lora_sft.yaml         # LoRA (single-GPU friendly)
│       ├── qwen3_5_lora_sft.yaml
│       ├── qwen3_full_sft_ds.yaml      # Full SFT + DeepSpeed ZeRO-2
│       ├── qwen3_5_full_sft_ds.yaml
│       ├── deepspeed_z2.json           # ZeRO-2 base
│       ├── deepspeed_z3.json           # ZeRO-3 (for OOM at Z2)
│       ├── deepspeed_z2_h100.json      # Hopper-tuned ZeRO-2
│       ├── h100_overrides.yaml         # H100 knobs (fa2, tf32, FP8 opt-in)
│       └── olmo3_poc_sft.yaml
│
├── data/                               # Data directory
│   ├── raw/                            # Downloaded, unprocessed datasets
│   │   ├── instructie/                 # InstructIE dataset (public)
│   │   ├── gollie_reference/           # GoLLIE reference (schema only)
│   │   └── .gitignore                  # Raw data excluded from git
│   ├── interim/                        # Normalized canonical format per dataset
│   │   ├── instructie/
│   │   ├── internal_kv/
│   │   └── gollie_reference/
│   ├── processed/                      # Merged, split, ready for training
│   │   ├── unified.jsonl               # Merged dataset
│   │   ├── splits/
│   │   │   ├── train.jsonl
│   │   │   ├── dev.jsonl
│   │   │   └── test.jsonl
│   │   └── llamafactory/               # Exported for LLaMA-Factory
│   │       ├── train.jsonl
│   │       └── dev.jsonl
│   └── metadata/                       # Dataset registry and info
│       ├── dataset_registry.yaml       # Central dataset metadata
│       └── dataset_registry.example.yaml
│
├── docs/                               # Documentation
│   ├── README.md                       # Index of all documentation
│   ├── architecture.md                 # System design and data flow
│   ├── canonical_schema.md             # Full schema specification
│   ├── dataset_policy.md               # Dataset licensing and policy
│   ├── training_flow.md                # Training pipeline walkthrough
│   └── olmo3_extension.md              # OLMo3 PoC status and roadmap
│
├── examples/                           # Example data and prompts
│   ├── canonical_samples/              # Sample records in canonical format
│   │   ├── kv_sample.json
│   │   ├── entity_relation_sample.json
│   │   └── unified_sample.json
│   └── prompts/                        # Example prompts for IE tasks
│
├── scripts/                            # Utility and workflow scripts
│   ├── download/                       # Dataset download scripts
│   │   ├── download_instructie.py      # Download InstructIE from HF
│   │   ├── download_reference_gollie_assets.py
│   │   └── download_internal_template.py
│   ├── preprocess/                     # Data normalization and merging
│   │   ├── normalize_instructie.py     # Convert InstructIE → canonical
│   │   ├── unify_ie_datasets.py        # Merge multiple datasets
│   │   ├── validate_canonical_dataset.py
│   │   └── build_internal_kv_template.py
│   ├── export/                         # Export to training format
│   │   ├── export_train_dev_test.py    # Create splits
│   │   └── export_to_llamafactory.py   # Export for LLaMA-Factory
│   ├── train/                          # Training scripts
│   │   ├── run_sft_qwen3.sh            # Qwen3-0.6B LoRA (single-GPU)
│   │   ├── run_sft_qwen35.sh           # Qwen3.5-0.8B LoRA (single-GPU)
│   │   ├── run_sft_qwen3_multigpu.sh   # torchrun + DeepSpeed launcher
│   │   ├── run_sft_qwen35_multigpu.sh
│   │   ├── run_sft_olmo3_poc.sh        # OLMo3 training (PoC)
│   │   ├── run_eval_qwen3.sh           # (legacy LLaMA-Factory eval)
│   │   ├── run_infer_qwen3.sh          # Inference/generation
│   │   └── merge_lora_qwen3.sh         # Merge LoRA weights into base
│   ├── eval/                           # IE-specific eval matrix
│   │   ├── run_predict.py              # Batched gen (flash-attn2 + LoRA)
│   │   ├── compute_metrics.py          # KV/entity/relation F1
│   │   ├── evaluate_end_to_end.py      # predict + score orchestrator
│   │   ├── run_eval_scenario.sh        # Dispatcher (--model/--variant/--mode)
│   │   ├── run_eval_qwen3_lora.sh      # Per-scenario wrappers
│   │   ├── run_eval_qwen3_full.sh
│   │   ├── run_eval_qwen35_lora.sh
│   │   ├── run_eval_qwen35_full.sh
│   │   └── run_eval_all.sh             # Full matrix sweep
│   └── utils/                          # Utility functions
│       ├── env_check.py                # Verify environment setup
│       └── print_dataset_stats.py      # Dataset statistics
│
├── src/                                # Python source code
│   ├── __init__.py
│   ├── datasets/                       # Dataset handling
│   │   ├── metadata.py                 # Dataset metadata classes
│   │   ├── registry.py                 # Dataset registry loader
│   │   ├── unified/                    # Canonical format utilities
│   │   │   ├── __init__.py
│   │   │   ├── validator.py            # Schema validation
│   │   │   ├── splitter.py             # Train/dev/test splitting
│   │   │   ├── merger.py               # Dataset merging
│   │   │   └── stats.py                # Dataset statistics
│   │   └── gollie_reference/           # GoLLIE schema reference (read-only)
│   │       ├── __init__.py
│   │       ├── schema_patterns.py
│   │       └── task_reference.py
│   ├── models/                         # Model adapters
│   │   ├── model_registry.py
│   │   ├── qwen.py                     # Qwen3 adapter
│   │   └── olmo.py                     # OLMo3 adapter (basic)
│   ├── training/                       # Training orchestration
│   │   ├── config_builder.py           # Build LLaMA-Factory configs
│   │   ├── dataset_registry_builder.py # Build dataset registries
│   │   ├── llamafactory_runner.py      # Run LLaMA-Factory trainer
│   │   ├── eval_runner.py              # Run evaluation
│   │   ├── inference_runner.py         # Run inference
│   │   └── ie_metrics.py               # IE metrics (KV/entity/relation F1)
│   ├── olmo3_poc/                      # OLMo3 proof-of-concept
│   │   ├── __init__.py
│   │   ├── adapter.py                  # OLMo3 adapter framework
│   │   ├── conversion.py               # Format conversion utilities
│   │   └── notes.py                    # Status and TODO tracking
│   └── common/                         # Shared utilities (if exists)
│       └── ...
│
└── tests/                              # Unit and integration tests
    └── (test files)
```

## Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to repository
cd ie-sft-poc

# Copy environment template
cp .env.example .env

# Edit .env with your settings (GPU, paths, HF token, etc.)
# At minimum, set:
# - CUDA_VISIBLE_DEVICES (GPU IDs or empty for CPU)
# - HF_TOKEN (HuggingFace API token if downloading models)
nano .env
```

### 2. Verify Environment

```bash
# Check environment and dependencies
python scripts/utils/env_check.py
```

### 3. Download Data

```bash
# Download InstructIE dataset (default, public, license-compliant)
python scripts/download/download_instructie.py

# Optionally: Download GoLLIE reference assets (schema only, not for training)
python scripts/download/download_reference_gollie_assets.py
```

### 4. Preprocess Datasets

```bash
# Normalize InstructIE to canonical format
python scripts/preprocess/normalize_instructie.py

# Merge multiple datasets (if multiple sources available)
python scripts/preprocess/unify_ie_datasets.py \
  --input data/interim/instructie/*.jsonl \
  --output data/processed/unified.jsonl

# Validate the unified dataset
python scripts/preprocess/validate_canonical_dataset.py \
  --input data/processed/unified.jsonl
```

### 5. Create Train/Dev/Test Splits

```bash
# Export splits for training
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/splits/ \
  --train-ratio 0.8 \
  --dev-ratio 0.1
```

### 6. Export to LLaMA-Factory Format

```bash
# Convert splits to LLaMA-Factory format
python scripts/export/export_to_llamafactory.py \
  --input data/processed/splits/ \
  --output data/processed/llamafactory/
```

### 7. Run Training

Single-GPU LoRA (default recipe):

```bash
bash scripts/train/run_sft_qwen3.sh     # Qwen3-0.6B   + LoRA
bash scripts/train/run_sft_qwen35.sh    # Qwen3.5-0.8B + LoRA
bash scripts/train/run_sft_olmo3_poc.sh # OLMo3 PoC (when models available)
```

Multi-GPU full SFT via DeepSpeed (ZeRO-2 by default):

```bash
# Auto-detects NPROC from nvidia-smi.
bash scripts/train/run_sft_qwen3_multigpu.sh
bash scripts/train/run_sft_qwen35_multigpu.sh

# Override: pick a specific config (LoRA + DeepSpeed works too).
CONFIG=configs/sft/qwen3_full_sft_ds.yaml NPROC=4 \
  bash scripts/train/run_sft_qwen3_multigpu.sh
```

H100 / Hopper: every SFT YAML ships `flash_attn: fa2`. For Hopper-tuned
communication and bigger microbatches, layer `configs/sft/h100_overrides.yaml`
on top of a base config and switch `deepspeed:` to `deepspeed_z2_h100.json`.
FP8 via Transformer Engine is opt-in — see comments in `h100_overrides.yaml`.

LoRA vs Full SFT quick reference:

| | LoRA (`*_lora_sft.yaml`) | Full SFT (`*_full_sft_ds.yaml`) |
|---|---|---|
| Trainable params | ~0.5–1% (rank-16 adapter) | 100% |
| GPU memory | fits on 1x 24–40GB | needs 2–8x 80GB w/ DeepSpeed |
| LR default | `2e-4` | `1e-5` |
| Checkpoint size | adapter only | full model |
| Eval `--variant` | `lora` | `full` |

### 8. Run Evaluation

IE evaluation uses `src/training/ie_metrics.py` to compute KV field PRF1,
entity (type, text) set F1, and relation triple F1 — not LLaMA-Factory's
loss/MC accuracy.

```bash
# One scenario (model x variant x mode)
bash scripts/eval/run_eval_qwen3_lora.sh                     # unified mode
MODE=kv       bash scripts/eval/run_eval_qwen3_lora.sh
MODE=entity   bash scripts/eval/run_eval_qwen35_full.sh
MODE=relation bash scripts/eval/run_eval_qwen35_lora.sh

# Full matrix: {qwen3, qwen3.5} x {lora, full} x {kv, entity, relation, unified}
# Missing checkpoints are skipped automatically.
bash scripts/eval/run_eval_all.sh

# Smoke test (first 50 records)
LIMIT=50 bash scripts/eval/run_eval_qwen3_lora.sh
```

Outputs land in `outputs/eval/<tag>-<variant>-<mode>/{test_predictions.jsonl, metrics.json}`.
See `scripts/eval/README.md` for the full env-var reference.

### 9. Run Inference

```bash
# Generate predictions with the model
bash scripts/train/run_infer_qwen3.sh
```

### 10. Merge LoRA Weights (Optional)

```bash
# Merge LoRA weights into the base model for deployment
bash scripts/train/merge_lora_qwen3.sh
```

## Dataset Policy

### Default-Enabled Datasets

**InstructIE** (default): Public, license-compliant dataset available from HuggingFace. Safe to download and use for training. No additional permissions needed.

```bash
python scripts/download/download_instructie.py
```

### Reference-Only Datasets

**GoLLIE** (reference only): The GoLLIE framework is used as a design reference for the canonical schema but is NOT a training data dependency. Reference assets may be downloaded for schema comparison, but GoLLIE data itself is not included in training by default.

```bash
# Reference only, not for training
python scripts/download/download_reference_gollie_assets.py
```

### Optional/Restricted Datasets

**IEPile**: NOT enabled by default. Opt-in only with explicit configuration. Check dataset license before enabling.

**Internal Data**: Separate onboarding and access procedures apply. Contact internal data administrators for setup.

See `docs/dataset_policy.md` for complete licensing information and how to add new datasets.

## Key Features

### Canonical Schema Support
- **Unified Format**: Single JSONL format supports KV, entity, and relation extraction
- **Task Composition**: Train on one or all three task types simultaneously
- **Schema-Conditioned Extraction**: Models learn task definitions and constraints

### Model Support
- **Qwen3-0.6B**: Lightweight, efficient for development and research
- **Qwen3.5-0.8B**: Slightly larger variant for improved quality
- **OLMo3 (PoC)**: Future extension currently in proof-of-concept stage

### Training Infrastructure
- **LLaMA-Factory Backend**: Efficient SFT with LoRA and full fine-tuning
- **DeepSpeed ZeRO-2 / ZeRO-3**: Multi-GPU full SFT out of the box
- **FlashAttention-2**: Default on all configs (A100/H100/L40S)
- **H100 (Hopper)**: tuned DeepSpeed profile, tf32, torch.compile, FP8 opt-in
- **Mixed-Precision**: bf16 on Hopper/Ampere, fp16 fallback for Volta
- **Experiment Tracking**: TensorBoard by default, W&B optional

### Evaluation
- **IE-specific metrics**: KV field PRF1, entity set F1, relation triple F1
- **Scenario matrix**: {qwen3, qwen3.5} × {lora, full} × {kv, entity, relation, unified}
- **Robust JSON parsing**: code-fence / brace-block fallback with parse-failure tracking

### OLMo3-style 4-stage pipeline (applied to Qwen3 / Qwen3.5)
Not OLMo3 the model — the OLMo3 *training recipe*: mid-training → SFT → DPO → RLVR (GRPO-lite with verifiable IE-F1 rewards). One-shot runners in `scripts/train/olmo3_style/`. See [docs/olmo3_style_pipeline.md](docs/olmo3_style_pipeline.md).

```bash
bash scripts/train/olmo3_style/run_pipeline_qwen3.sh     # 4 stages + data prep
bash scripts/train/olmo3_style/run_pipeline_qwen35.sh
```

### Data Pipeline
- **Download**: Automated dataset acquisition from HuggingFace
- **Normalize**: Convert dataset-specific formats to canonical schema
- **Validate**: Schema validation and consistency checking
- **Merge**: Combine multiple datasets into unified training sets
- **Split**: Create train/dev/test splits with configurable ratios
- **Export**: Format conversion for LLaMA-Factory consumption

## Configuration

All major components can be configured via:

1. **Environment Variables** (`.env` file)
2. **YAML Configuration Files** (in `configs/`)
3. **CLI Arguments** (when available)

Key configuration areas:
- **Model Selection**: `configs/models/*.yaml`
- **Dataset Metadata**: `configs/datasets/*.yaml`
- **Training Parameters**: `configs/sft/*.yaml`

See individual config files for detailed parameter explanations.

## Contributing and Development

### Adding a New Dataset

1. Create dataset normalization script in `scripts/preprocess/`
2. Add dataset config in `configs/datasets/`
3. Document in `data/metadata/dataset_registry.yaml`
4. Update `docs/dataset_policy.md` if licensing applies
5. Add integration test in `tests/`

### Adding a New Model Family

1. Create model adapter in `src/models/`
2. Add model config in `configs/models/`
3. Create training script in `scripts/train/`
4. Add to model registry in `src/models/model_registry.py`
5. Document in `docs/training_flow.md`

### Development Best Practices

- Always validate datasets after preprocessing
- Test on small dataset splits before full training
- Document any new task types in the canonical schema
- Add unit tests for new utility functions
- Update this README and relevant docs as you extend the system

## OLMo3 Extension

OLMo3 support is currently in **proof-of-concept (PoC)** stage. The adapter framework is implemented, but actual training awaits the public OLMo3 model release. See `docs/olmo3_extension.md` for detailed status, known differences, and the roadmap for full integration.

## Documentation Index

- **[architecture.md](docs/architecture.md)**: System design, module responsibilities, data flow
- **[canonical_schema.md](docs/canonical_schema.md)**: Full schema specification, validation rules, examples
- **[dataset_policy.md](docs/dataset_policy.md)**: Dataset licensing, opt-in procedures, compliance
- **[training_flow.md](docs/training_flow.md)**: End-to-end training walkthrough, hyperparameter tuning
- **[olmo3_extension.md](docs/olmo3_extension.md)**: OLMo3 *model* support status
- **[olmo3_style_pipeline.md](docs/olmo3_style_pipeline.md)**: OLMo3-style 4-stage *training recipe* (mid-train → SFT → DPO → RLVR) applied to Qwen3/3.5

## Troubleshooting

### GPU Memory Issues
- Reduce `TRAINING_BATCH_SIZE` in `.env`
- Increase `TRAINING_GRADIENT_ACCUMULATION` for effective batch size
- Enable `TRAINING_BF16` for mixed-precision training

### Dataset Validation Errors
- Run `python scripts/preprocess/validate_canonical_dataset.py` to check specific issues
- See `docs/canonical_schema.md` for schema validation rules
- Check if text spans in entity/relation annotations match actual text

### HuggingFace Download Issues
- Ensure `HF_TOKEN` is set in `.env` and has required permissions
- Try `HF_HUB_DISABLE_TELEMETRY=1` if experiencing connection issues
- Clear cache with `rm -rf ~/.cache/huggingface` if stuck

### Model Loading Errors
- Ensure model is available on HuggingFace Hub or locally cached
- Check `CUDA_VISIBLE_DEVICES` if CUDA/GPU errors occur
- For OLMo3, verify `OLMO3_TRUST_REMOTE_CODE=true` in `.env`

## References

- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **InstructIE Dataset**: https://huggingface.co/datasets/...
- **Qwen Models**: https://huggingface.co/Qwen
- **GoLLIE Framework**: https://arxiv.org/abs/2310.03144 (schema inspiration)
- **OLMo3**: https://allenai.org/ollm (when available)

## License

This repository is for internal research use. See individual dataset licenses in `docs/dataset_policy.md`.

## Contact

For questions, issues, or contributions, reach out to the internal research team.

---

**Last Updated**: April 2026  
**Repository Status**: Active Research  
**Primary Contact**: Research Team
