# Training Flow and Workflow

Complete end-to-end training pipeline documentation, including data preparation, training configuration, execution, evaluation, and deployment.

## Prerequisites

### System Requirements

**GPU**: Recommended but not required
- NVIDIA GPU with 16GB+ VRAM (for batch size 4)
- Recent CUDA version (11.8+) and cuDNN

**CPU-only**: Possible but slow
- Training will be significantly slower
- Set `CUDA_VISIBLE_DEVICES=""` in `.env`

**Storage**: ~50-100 GB
- Raw data: ~10-20 GB
- Interim data: ~20 GB
- Processed data: ~5 GB
- Checkpoints: ~5-10 GB

**Memory**: 32GB+ system RAM recommended

### Software Requirements

```bash
# Python 3.10+
python --version

# Required packages (install via pip)
pip install -r requirements.txt  # When available

# Key dependencies:
pip install transformers torch peft datasets bitsandbytes wandb llamafactory
```

### Configuration

Copy and customize `.env`:

```bash
cp .env.example .env
# Edit .env with your settings:
# - CUDA_VISIBLE_DEVICES (GPU selection)
# - HF_TOKEN (if using private models)
# - Model and training hyperparameters
```

Verify setup:

```bash
python scripts/utils/env_check.py
```

## Data Preparation Pipeline

### Step 1: Download Datasets

Download raw datasets from external sources:

```bash
# Download InstructIE (primary, public dataset)
python scripts/download/download_instructie.py

# Optional: Download GoLLIE reference (design reference only)
python scripts/download/download_reference_gollie_assets.py

# Optional: Download internal templates (if available)
python scripts/download/download_internal_template.py
```

**Output**: `data/raw/{dataset_name}/*.jsonl`

**Time**: 5-30 minutes depending on dataset size and connection

### Step 2: Normalize to Canonical Format

Convert each dataset to canonical IE schema:

```bash
# Normalize InstructIE
python scripts/preprocess/normalize_instructie.py \
  --input-dir data/raw/instructie/ \
  --output-dir data/interim/instructie/ \
  --validate

# Expected output:
# data/interim/instructie/train.jsonl
# data/interim/instructie/validation.jsonl
# data/interim/instructie/test.jsonl
```

**Checks Performed**:
- Schema validation (JSON structure)
- Required fields (id, text, task_types, answer)
- Task type consistency
- Entity span validation
- Relation field validity

**Time**: 5-10 minutes per dataset

### Step 3: Validate Normalized Data

Ensure data quality after normalization:

```bash
python scripts/preprocess/validate_canonical_dataset.py \
  --input data/interim/instructie/train.jsonl \
  --strict \
  --report-file validation_report.json
```

**Output**: Validation report with:
- Total records validated
- Valid/invalid counts
- Errors and warnings
- Issues per record (if any)

**Sample Output**:
```
Validation Status: PASS
File: data/interim/instructie/train.jsonl
Total records: 70000
Valid: 69998
Invalid: 2
Warnings: 15
```

**Fix Errors**:
1. Check `validation_report.json` for problematic records
2. Identify issue (bad span, wrong type, etc.)
3. Either fix source or skip records
4. Re-validate

### Step 4: Merge Multiple Datasets

Combine multiple normalized datasets into single unified file:

```bash
# Merge all interim datasets
python scripts/preprocess/unify_ie_datasets.py \
  --input data/interim/*/*.jsonl \
  --output data/processed/unified.jsonl \
  --validate

# Optional: Downsample for quick testing
python scripts/preprocess/unify_ie_datasets.py \
  --input data/interim/*/*.jsonl \
  --output data/processed/unified_sample.jsonl \
  --sample-ratio 0.1  # Use only 10%
```

**Merging Strategy**:
- Preserves metadata (dataset source, license, split)
- Deduplicates by record ID
- Validates output
- Generates merge report

**Time**: 5-15 minutes

### Step 5: Create Train/Dev/Test Splits

Split unified dataset into training, development, and test sets:

```bash
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/splits/ \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

**Output**:
- `data/processed/splits/train.jsonl` (80%)
- `data/processed/splits/dev.jsonl` (10%)
- `data/processed/splits/test.jsonl` (10%)
- `data/processed/splits/split_info.json` (metadata)

**Optional**: Stratified splitting by task type:

```bash
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/splits/ \
  --stratify task_types  # Ensure balanced distribution
```

### Step 6: Export to LLaMA-Factory Format

Convert canonical format to LLaMA-Factory conversation format:

```bash
python scripts/export/export_to_llamafactory.py \
  --input data/processed/splits/ \
  --output data/processed/llamafactory/ \
  --template detailed
```

**Output**:
- `data/processed/llamafactory/train.jsonl` (conversation format)
- `data/processed/llamafactory/dev.jsonl` (conversation format)
- `data/processed/llamafactory/export_info.json`

**Format**:
```json
{
  "messages": [
    {"role": "system", "content": "You are an information extraction expert..."},
    {"role": "user", "content": "Extract entities from: [TEXT]\nSchema: [SCHEMA]"},
    {"role": "assistant", "content": "{\"entity\": [...]}"}
  ]
}
```

**Time**: 5-10 minutes

## Training Configuration

### Configuration Files

Training is configured via YAML files in `configs/sft/`:

**Key Parameters**:
- `model_name_or_path`: HuggingFace model ID
- `train_file`: Path to training data
- `eval_file`: Path to validation data
- `per_device_train_batch_size`: Batch size per GPU
- `learning_rate`: Learning rate for optimizer
- `num_train_epochs`: Number of training epochs
- `lora_rank`: LoRA adapter rank
- `output_dir`: Checkpoint output directory

### Example Configuration (qwen3_lora_sft.yaml)

```yaml
# Model Configuration
model_name_or_path: Qwen/Qwen3-0.6B
template: qwen
cutoff_len: 2048

# Data Configuration
train_file: data/processed/llamafactory/train.jsonl
eval_file: data/processed/llamafactory/dev.jsonl

# Training Hyperparameters
output_dir: ./outputs/qwen3-sft
per_device_train_batch_size: 4
per_device_eval_batch_size: 4
learning_rate: 5e-5
num_train_epochs: 3
warmup_steps: 500
max_grad_norm: 1.0
weight_decay: 0.01

# Optimization
optim: paged_adamw_32bit
gradient_accumulation_steps: 2
bf16: true

# LoRA Configuration
use_lora: true
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

# Evaluation & Logging
evaluation_strategy: steps
eval_steps: 100
save_strategy: steps
save_steps: 100
logging_steps: 10
report_to: [tensorboard]
save_total_limit: 3

# Advanced
seed: 42
gradient_checkpointing: true
```

### Hyperparameter Tuning

#### Learning Rate

**Default**: 5e-5 (for LoRA fine-tuning)

**Range to test**: 1e-5 to 1e-3

**Tuning strategy**:
1. Start with default 5e-5
2. If loss doesn't decrease, try 1e-4
3. If loss is unstable, try 1e-5
4. Monitor validation F1 for best value

```bash
# Try different learning rate
learning_rate: 1e-4  # Edit in config
```

#### Batch Size

**Default**: 4 per device (memory efficient)

**Considerations**:
- Larger batch = less noise, better generalization
- Smaller batch = faster updates, noisier gradients
- Limited by GPU memory

**Memory-Batch Size Relationship**:
- 16GB GPU: batch size 4-8 (with gradient accumulation)
- 32GB GPU: batch size 8-16
- 80GB GPU: batch size 32+

**Recommendation**:
- Start with default 4
- Increase if GPU has free memory
- Use gradient accumulation for larger effective batches

```bash
# Larger batch (if you have 32GB+ GPU)
per_device_train_batch_size: 8
gradient_accumulation_steps: 1  # Total: 8
```

#### Number of Epochs

**Default**: 3 epochs

**Considerations**:
- More epochs = risk of overfitting (but dataset is large)
- Fewer epochs = underfitting
- Validation loss should level off

**Recommendation**:
- Start with 3 for initial training
- Increase to 5 if validation loss still decreasing
- Decrease to 1 for quick testing

#### LoRA Rank

**Default**: 8 (good balance)

**Trade-offs**:
- Rank 8: 99% fewer parameters, good for quick experiments
- Rank 16: Better quality, slightly slower
- Rank 32: Even better quality, more memory

**Recommendation**:
- Use 8 for development/testing
- Use 16 for production training

```bash
# Higher quality (slower, more memory)
lora_rank: 16
lora_alpha: 32  # Typically 2x rank
```

#### Warmup

**Default**: 500 warmup steps

**Rationale**: Gradually increase learning rate from 0

**Recommendation**: 10% of total training steps

```python
# Calculate for your setup:
total_steps = (num_examples * num_epochs) / (batch_size * grad_accumulation)
warmup_steps = int(total_steps * 0.1)
# For 56k training examples, 3 epochs, batch 4, accumulation 2:
# total_steps = (56000 * 3) / (4 * 2) = 21,000
# warmup_steps = 2,100
```

## Training Execution

### Basic Training Command

```bash
# Train Qwen3-0.6B with default config
bash scripts/train/run_sft_qwen3.sh

# With custom config
bash scripts/train/run_sft_qwen3.sh --config configs/sft/custom_qwen3.yaml

# Dry run (show command without training)
bash scripts/train/run_sft_qwen3.sh --dry-run
```

### Multi-GPU Training (DeepSpeed)

For full SFT across multiple GPUs use the DeepSpeed-aware launchers. They
auto-detect NPROC via `nvidia-smi` and wrap `llamafactory-cli` under
`torchrun`. The base configs use ZeRO-2; switch `deepspeed:` to
`deepspeed_z3.json` only if ZeRO-2 OOMs at your chosen batch size.

```bash
# Qwen3-0.6B full SFT (2-8 GPU recommended)
bash scripts/train/run_sft_qwen3_multigpu.sh

# Qwen3.5-0.8B full SFT
bash scripts/train/run_sft_qwen35_multigpu.sh

# Override config / NPROC
CONFIG=configs/sft/qwen3_full_sft_ds.yaml NPROC=4 \
  bash scripts/train/run_sft_qwen3_multigpu.sh

# Or run LoRA + DDP on a subset of GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_sft_qwen3.sh
```

Effective batch size = `per_device_train_batch_size * gradient_accumulation_steps * N_GPUS`.

### H100 / Hopper

Every SFT YAML ships `flash_attn: fa2` — FlashAttention-2 works on both
Ampere and Hopper. For Hopper-specific tuning apply
`configs/sft/h100_overrides.yaml` on top of a base config (tf32,
`torch.compile`, bigger microbatch) and set:

```yaml
deepspeed: configs/sft/deepspeed_z2_h100.json   # Hopper-tuned buckets + RR grads
```

FP8 via NVIDIA Transformer Engine is opt-in — uncomment the `fp8: true`
block in `h100_overrides.yaml` only if TE is installed and your model has
TE layer replacements.

See `configs/sft/h100_overrides.yaml` for the full list of knobs.

### Monitoring Training

#### TensorBoard

```bash
# Start TensorBoard in separate terminal
tensorboard --logdir outputs/qwen3-sft

# Open http://localhost:6006 in browser
```

#### Weights & Biases (Optional)

```bash
# Enable W&B in config
report_to: [wandb]
wandb_project: ie-sft-poc

# Set token in .env
export WANDB_API_KEY=<your_key>

# Run training
bash scripts/train/run_sft_qwen3.sh
```

### Common Issues During Training

#### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size:
   ```yaml
   per_device_train_batch_size: 2
   gradient_accumulation_steps: 4  # Keep effective size
   ```

2. Enable gradient checkpointing:
   ```yaml
   gradient_checkpointing: true
   ```

3. Use lower precision:
   ```yaml
   fp16: true  # More aggressive than bf16
   ```

#### Loss Not Decreasing

**Possible causes**:
1. Learning rate too high: Try 1e-5
2. Learning rate too low: Try 1e-4
3. Bad data: Validate with `validate_canonical_dataset.py`
4. Model too small: Acceptable with 0.6B model

**Debugging**:
```bash
# Train on small subset first
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/test_splits/ \
  --train-ratio 0.01  # 1% for quick test

# Then run training with this small dataset
```

#### Training Hangs or Crashes

**Check**:
1. GPU drivers: `nvidia-smi`
2. CUDA availability: `torch.cuda.is_available()`
3. Data format: `validate_canonical_dataset.py`
4. Disk space: Check `/tmp` and output directory

**Recover from checkpoint**:
```bash
# Training auto-saves checkpoints
# Resume from latest: edit config
output_dir: ./outputs/qwen3-sft
# Delete or rename incomplete checkpoint
# Re-run training (it will find existing checkpoint)
```

## Evaluation Methodology

### Evaluation on Dev Set

Evaluation runs automatically during training at specified intervals:

```yaml
evaluation_strategy: steps  # or "epoch"
eval_steps: 100            # Evaluate every 100 steps
```

### Manual Evaluation

For IE-specific scoring (KV PRF1 / entity set F1 / relation triple F1) use
the scenario matrix under `scripts/eval/`:

```bash
# Single scenario: {model} x {variant} x {mode}
bash scripts/eval/run_eval_qwen3_lora.sh                   # mode=unified
MODE=kv       bash scripts/eval/run_eval_qwen3_lora.sh
MODE=entity   bash scripts/eval/run_eval_qwen35_full.sh
MODE=relation bash scripts/eval/run_eval_qwen35_lora.sh

# Full matrix (2 models x 2 variants x 4 modes; skips missing ckpts)
bash scripts/eval/run_eval_all.sh
```

Each scenario writes to `outputs/eval/<tag>-<variant>-<mode>/`:

- `test_predictions.jsonl` — `{id, prompt, prediction, gold}` per row
- `metrics.json` — per-task PRF1 plus parse-failure counts

Env knobs: `TEST_FILE`, `BATCH_SIZE`, `MAX_NEW`, `LIMIT`, `DTYPE`
(`bf16` default; use `fp16` on Volta). See `scripts/eval/README.md`.

### Metrics

Implemented in `src/training/ie_metrics.py`.

**KV Extraction** — `score_kv`:
- PRF1 over `(record_idx, field_name)` pairs
- Exact-match on normalized (lowercase + whitespace-collapsed) value

**Entity Extraction** — `score_entities`:
- Multiset F1 over `(entity_type, text)` tuples
- Duplicates count; wrong type = miss

**Relation Extraction** — `score_relations`:
- Triple F1 on `(head, relation, tail)`
- `require_types=True` for the stricter typed variant (head/tail types must match too)

**Robustness** — `parse_prediction` strips ```json code fences and falls
back to the first balanced `{...}` block, so mild generation noise (extra
prose, trailing tokens) doesn't blow up scoring. Parse failures are
counted separately in `metrics.json`.

**Overall**:
- Perplexity: Model confidence on validation set
- Loss: Cross-entropy loss on validation data

### Interpretation

- **High F1 (>85%)**: Model learning well
- **Medium F1 (70-85%)**: Acceptable, tuning may help
- **Low F1 (<70%)**: Model struggling, check data quality
- **F1 increasing over epochs**: Good training progress
- **F1 plateauing**: Good validation performance achieved
- **F1 decreasing**: Overfitting detected

### Custom Evaluation

For task-specific evaluation beyond standard metrics:

```python
# src/training/eval_runner.py
# Extend eval_runner to add custom metrics

from src.datasets.unified.validator import validate_dataset
from src.training.inference_runner import run_inference

# Get model predictions
predictions = run_inference(model, test_data)

# Compare with ground truth
for pred, truth in zip(predictions, test_data):
    # Custom evaluation logic
    pass
```

## Inference and Generation

### Single-Shot Inference

```bash
# Generate predictions for a single input
bash scripts/train/run_infer_qwen3.sh \
  --model-path ./outputs/qwen3-sft-latest/ \
  --prompt "Apple was founded by Steve Jobs in California."
```

**Output**:
```json
{
  "entity": [
    {"text": "Apple", "type": "ORG", "start": 0, "end": 5},
    {"text": "Steve Jobs", "type": "PERSON", "start": 21, "end": 31},
    {"text": "California", "type": "LOC", "start": 35, "end": 45}
  ],
  "relation": [
    {
      "head": "Apple",
      "head_type": "ORG",
      "relation": "founded_by",
      "tail": "Steve Jobs",
      "tail_type": "PERSON"
    }
  ]
}
```

### Batch Inference

For processing multiple documents:

```python
from src.training.inference_runner import run_inference_batch

# Load model
model = ...

# Prepare batch
texts = [
    "Text 1 for extraction",
    "Text 2 for extraction",
    # ... more texts
]

# Run inference
predictions = run_inference_batch(model, texts)

# Save predictions
import json
with open("predictions.jsonl", "w") as f:
    for pred in predictions:
        json.dump(pred, f)
        f.write("\n")
```

### Interactive Inference

```bash
# Read prompts from stdin
bash scripts/train/run_infer_qwen3.sh \
  --model-path ./outputs/qwen3-sft-latest/ \
  --interactive

# Type text, get predictions (press Ctrl+D to exit)
```

### Inference Parameters

Configuration via command-line or code:

```python
# Temperature (0.0 = deterministic, 1.0+ = random)
temperature: 0.7

# Max generation length
max_length: 256

# Beam search
num_beams: 1  # 1 = greedy, >1 = beam search

# Top-k sampling
top_k: 50

# Nucleus (top-p) sampling
top_p: 0.95
```

**Recommendations**:
- For extraction (deterministic): temperature 0.1, greedy
- For generation (creative): temperature 0.7, top_p 0.9

## Model Deployment

### Merging LoRA Weights

LoRA weights must be merged for production deployment:

```bash
bash scripts/train/merge_lora_qwen3.sh \
  --base-model Qwen/Qwen3-0.6B \
  --lora-path ./outputs/qwen3-sft-checkpoint-500/ \
  --output-path ./outputs/merged-qwen3-final/
```

**Output**: Single merged model directory

**Advantages**:
- Single model file (no separate LoRA files)
- Standard transformers API compatible
- Deployable with standard inference engines

### Loading Merged Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./outputs/merged-qwen3-final/")
tokenizer = AutoTokenizer.from_pretrained("./outputs/merged-qwen3-final/")

# Use like standard model
```

### Loading with LoRA Weights Separate

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
model = PeftModel.from_pretrained(model, "./outputs/qwen3-sft-checkpoint-500/")

# Use merged model
model = model.merge_and_unload()
```

## Advanced Topics

### Multi-GPU Distributed Training

Enable automatic distributed training:

```yaml
# Config automatically uses all visible GPUs
# No additional config needed, set environment:
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_sft_qwen3.sh
```

**How it works**:
- LLaMA-Factory detects multiple GPUs
- Automatically sets up data parallel training
- Each GPU gets subset of batch
- Gradients synchronized across GPUs

**Performance**:
- 2 GPUs: ~1.8x speedup
- 4 GPUs: ~3.5x speedup
- 8 GPUs: ~7x speedup

### Gradient Accumulation

Simulate larger batch size without OOM:

```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
# Effective batch size: 4 * 4 = 16
```

**Trade-off**:
- Reduces memory by batch_size factor
- Increases time per step
- Effective batch size remains large

### Mixed Precision Training

Use lower precision for faster training:

```yaml
# bfloat16 (recommended for A100, H100)
bf16: true

# float16 (for older GPUs)
fp16: true
```

**Performance**:
- ~50% faster
- ~50% less memory
- Minimal quality impact with modern GPUs

### Gradient Checkpointing

Trade memory for speed:

```yaml
gradient_checkpointing: true
```

Saves intermediate activations only, recomputes as needed.

**Use when**:
- OOM errors with gradient accumulation
- Batch size still too large

## Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| OOM (Out of Memory) | Batch size too large | Reduce batch size, enable checkpointing |
| Loss not decreasing | Bad LR, bad data, bad config | Validate data, try LR 1e-4, check config |
| Training hangs | Data issue, GPU driver issue | Check disk, reinstall CUDA, validate data |
| Slow training | CPU bottleneck, bad data loading | Check DataLoader workers, SSD for data |
| Low F1 on validation | Model underfitting or data quality | Train longer, check label quality |
| Validation F1 decreasing | Overfitting | Reduce epochs, increase dropout |
| Model won't load | Checkpoint corrupted or missing | Check output dir, verify checkpoint path |

## Performance Benchmarks

**System**: NVIDIA A100, Qwen3-0.6B, 56k training examples

| Batch Size | Epochs | Time | Memory | Final F1 |
|------------|--------|------|--------|----------|
| 4 | 3 | 4h | 16GB | 82% |
| 8 | 3 | 3h | 22GB | 83% |
| 16 | 3 | 2.5h | 28GB | 84% |

**Key observations**:
- Larger batches faster but need more memory
- F1 improves with batch size (better gradient estimates)
- 3 epochs adequate for this dataset size

## Reproducibility

To ensure reproducible results:

```yaml
# Set random seed
seed: 42

# Fixed parameters
evaluation_strategy: steps
eval_steps: 100
save_steps: 100
```

```bash
# Use same environment
python --version
pip freeze > requirements_frozen.txt

# Document hardware
nvidia-smi
torch.cuda.get_device_name()
```

---

**Last Updated**: April 2026  
**Training Version**: 1.0
