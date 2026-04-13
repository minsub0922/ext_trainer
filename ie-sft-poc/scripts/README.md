# Scripts Documentation

This directory contains utility and workflow scripts for the IE-SFT-PoC pipeline. Scripts are organized into functional categories with clear responsibilities.

## Directory Structure

```
scripts/
├── download/         # Dataset acquisition
├── preprocess/       # Data normalization and cleaning
├── export/           # Format conversion for training
├── train/            # Training, evaluation, and inference
└── utils/            # Utility functions
```

## Download Scripts

Scripts for acquiring datasets from external sources (HuggingFace Hub, internal stores, etc.).

| Script | Purpose | Key Arguments | Example |
|--------|---------|---------------|---------|
| `download_instructie.py` | Download InstructIE dataset from HuggingFace | `--output-dir`, `--split` | `python scripts/download/download_instructie.py --output-dir data/raw/instructie/` |
| `download_reference_gollie_assets.py` | Download GoLLIE reference schemas (not training data) | `--output-dir` | `python scripts/download/download_reference_gollie_assets.py --output-dir data/raw/gollie_reference/` |
| `download_internal_template.py` | Download internal dataset templates (requires auth) | `--auth-token`, `--output-dir` | `python scripts/download/download_internal_template.py --auth-token $TOKEN` |

### download_instructie.py

Downloads the InstructIE dataset from HuggingFace Hub and stores in JSONL format.

**Usage:**
```bash
python scripts/download/download_instructie.py [OPTIONS]
```

**Options:**
- `--output-dir PATH` (default: `data/raw/instructie/`) - Output directory for downloaded files
- `--split SPLIT` (default: all splits) - Specific split to download (train, validation, test)
- `--max-records N` - Limit number of records (default: no limit)
- `--cache-dir PATH` - HuggingFace cache directory

**Example:**
```bash
# Download all splits
python scripts/download/download_instructie.py

# Download only training split to custom location
python scripts/download/download_instructie.py \
  --output-dir /mnt/data/instructie/ \
  --split train

# Download first 10k records for testing
python scripts/download/download_instructie.py --max-records 10000
```

**Output:**
- `data/raw/instructie/train.jsonl`
- `data/raw/instructie/validation.jsonl`
- `data/raw/instructie/test.jsonl`

### download_reference_gollie_assets.py

Downloads GoLLIE reference schemas, task definitions, and patterns. These are used for schema design reference only, not as training data.

**Usage:**
```bash
python scripts/download/download_reference_gollie_assets.py [OPTIONS]
```

**Options:**
- `--output-dir PATH` (default: `data/raw/gollie_reference/`) - Output directory
- `--include-schemas` - Download schema definitions
- `--include-tasks` - Download task descriptions

**Example:**
```bash
python scripts/download/download_reference_gollie_assets.py
```

**Output:**
- `data/raw/gollie_reference/schemas.json`
- `data/raw/gollie_reference/tasks.json`
- `data/raw/gollie_reference/metadata.json`

### download_internal_template.py

Downloads internal dataset templates and configurations. Requires authentication.

**Usage:**
```bash
python scripts/download/download_internal_template.py [OPTIONS]
```

**Options:**
- `--auth-token TOKEN` - Authentication token (or set via `INTERNAL_AUTH_TOKEN` env var)
- `--output-dir PATH` (default: `data/raw/internal/`) - Output directory
- `--template-type TYPE` - Specific template type to download

**Example:**
```bash
python scripts/download/download_internal_template.py \
  --auth-token $INTERNAL_TOKEN \
  --template-type kv_extraction
```

## Preprocess Scripts

Scripts for normalizing datasets to canonical format and validating data quality.

| Script | Purpose | Key Arguments | Example |
|--------|---------|---------------|---------|
| `normalize_instructie.py` | Convert InstructIE → canonical format | `--input-dir`, `--output-dir` | `python scripts/preprocess/normalize_instructie.py --input-dir data/raw/instructie/ --output-dir data/interim/instructie/` |
| `unify_ie_datasets.py` | Merge multiple datasets into single unified JSONL | `--input`, `--output` | `python scripts/preprocess/unify_ie_datasets.py --input data/interim/*/*.jsonl --output data/processed/unified.jsonl` |
| `validate_canonical_dataset.py` | Validate schema, required fields, and consistency | `--input` | `python scripts/preprocess/validate_canonical_dataset.py --input data/processed/unified.jsonl` |
| `build_internal_kv_template.py` | Build internal KV template from raw data | `--input-dir`, `--output-dir` | `python scripts/preprocess/build_internal_kv_template.py --input-dir data/raw/internal/ --output-dir data/interim/internal_kv/` |

### normalize_instructie.py

Converts InstructIE dataset (original format) to canonical IE schema format.

**Usage:**
```bash
python scripts/preprocess/normalize_instructie.py [OPTIONS]
```

**Options:**
- `--input-dir PATH` (default: `data/raw/instructie/`) - Input directory with JSONL files
- `--output-dir PATH` (default: `data/interim/instructie/`) - Output directory for normalized files
- `--task-types LIST` (default: all) - Filter to specific task types: kv, entity, relation
- `--validate` - Run validation after normalization
- `--strict` - Treat warnings as errors

**Example:**
```bash
# Basic normalization
python scripts/preprocess/normalize_instructie.py

# With validation enabled
python scripts/preprocess/normalize_instructie.py --validate

# Only entity and relation tasks
python scripts/preprocess/normalize_instructie.py \
  --task-types entity,relation

# Strict mode (fail on any issues)
python scripts/preprocess/normalize_instructie.py --strict
```

**Output:**
- `data/interim/instructie/train.jsonl`
- `data/interim/instructie/validation.jsonl`
- `data/interim/instructie/test.jsonl`

**Records Format:**
```json
{
  "id": "instructie_train_001",
  "text": "Apple was founded by Steve Jobs in California.",
  "lang": "en",
  "source": "instructie",
  "task_types": ["entity", "relation"],
  "schema": { "kv": [], "entity": ["ORG", "PERSON", "LOC"], "relation": ["founded_by"] },
  "answer": {
    "kv": {},
    "entity": [...],
    "relation": [...]
  },
  "meta": {
    "dataset": "instructie",
    "license": "cc-by-4.0",
    "split": "train"
  }
}
```

### unify_ie_datasets.py

Merges multiple normalized datasets into a single unified JSONL file, combining all task types.

**Usage:**
```bash
python scripts/preprocess/unify_ie_datasets.py [OPTIONS]
```

**Options:**
- `--input PATTERN` - Glob pattern for input files (e.g., `data/interim/*/*.jsonl`)
- `--output PATH` - Output unified JSONL file path
- `--max-records N` - Limit total merged records
- `--sample-ratio FLOAT` - Downsample each dataset (0.0-1.0)
- `--validate` - Validate merged output
- `--remove-duplicates` - Deduplicate by record ID

**Example:**
```bash
# Merge all interim datasets
python scripts/preprocess/unify_ie_datasets.py \
  --input data/interim/*/*.jsonl \
  --output data/processed/unified.jsonl

# Merge with validation
python scripts/preprocess/unify_ie_datasets.py \
  --input data/interim/*/*.jsonl \
  --output data/processed/unified.jsonl \
  --validate

# Merge only 50% of each dataset (for quick testing)
python scripts/preprocess/unify_ie_datasets.py \
  --input data/interim/*/*.jsonl \
  --output data/processed/unified_sample.jsonl \
  --sample-ratio 0.5
```

**Output:**
- Single JSONL file with merged records: `data/processed/unified.jsonl`
- Records contain metadata indicating original dataset source

### validate_canonical_dataset.py

Validates a canonical format JSONL file for schema correctness, required fields, and data consistency.

**Usage:**
```bash
python scripts/preprocess/validate_canonical_dataset.py [OPTIONS]
```

**Options:**
- `--input PATH` - JSONL file to validate
- `--strict` - Treat all warnings as errors
- `--report-file PATH` - Save validation report as JSON
- `--sample N` - Validate only first N records (for quick checks)

**Validation Checks:**
- Schema validation (correct JSON structure)
- Required fields: `id`, `text`, `task_types`, `answer`
- Task type validity (must be in: kv, entity, relation)
- Entity annotations: `text`, `type`, `start`, `end` required
- Relation annotations: `head`, `head_type`, `relation`, `tail`, `tail_type` required
- Text length constraints (1-10,000 characters)
- Span overlap detection
- Consistency between declared `task_types` and actual content

**Example:**
```bash
# Validate a dataset
python scripts/preprocess/validate_canonical_dataset.py \
  --input data/processed/unified.jsonl

# Strict validation with report
python scripts/preprocess/validate_canonical_dataset.py \
  --input data/processed/unified.jsonl \
  --strict \
  --report-file validation_report.json

# Quick validation of first 100 records
python scripts/preprocess/validate_canonical_dataset.py \
  --input data/processed/unified.jsonl \
  --sample 100
```

**Output:**
```
Validation Status: PASS
File: data/processed/unified.jsonl
Total records: 80000
Valid: 79998
Invalid: 2
Errors: ...
Warnings: ...
```

### build_internal_kv_template.py

Converts internal KV extraction template data to canonical format.

**Usage:**
```bash
python scripts/preprocess/build_internal_kv_template.py [OPTIONS]
```

**Options:**
- `--input-dir PATH` - Input directory with raw internal data
- `--output-dir PATH` (default: `data/interim/internal_kv/`) - Output directory
- `--format FORMAT` - Input format specification
- `--validate` - Validate output

**Example:**
```bash
python scripts/preprocess/build_internal_kv_template.py \
  --input-dir data/raw/internal/ \
  --output-dir data/interim/internal_kv/ \
  --validate
```

## Export Scripts

Scripts for converting processed data to formats compatible with training systems.

| Script | Purpose | Key Arguments | Example |
|--------|---------|---------------|---------|
| `export_train_dev_test.py` | Create train/dev/test splits from unified data | `--input`, `--output`, `--train-ratio` | `python scripts/export/export_train_dev_test.py --input data/processed/unified.jsonl --output data/processed/splits/ --train-ratio 0.8` |
| `export_to_llamafactory.py` | Convert to LLaMA-Factory format with instruction templates | `--input`, `--output`, `--template` | `python scripts/export/export_to_llamafactory.py --input data/processed/splits/ --output data/processed/llamafactory/` |

### export_train_dev_test.py

Splits unified JSONL dataset into train, dev, and test sets.

**Usage:**
```bash
python scripts/export/export_train_dev_test.py [OPTIONS]
```

**Options:**
- `--input PATH` - Input unified JSONL file
- `--output PATH` (default: `data/processed/splits/`) - Output directory for splits
- `--train-ratio FLOAT` (default: 0.8) - Fraction for training
- `--dev-ratio FLOAT` (default: 0.1) - Fraction for development
- `--test-ratio FLOAT` (default: 0.1) - Fraction for testing
- `--seed INT` (default: 42) - Random seed for reproducibility
- `--stratify FIELD` - Stratify by field (e.g., task_types, source)

**Example:**
```bash
# Standard 80/10/10 split
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/splits/

# Custom split: 70/15/15
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/splits/ \
  --train-ratio 0.7 \
  --dev-ratio 0.15

# Stratified by task type (ensure balanced distribution)
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/splits/ \
  --stratify task_types
```

**Output:**
- `data/processed/splits/train.jsonl`
- `data/processed/splits/dev.jsonl`
- `data/processed/splits/test.jsonl`
- `data/processed/splits/split_info.json` (metadata about split)

### export_to_llamafactory.py

Converts canonical format to LLaMA-Factory training format with instruction templates and prompt-response pairs.

**Usage:**
```bash
python scripts/export/export_to_llamafactory.py [OPTIONS]
```

**Options:**
- `--input PATH` - Input directory with train/dev/test JSONL splits
- `--output PATH` (default: `data/processed/llamafactory/`) - Output directory
- `--template TEMPLATE` (default: auto) - Prompt template style
- `--include-test` - Export test set (default: train/dev only)
- `--instruction-format FORMAT` - How to format IE instructions

**Template Options:**
- `auto`: Auto-detect from config
- `minimal`: Concise instructions
- `detailed`: Comprehensive schema specifications
- `schema-conditional`: Include full schema in prompt

**Example:**
```bash
# Basic export
python scripts/export/export_to_llamafactory.py \
  --input data/processed/splits/ \
  --output data/processed/llamafactory/

# With test set and detailed template
python scripts/export/export_to_llamafactory.py \
  --input data/processed/splits/ \
  --output data/processed/llamafactory/ \
  --include-test \
  --template detailed

# Schema-conditional format (include schema in every prompt)
python scripts/export/export_to_llamafactory.py \
  --input data/processed/splits/ \
  --output data/processed/llamafactory/ \
  --template schema-conditional
```

**Output:**
- `data/processed/llamafactory/train.jsonl` (conversation format)
- `data/processed/llamafactory/dev.jsonl` (conversation format)
- `data/processed/llamafactory/export_info.json` (metadata)

**Format Example:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an information extraction expert..."
    },
    {
      "role": "user",
      "content": "Extract entities and relations from: [TEXT]\nSchema: [SCHEMA]"
    },
    {
      "role": "assistant",
      "content": "{\"entity\": [...], \"relation\": [...]}"
    }
  ]
}
```

## Training Scripts

Bash scripts for orchestrating training, evaluation, and inference using LLaMA-Factory backend.

| Script | Purpose | Key Arguments | Example |
|--------|---------|---------------|---------|
| `run_sft_qwen3.sh` | Train Qwen3-0.6B with LoRA | `--config`, `--dry-run` | `bash scripts/train/run_sft_qwen3.sh` |
| `run_sft_qwen35.sh` | Train Qwen3.5-0.8B with LoRA | `--config`, `--dry-run` | `bash scripts/train/run_sft_qwen35.sh` |
| `run_sft_olmo3_poc.sh` | Train OLMo3 (experimental) | `--config`, `--dry-run` | `bash scripts/train/run_sft_olmo3_poc.sh` |
| `run_eval_qwen3.sh` | Evaluate trained Qwen3 model | `--model-path`, `--data-path` | `bash scripts/train/run_eval_qwen3.sh` |
| `run_infer_qwen3.sh` | Run inference/generation with trained model | `--model-path`, `--prompt` | `bash scripts/train/run_infer_qwen3.sh` |
| `merge_lora_qwen3.sh` | Merge LoRA weights into base model | `--base-model`, `--lora-path` | `bash scripts/train/merge_lora_qwen3.sh` |

### run_sft_qwen3.sh

Trains Qwen3-0.6B model on IE tasks using LoRA fine-tuning.

**Usage:**
```bash
bash scripts/train/run_sft_qwen3.sh [OPTIONS]
```

**Options:**
- `--config PATH` (default: `configs/sft/qwen3_lora_sft.yaml`) - Training config file
- `--dry-run` - Show command without executing

**Environment Variables:**
- `CONFIG_PATH` - Override config path
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `WANDB_DISABLED` - Disable W&B logging

**Example:**
```bash
# Standard training with default config
bash scripts/train/run_sft_qwen3.sh

# With custom config
bash scripts/train/run_sft_qwen3.sh --config configs/sft/custom_qwen3.yaml

# Dry run (show command without training)
bash scripts/train/run_sft_qwen3.sh --dry-run

# With specific GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/train/run_sft_qwen3.sh

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_sft_qwen3.sh
```

**Output:**
- Checkpoint directory: `./outputs/qwen3-sft-*/` (or path in config)
- Model weights, optimizer states, and training logs
- Validation metrics logged to W&B (if enabled) or tensorboard

**Typical Runtime:**
- GPU: A100 or newer: 4-8 hours for 3 epochs on 80k dataset
- GPU: RTX 4090: 8-16 hours
- Multi-GPU scales roughly linearly

### run_sft_qwen35.sh

Trains Qwen3.5-0.8B (slightly larger) model on IE tasks.

**Usage:**
```bash
bash scripts/train/run_sft_qwen35.sh [OPTIONS]
```

**Options:**
- `--config PATH` (default: `configs/sft/qwen3_5_lora_sft.yaml`)
- `--dry-run`

**Example:**
```bash
bash scripts/train/run_sft_qwen35.sh
```

### run_sft_olmo3_poc.sh

Trains OLMo3 model (experimental, requires OLMo3 model availability).

**Usage:**
```bash
bash scripts/train/run_sft_olmo3_poc.sh [OPTIONS]
```

**Options:**
- `--config PATH` (default: `configs/sft/olmo3_poc_sft.yaml`)
- `--dry-run`

**Status:**
- Currently experimental
- Awaits public OLMo3 model release
- May require adjustment of hyperparameters based on model characteristics

**Example:**
```bash
bash scripts/train/run_sft_olmo3_poc.sh --dry-run  # Check command first
```

### run_eval_qwen3.sh

Evaluates trained Qwen3 model on validation/test datasets.

**Usage:**
```bash
bash scripts/train/run_eval_qwen3.sh [OPTIONS]
```

**Options:**
- `--model-path PATH` - Path to model checkpoint or HF model ID
- `--data-path PATH` - Path to evaluation data
- `--split SPLIT` (default: dev) - Evaluation split (dev, test)

**Example:**
```bash
# Evaluate latest checkpoint
bash scripts/train/run_eval_qwen3.sh \
  --model-path ./outputs/qwen3-sft-latest/ \
  --data-path data/processed/llamafactory/dev.jsonl

# Evaluate on test set
bash scripts/train/run_eval_qwen3.sh \
  --model-path ./outputs/qwen3-sft-latest/ \
  --data-path data/processed/llamafactory/test.jsonl \
  --split test
```

**Metrics:**
- Entity F1, precision, recall
- Relation extraction metrics
- KV accuracy
- Overall extraction quality

### run_infer_qwen3.sh

Runs inference/generation with trained Qwen3 model.

**Usage:**
```bash
bash scripts/train/run_infer_qwen3.sh [OPTIONS]
```

**Options:**
- `--model-path PATH` - Model checkpoint or HF ID
- `--prompt TEXT` - Input text to extract from
- `--interactive` - Interactive mode (read prompts from stdin)

**Example:**
```bash
# Single prompt
bash scripts/train/run_infer_qwen3.sh \
  --model-path ./outputs/qwen3-sft-latest/ \
  --prompt "Apple was founded by Steve Jobs in California."

# Interactive mode
bash scripts/train/run_infer_qwen3.sh \
  --model-path ./outputs/qwen3-sft-latest/ \
  --interactive
```

### merge_lora_qwen3.sh

Merges trained LoRA weights into the base model for deployment.

**Usage:**
```bash
bash scripts/train/merge_lora_qwen3.sh [OPTIONS]
```

**Options:**
- `--base-model PATH` (default: Qwen/Qwen3-0.6B)
- `--lora-path PATH` - Path to LoRA checkpoint
- `--output-path PATH` (default: `./outputs/merged-qwen3/`)

**Example:**
```bash
# Merge LoRA weights
bash scripts/train/merge_lora_qwen3.sh \
  --base-model Qwen/Qwen3-0.6B \
  --lora-path ./outputs/qwen3-sft-checkpoint-500/ \
  --output-path ./outputs/merged-qwen3-final/
```

**Output:**
- Merged model saved to output path
- Single model directory compatible with standard transformers API
- No separate LoRA weights needed for inference

## Utility Scripts

Utility scripts for environment checking and data exploration.

| Script | Purpose | Key Arguments | Example |
|--------|---------|---------------|---------|
| `env_check.py` | Verify environment setup and dependencies | None (reads from `.env`) | `python scripts/utils/env_check.py` |
| `print_dataset_stats.py` | Print statistics about datasets | `--input`, `--detailed` | `python scripts/utils/print_dataset_stats.py --input data/processed/unified.jsonl` |

### env_check.py

Verifies that the environment is properly configured for running the IE-SFT-PoC pipeline.

**Usage:**
```bash
python scripts/utils/env_check.py [OPTIONS]
```

**Checks:**
- Python version compatibility
- Required packages installed (transformers, torch, llamafactory, etc.)
- GPU/CUDA availability
- Environment variables set correctly
- Paths exist and are readable
- HuggingFace token configured
- Dataset files accessible

**Example:**
```bash
python scripts/utils/env_check.py
```

**Output:**
```
Environment Check Results
========================

Python:        ✓ 3.10.12
PyTorch:       ✓ 2.0.1+cu118
Transformers:  ✓ 4.36.2
LLaMA-Factory: ✓ installed
CUDA:          ✓ available (8 devices)
GPU Memory:    ✓ 81GB total

Paths:
  data/raw/           ✓ exists
  data/interim/       ✓ exists
  data/processed/     ✓ exists
  configs/            ✓ exists

Environment Variables:
  CUDA_VISIBLE_DEVICES:    ✓ 0
  HF_TOKEN:                ⚠ not set (optional for public models)
  WANDB_DISABLED:          ✓ true

Status: READY
```

### print_dataset_stats.py

Prints statistical summaries of canonical format datasets.

**Usage:**
```bash
python scripts/utils/print_dataset_stats.py [OPTIONS]
```

**Options:**
- `--input PATH` - JSONL file to analyze
- `--detailed` - Show detailed statistics (slower on large files)
- `--output-file PATH` - Save stats to JSON file

**Statistics Shown:**
- Total records
- Split distribution (train/dev/test)
- Task type distribution (KV/entity/relation)
- Language distribution
- Source distribution
- Average text length
- Entity/relation counts

**Example:**
```bash
# Quick stats
python scripts/utils/print_dataset_stats.py \
  --input data/processed/unified.jsonl

# Detailed analysis
python scripts/utils/print_dataset_stats.py \
  --input data/processed/unified.jsonl \
  --detailed

# Save to file
python scripts/utils/print_dataset_stats.py \
  --input data/processed/unified.jsonl \
  --detailed \
  --output-file dataset_stats.json
```

## Common Workflows

### Complete Data Pipeline

```bash
# 1. Download data
python scripts/download/download_instructie.py

# 2. Normalize to canonical format
python scripts/preprocess/normalize_instructie.py

# 3. Merge datasets (if multiple)
python scripts/preprocess/unify_ie_datasets.py \
  --input data/interim/*/*.jsonl \
  --output data/processed/unified.jsonl

# 4. Validate
python scripts/preprocess/validate_canonical_dataset.py \
  --input data/processed/unified.jsonl

# 5. Create splits
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/splits/

# 6. Export for LLaMA-Factory
python scripts/export/export_to_llamafactory.py \
  --input data/processed/splits/ \
  --output data/processed/llamafactory/
```

### Quick Development Testing

```bash
# Download, process, and train on small subset
python scripts/download/download_instructie.py --max-records 5000
python scripts/preprocess/normalize_instructie.py --validate
python scripts/preprocess/unify_ie_datasets.py \
  --input data/interim/*/*.jsonl \
  --output data/processed/unified.jsonl \
  --max-records 5000
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/splits/
python scripts/export/export_to_llamafactory.py \
  --input data/processed/splits/ \
  --output data/processed/llamafactory/

# Quick 1-epoch training
cp configs/sft/qwen3_lora_sft.yaml configs/sft/quick_test.yaml
# Edit quick_test.yaml: num_train_epochs: 1, eval_steps: 100
bash scripts/train/run_sft_qwen3.sh --config configs/sft/quick_test.yaml
```

### Production Training

```bash
# Full pipeline with validation
bash scripts/utils/env_check.py  # Verify setup

python scripts/download/download_instructie.py  # Get full dataset
python scripts/preprocess/normalize_instructie.py --validate
python scripts/preprocess/unify_ie_datasets.py \
  --input data/interim/*/*.jsonl \
  --output data/processed/unified.jsonl \
  --validate
python scripts/export/export_train_dev_test.py \
  --input data/processed/unified.jsonl \
  --output data/processed/splits/ \
  --train-ratio 0.8 \
  --dev-ratio 0.1
python scripts/export/export_to_llamafactory.py \
  --input data/processed/splits/ \
  --output data/processed/llamafactory/ \
  --include-test

# Train on all GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/run_sft_qwen3.sh

# Evaluate
bash scripts/train/run_eval_qwen3.sh \
  --model-path ./outputs/qwen3-sft-latest/ \
  --data-path data/processed/llamafactory/test.jsonl

# Merge for deployment
bash scripts/train/merge_lora_qwen3.sh \
  --lora-path ./outputs/qwen3-sft-latest/
```

---

For more details, see the main README.md and documentation in docs/ directory.
