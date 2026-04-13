# Configuration Files

This directory contains all YAML configuration files for the IE-SFT-PoC system. Configurations control model selection, dataset metadata, and training parameters.

## Directory Structure

```
configs/
├── models/          # Model definitions
├── datasets/        # Dataset metadata
└── sft/             # LLaMA-Factory SFT training configs
```

## models/ Directory

Model configuration files define how models are instantiated, tokenized, and used for training and inference.

### File Format

Each model config is a YAML file with the following structure:

```yaml
### Model Name
model_name_or_path: <HF_model_id_or_path>  # HuggingFace model ID or local path
template: <template_name>                   # Chat template (qwen, olmo, etc.)
family: <model_family>                      # Model family for adapter selection
recommended_batch_size: <int>               # Suggested batch size
default_max_length: <int>                   # Default sequence length
bf16: <bool>                                # bf16 support
notes: <string>                             # Optional description
```

### Available Models

**qwen3_0_6b.yaml**
- Model: Qwen/Qwen3-0.6B
- Size: 600 million parameters
- Recommended for: Development, research, resource-constrained environments
- Template: qwen
- Max Length: 2048 tokens

**qwen3_5_0_8b.yaml**
- Model: Qwen/Qwen3.5-0.8B
- Size: 800 million parameters
- Recommended for: Better quality than 0.6B with reasonable resource requirements
- Template: qwen
- Max Length: 2048 tokens

**olmo3_poc.yaml**
- Model: allenai/OLMo3-base (when available)
- Size: Variable (depends on OLMo3 release)
- Status: Proof-of-concept, awaiting official release
- Template: olmo (to be confirmed)
- Max Length: TBD (to be confirmed)
- Notes: Experimental, may require trust_remote_code=True

### How to Override Model Configuration

#### Option 1: CLI Arguments
Some scripts accept `--model` or `--model-config` arguments:
```bash
python scripts/train/run_sft.py --model qwen3-0.6b
```

#### Option 2: Environment Variables
Set model-related env vars in `.env`:
```bash
DEFAULT_MODEL_FAMILY=qwen3
DEFAULT_MODEL_NAME=qwen3-0.6b
HF_HOME=./models
```

#### Option 3: Custom Config File
Create a new config file and reference it:
```bash
python scripts/train/run_sft.py --model-config configs/models/custom_model.yaml
```

### Adding a New Model

1. Create a new file `configs/models/<model_name>.yaml`
2. Fill in the required fields based on the model's specifications
3. Update `src/models/model_registry.py` to register the model
4. Create or update adapter in `src/models/` if it's a new model family
5. Test with `scripts/utils/env_check.py`

Example for a hypothetical new model:
```yaml
### Custom Model Config
model_name_or_path: custom-org/custom-model-7b
template: custom
family: custom
recommended_batch_size: 8
default_max_length: 4096
bf16: true
notes: "Custom model for IE tasks, supports longer contexts"
```

## datasets/ Directory

Dataset configuration files define metadata, source paths, and preprocessing details for datasets.

### File Format

Each dataset config is a YAML file with structure:

```yaml
### Dataset Name
dataset_name: <string>                     # Canonical dataset name
source: <url_or_path>                      # Source location (HF Hub, local, etc.)
description: <string>                      # Brief description
task_types: [<list>]                       # Supported tasks: [kv, entity, relation]
license: <string>                          # License identifier (cc-by-4.0, etc.)
size_approx: <int>                         # Approximate number of records
split_ratio: {train: <float>, dev: <float>, test: <float>}  # Default splits
format: <string>                           # Input format (instructie, gollie, etc.)
enabled: <bool>                            # Include in training by default
notes: <string>                            # Additional info
```

### Available Datasets

**instructie.yaml**
- Source: HuggingFace dataset repository
- License: Public, permissive (CC-BY or similar)
- Tasks: KV, entity, relation extraction
- Size: ~80k examples (approx)
- Status: Enabled by default, actively used
- Notes: Primary training dataset, well-maintained

**unified_ie_example.yaml**
- Source: Example/template configuration
- Purpose: Shows how to configure a unified IE dataset
- Status: Reference only, not a real dataset

**internal_kv_example.yaml**
- Source: Internal KV extraction template
- Purpose: Example for internal dataset setup
- Status: Reference only, requires separate access procedures

### How to Override Dataset Configuration

#### Option 1: Environment Variable
```bash
export DATASET_REGISTRY=data/metadata/custom_registry.yaml
python scripts/train/train.py
```

#### Option 2: Edit data/metadata/dataset_registry.yaml
The central registry file loads all datasets. Edit or add entries there.

#### Option 3: Command-Line Arguments
Some scripts accept dataset arguments:
```bash
python scripts/preprocess/normalize_instructie.py \
  --input-config configs/datasets/instructie.yaml \
  --output-dir data/interim/instructie/
```

### Adding a New Dataset

1. Create `configs/datasets/<dataset_name>.yaml` with metadata
2. Create normalization script in `scripts/preprocess/normalize_<dataset_name>.py`
3. Add entry to `data/metadata/dataset_registry.yaml`
4. Document license in `docs/dataset_policy.md`
5. Test with:
   ```bash
   python scripts/preprocess/normalize_<dataset_name>.py
   python scripts/preprocess/validate_canonical_dataset.py --input data/interim/<dataset_name>/
   ```

## sft/ Directory

LLaMA-Factory SFT (Supervised Fine-Tuning) configuration files define training hyperparameters, data paths, and model settings.

### File Format

SFT configs are YAML files compatible with LLaMA-Factory's trainer config schema:

```yaml
# LLaMA-Factory trainer configuration
# See: https://github.com/hiyouga/LLaMA-Factory

### Model Configuration
model_name_or_path: <string>               # Model path or HF ID
template: <string>                         # Chat template
cutoff_len: <int>                          # Max sequence length

### Data Configuration
train_file: <string>                       # Path to training JSONL
eval_file: <string>                        # Path to eval JSONL
dataset: <string>                          # Dataset name for registry

### Training Hyperparameters
output_dir: <string>                       # Checkpoint output directory
per_device_train_batch_size: <int>         # Batch size
per_device_eval_batch_size: <int>          # Eval batch size
learning_rate: <float>                     # Learning rate (typically 1e-4 to 5e-5)
num_train_epochs: <int>                    # Number of training epochs
warmup_steps: <int>                        # Warmup steps
weight_decay: <float>                      # Weight decay for regularization
max_grad_norm: <float>                     # Gradient clipping

### Optimization
optim: <string>                            # Optimizer (paged_adamw_32bit, adamw_8bit)
gradient_accumulation_steps: <int>         # Accumulation steps
bf16: <bool>                               # Use bfloat16 precision
fp16: <bool>                               # Use float16 precision

### LoRA Configuration
use_lora: <bool>                           # Enable LoRA fine-tuning
lora_rank: <int>                           # LoRA rank (8-16 typical)
lora_alpha: <int>                          # LoRA alpha (scaling factor)
lora_dropout: <float>                      # LoRA dropout (0.05-0.1 typical)

### Evaluation & Logging
evaluation_strategy: <string>              # "steps", "epoch", or "no"
eval_steps: <int>                          # Evaluate every N steps
save_strategy: <string>                    # "steps" or "epoch"
save_steps: <int>                          # Save checkpoint every N steps
logging_steps: <int>                       # Log metrics every N steps
report_to: [<list>]                        # Report to wandb, tensorboard, etc.

### Advanced
seed: <int>                                # Random seed
dataloader_pin_memory: <bool>              # Pin memory for DataLoader
ddp_find_unused_parameters: <bool>         # For multi-GPU training
gradient_checkpointing: <bool>             # Memory optimization
```

### Available SFT Configs

**qwen3_lora_sft.yaml**
- Model: Qwen3-0.6B
- Method: LoRA fine-tuning
- Batch Size: 4 per device
- Learning Rate: 5e-5
- Epochs: 3
- Recommended for: Development, resource-constrained training

**qwen3_5_lora_sft.yaml**
- Model: Qwen3.5-0.8B
- Method: LoRA fine-tuning
- Batch Size: 4 per device
- Learning Rate: 5e-5
- Epochs: 3
- Recommended for: Better quality than 0.6B variant

**olmo3_poc_sft.yaml**
- Model: OLMo3 (PoC)
- Status: Experimental, awaiting official release
- Method: LoRA fine-tuning
- Notes: May require adjustments once real OLMo3 is available

### How to Use SFT Configs

#### Option 1: Use Training Scripts with Default Config
```bash
bash scripts/train/run_sft_qwen3.sh
# Uses configs/sft/qwen3_lora_sft.yaml by default
```

#### Option 2: Specify Custom Config
```bash
bash scripts/train/run_sft_qwen3.sh --config configs/sft/custom_config.yaml
```

#### Option 3: Direct LLaMA-Factory Invocation
```bash
llamafactory-cli train configs/sft/qwen3_lora_sft.yaml
```

#### Option 4: Override Parameters with Env Vars
Some parameters can be overridden via environment:
```bash
export TRAINING_LR=1e-4
export TRAINING_BATCH_SIZE=8
bash scripts/train/run_sft_qwen3.sh
```

### Creating a Custom Training Config

1. Copy an existing config as a template:
   ```bash
   cp configs/sft/qwen3_lora_sft.yaml configs/sft/my_custom_training.yaml
   ```

2. Edit parameters for your experiment:
   ```yaml
   # Increase learning rate for faster convergence
   learning_rate: 1e-4
   
   # Increase batch size if you have GPU memory
   per_device_train_batch_size: 8
   
   # More epochs for better convergence
   num_train_epochs: 5
   
   # Larger LoRA rank for higher capacity
   lora_rank: 16
   ```

3. Run training with your config:
   ```bash
   llamafactory-cli train configs/sft/my_custom_training.yaml
   ```

### Configuration Best Practices

1. **Batch Size**: Start with recommended value, reduce if OOM errors occur
2. **Learning Rate**: Typical range is 1e-4 to 5e-5 for LoRA fine-tuning
3. **Warmup**: Set to 10% of total steps for stability
4. **Evaluation**: Use small eval set for frequent feedback
5. **LoRA Rank**: 8-16 for parameter efficiency, 32+ for higher capacity
6. **Save Checkpoints**: Save every N steps to recover from crashes
7. **Mixed Precision**: Use bf16 on modern NVIDIA GPUs (A100, H100)

### Common Configuration Scenarios

**Quick Testing (Minimal Resources)**
```yaml
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
num_train_epochs: 1
learning_rate: 5e-5
lora_rank: 8
gradient_accumulation_steps: 4
```

**Production Training (High Quality)**
```yaml
per_device_train_batch_size: 16
num_train_epochs: 3
learning_rate: 1e-4
lora_rank: 16
gradient_accumulation_steps: 2
weight_decay: 0.01
warmup_steps: 500
```

**Multi-GPU Training**
```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
ddp_find_unused_parameters: true
# Run with: CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train config.yaml
```

## Environment Variable Overrides

Configuration can be overridden via environment variables in `.env`:

```bash
# Model selection
DEFAULT_MODEL_FAMILY=qwen3
DEFAULT_MODEL_NAME=qwen3-0.6b

# Training parameters
TRAINING_LR=5e-5
TRAINING_BATCH_SIZE=4
TRAINING_EPOCHS=3
TRAINING_MAX_LENGTH=2048
TRAINING_BF16=true
TRAINING_GRADIENT_ACCUMULATION=2

# Dataset selection
DATASET_REGISTRY=data/metadata/dataset_registry.yaml
DEFAULT_DATASET_SPLIT=train
```

See `.env.example` for complete list of configuration options.

## Configuration Validation

Validate configurations before training:

```bash
# Check environment
python scripts/utils/env_check.py

# Check specific config file
python -c "import yaml; print(yaml.safe_load(open('configs/sft/qwen3_lora_sft.yaml')))"
```

## Additional Resources

- **LLaMA-Factory Documentation**: https://github.com/hiyouga/LLaMA-Factory
- **HuggingFace Model Hub**: https://huggingface.co/models
- **YAML Syntax**: https://yaml.org/

---

For questions about specific configurations, see the main documentation or contact the research team.
