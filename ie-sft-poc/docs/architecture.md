# System Architecture

Complete architecture documentation for IE-SFT-PoC, including system design, module responsibilities, data flow, and extension points.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    IE-SFT-PoC System                           │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Download    │      │  Preprocess  │      │   Export     │
│  Scripts     │ ───► │  & Normalize │ ───► │  to Training │
└──────────────┘      └──────────────┘      └──────────────┘
       ▲                     ▲                      ▲
       │                     │                      │
   ┌───────────────────────────────────────────────────────┐
   │          Core Modules (src/)                         │
   │                                                      │
   │  ┌────────┐  ┌────────┐  ┌────────────┐           │
   │  │Datasets│  │ Models │  │ Training   │           │
   │  └────────┘  └────────┘  └────────────┘           │
   │      ▲           ▲               ▲                  │
   │      └───────────┴───────────────┘                  │
   │              │                                      │
   │      ┌───────────────┐                             │
   │      │Canonical      │                             │
   │      │Schema & Utils │                             │
   │      └───────────────┘                             │
   └───────────────────────────────────────────────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Train Data   │      │  Evaluation  │      │  Inference   │
│             │      │             │      │             │
│LLaMA-Factory│      │Metrics,F1   │      │Generation   │
└──────────────┘      └──────────────┘      └──────────────┘
```

## System Components

### 1. Data Pipeline Layer (scripts/)

**Responsibility**: Acquire, normalize, merge, split, and export datasets for training.

**Components**:
- `download/`: Fetch datasets from external sources (HuggingFace, internal stores)
- `preprocess/`: Normalize to canonical format, validate, merge datasets
- `export/`: Convert canonical format to training frameworks (LLaMA-Factory)
- `utils/`: Environment checks, statistics, utilities

**Key Outputs**:
- Raw datasets (`data/raw/`)
- Normalized canonical format (`data/interim/`)
- Training splits (`data/processed/splits/`)
- LLaMA-Factory format (`data/processed/llamafactory/`)

### 2. Core Modules (src/)

#### 2.1 Dataset Module (src/datasets/)

**Responsibility**: Handle dataset metadata, registry, validation, and canonical format utilities.

**Components**:
- `metadata.py`: Dataset metadata classes and configurations
- `registry.py`: Dataset registry loader and lookup
- `unified/`: Canonical format handling
  - `validator.py`: Schema validation and consistency checks
  - `splitter.py`: Train/dev/test splitting
  - `merger.py`: Dataset merging and composition
  - `stats.py`: Dataset statistics and analysis
- `gollie_reference/`: GoLLIE reference implementation (read-only design reference)

**Interfaces**:
```python
# Load dataset metadata
from src.datasets.registry import load_dataset_registry
registry = load_dataset_registry('data/metadata/dataset_registry.yaml')

# Validate canonical format
from src.datasets.unified.validator import validate_dataset
report = validate_dataset('data/interim/instructie/train.jsonl')

# Get dataset statistics
from src.datasets.unified.stats import compute_stats
stats = compute_stats('data/processed/unified.jsonl')
```

**Key Concepts**:
- Datasets can define multiple task types (KV, entity, relation)
- All datasets normalized to canonical JSONL format
- Validation ensures schema consistency before training
- Metadata tracks source, license, and split information

#### 2.2 Model Module (src/models/)

**Responsibility**: Define and manage model families, adapters, and configurations.

**Components**:
- `model_registry.py`: Central model registry and lookup
- `qwen.py`: Qwen3 model adapter
- `olmo.py`: OLMo3 model adapter (basic implementation)

**Key Adapter Methods**:
```python
class ModelAdapter:
    @property
    def model_name_or_path(self) -> str:
        """HuggingFace model ID or local path"""
    
    @property
    def template(self) -> str:
        """Chat template name (qwen, olmo, etc.)"""
    
    @property
    def max_length(self) -> int:
        """Default maximum sequence length"""
    
    def tokenizer_quirks(self) -> dict:
        """Special tokens and tokenizer config"""
    
    def validate_environment(self) -> bool:
        """Check if model dependencies available"""
```

**Adapter Pattern**:
Each model family implements:
1. Model name and HuggingFace ID
2. Chat template format
3. Special tokens (BOS, EOS, PAD, etc.)
4. Context length and batch size recommendations
5. Environment validation

**Extension Example**:
```python
class CustomModelAdapter(ModelAdapter):
    @property
    def model_name_or_path(self):
        return "custom-org/custom-model-7b"
    
    @property
    def template(self):
        return "custom"
    
    def tokenizer_quirks(self):
        return {
            'bos_token': '<s>',
            'eos_token': '</s>',
            'pad_token': '<pad>',
        }
```

#### 2.3 Training Module (src/training/)

**Responsibility**: Orchestrate training pipeline using LLaMA-Factory backend.

**Components**:
- `config_builder.py`: Build LLaMA-Factory training configs
- `dataset_registry_builder.py`: Create dataset registries for trainer
- `llamafactory_runner.py`: Execute LLaMA-Factory training
- `eval_runner.py`: Evaluation pipeline
- `inference_runner.py`: Inference and generation

**Training Flow**:
1. Load training config (YAML)
2. Resolve model and dataset paths
3. Build dataset registry (if needed)
4. Execute LLaMA-Factory trainer
5. Save checkpoints and logs
6. Evaluate on validation set

**Evaluation Metrics**:
- Entity extraction: Precision, recall, F1
- Relation extraction: Precision, recall, F1
- KV extraction: Accuracy, F1
- Overall loss and perplexity

#### 2.4 OLMo3 PoC Module (src/olmo3_poc/)

**Responsibility**: Experimental OLMo3 support and integration framework.

**Status**: Proof-of-concept, awaiting public model release.

**Components**:
- `adapter.py`: OLMo3 adapter skeleton with TODOs
- `conversion.py`: Format conversion utilities
- `notes.py`: Status tracking, known differences, TODO list

**Current State**:
- Adapter abstraction framework in place
- QwenAdapter as reference implementation
- OLMoAdapter skeleton with documented TODOs
- Placeholder conversion utilities
- Comprehensive notes on unknowns and gaps

**What's Needed**:
1. Official OLMo3 model release
2. Tokenizer and template documentation
3. Training benchmarking
4. Integration testing

See `docs/olmo3_extension.md` for detailed roadmap.

### 3. Configuration System (configs/)

**Responsibility**: Externalize all configurable parameters via YAML files.

**Components**:
- `models/`: Model definitions (one per family)
  - `qwen3_0_6b.yaml`
  - `qwen3_5_0_8b.yaml`
  - `olmo3_poc.yaml`

- `datasets/`: Dataset metadata configs
  - `instructie.yaml`
  - `unified_ie_example.yaml`
  - `internal_kv_example.yaml`

- `sft/`: LLaMA-Factory training configs
  - `qwen3_lora_sft.yaml`
  - `qwen3_5_lora_sft.yaml`
  - `olmo3_poc_sft.yaml`

**Configuration Overrides** (priority order):
1. CLI arguments (highest)
2. Environment variables
3. YAML config files
4. Hard-coded defaults (lowest)

### 4. Execution Orchestration (scripts/)

**Bash scripts** wrap Python modules to provide user-friendly interfaces:

```
scripts/train/run_sft_qwen3.sh
    ↓ (calls)
llamafactory-cli train configs/sft/qwen3_lora_sft.yaml
    ↓ (uses)
src/training/llamafactory_runner.py
    ↓ (loads)
src/models/qwen.py
src/datasets/unified/validator.py
    ↓ (trains)
Qwen3-0.6B model
```

## Data Flow

### Complete End-to-End Flow

```
┌─────────────────────┐
│  Raw Data Sources   │
│  (HuggingFace Hub)  │
└──────────┬──────────┘
           │
           │ scripts/download/download_instructie.py
           ▼
┌──────────────────────────────┐
│  data/raw/instructie/        │
│  (Original format JSONL)     │
└──────────┬───────────────────┘
           │
           │ scripts/preprocess/normalize_instructie.py
           │ Uses: src/datasets/unified/validator.py
           ▼
┌──────────────────────────────┐
│  data/interim/instructie/    │
│  (Canonical format JSONL)    │
│  ✓ Validated                 │
└──────────┬───────────────────┘
           │
           │ scripts/preprocess/unify_ie_datasets.py
           │ Uses: src/datasets/unified/merger.py
           ▼
┌──────────────────────────────┐
│  data/processed/unified.jsonl│
│  (All datasets merged)       │
│  ✓ Validated                 │
└──────────┬───────────────────┘
           │
           │ scripts/export/export_train_dev_test.py
           │ Uses: src/datasets/unified/splitter.py
           ▼
┌──────────────────────────────┐
│  data/processed/splits/      │
│  - train.jsonl (80%)         │
│  - dev.jsonl (10%)           │
│  - test.jsonl (10%)          │
│  ✓ Validated                 │
└──────────┬───────────────────┘
           │
           │ scripts/export/export_to_llamafactory.py
           │ Uses: src/training/config_builder.py
           ▼
┌──────────────────────────────┐
│  data/processed/llamafactory/│
│  - train.jsonl (conversation)
│  - dev.jsonl (conversation)  │
│  ✓ Ready for training        │
└──────────┬───────────────────┘
           │
           │ bash scripts/train/run_sft_qwen3.sh
           │ Uses: src/training/llamafactory_runner.py
           │       src/models/qwen.py
           ▼
┌──────────────────────────────┐
│  LLaMA-Factory Training      │
│  - LoRA initialization       │
│  - Data loading              │
│  - Training loop             │
│  - Checkpoint saving         │
│  - Evaluation                │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  outputs/qwen3-sft-*/        │
│  - checkpoint-N/             │
│  - trainer_state.json        │
│  - training logs             │
└──────────┬───────────────────┘
           │
           │ bash scripts/train/merge_lora_qwen3.sh
           │ Uses: src/models/qwen.py
           ▼
┌──────────────────────────────┐
│  outputs/merged-qwen3/       │
│  (Full model, ready to deploy│
└──────────────────────────────┘
```

### Data Record Flow Through Canonical Schema

```
InstructIE Record (original)
┌─────────────────────────────────────────┐
│ {                                       │
│   "id": "...",                          │
│   "text": "...",                        │
│   "entities": [...],    ← Task-specific │
│   "relations": [...]       structure    │
│ }                                       │
└────────────┬────────────────────────────┘
             │ Normalization
             ▼
Canonical Record (unified)
┌─────────────────────────────────────────┐
│ {                                       │
│   "id": "...",                          │
│   "text": "...",                        │
│   "task_types": ["entity", "relation"], │
│   "schema": {...},      ← Unified       │
│   "answer": {             structure     │
│     "entity": [...],                    │
│     "relation": [...]                   │
│   },                                    │
│   "meta": {...}                         │
│ }                                       │
└────────────┬────────────────────────────┘
             │ Validation
             ├─► Check schema
             ├─► Validate fields
             ├─► Check spans
             └─► Verify consistency
             │
             ▼ (if valid)
Ready for Training/Evaluation
```

## Module Dependencies

```
                    ┌─────────────────┐
                    │  scripts/       │
                    │  (entry points) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌─────────┐    ┌──────────┐   ┌─────────┐
        │download │    │preprocess│   │export   │
        └────┬────┘    └────┬─────┘   └────┬────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                    ┌───────▼────────┐
                    │  src/datasets/ │
                    │  src/models/   │
                    │  src/training/ │
                    └───────┬────────┘
                            │
                  ┌─────────▼─────────┐
                  │  configs/         │
                  │  data/metadata/   │
                  └───────────────────┘
```

**Dependency Rules**:
1. Scripts depend on core modules
2. Core modules don't depend on scripts
3. Config files loaded dynamically
4. No circular dependencies allowed

## Extension Points

### Adding a New Model Family

1. **Create Model Adapter** (`src/models/new_model.py`):
   ```python
   from src.models import ModelAdapter
   
   class NewModelAdapter(ModelAdapter):
       @property
       def model_name_or_path(self):
           return "org/new-model-7b"
       
       @property
       def template(self):
           return "new_template"
       
       # Implement other required methods...
   ```

2. **Register Model** (`src/models/model_registry.py`):
   ```python
   MODELS = {
       'new-7b': NewModelAdapter(),
   }
   ```

3. **Create Config** (`configs/models/new_model.yaml`)

4. **Create Training Script** (`scripts/train/run_sft_new_model.sh`)

### Adding a New Dataset

1. **Create Normalization Script** (`scripts/preprocess/normalize_new_dataset.py`)
   - Load from raw format
   - Map to canonical schema
   - Validate output

2. **Create Dataset Config** (`configs/datasets/new_dataset.yaml`)

3. **Update Registry** (`data/metadata/dataset_registry.yaml`)

4. **Document** (`docs/dataset_policy.md`)

### Adding a New Training Technique

1. **Extend TrainingRunner** (`src/training/llamafactory_runner.py`)
2. **Create Config Template** (`configs/sft/new_technique.yaml`)
3. **Update Documentation** (`docs/training_flow.md`)

## Design Decisions

### 1. Canonical Schema (Unified Format)

**Decision**: Single JSONL format for all IE tasks.

**Rationale**:
- Simplifies pipeline (one validation, one format)
- Enables multi-task training on single model
- Easier dataset composition and merging
- Reduces code duplication

**Alternative Considered**: Task-specific formats
- Would require format-specific validators
- Harder to mix datasets
- More complex training pipeline

### 2. Adapter Pattern for Models

**Decision**: Abstract model families via adapter pattern.

**Rationale**:
- Easy to add new model families
- Isolates model-specific logic
- Consistent interface across families
- Enables quick experimentation

**Alternative Considered**: Monolithic model loader
- Would require extensive if/elif chains
- Harder to extend
- More tightly coupled

### 3. LLaMA-Factory as Training Backend

**Decision**: Use LLaMA-Factory instead of custom trainer.

**Rationale**:
- Production-ready, well-tested
- Supports LoRA, quantization, multi-GPU
- Active community and maintenance
- Flexible configuration system

**Alternative Considered**: Custom training loop
- More control, but more complexity
- Duplicates well-tested functionality
- Harder to maintain

### 4. YAML Configuration Over Environment

**Decision**: Centralized YAML configs for reproducibility.

**Rationale**:
- Reproducible experiments
- Easy to version control
- Single source of truth
- CLI override capability

**Alternative Considered**: Pure environment variables
- Would require many ENV vars
- Harder to track experiments
- Less reproducible

## Performance Considerations

### Memory Efficiency
- LoRA reduces trainable parameters by 99%
- Batch size 4 fits in 16GB VRAM (Qwen3-0.6B)
- Gradient accumulation allows larger effective batch size
- bf16 mixed precision reduces memory by 50%

### Training Speed
- Expected 4-8 hours for 80k examples on A100
- Multi-GPU scales roughly linearly up to 4 GPUs
- Evaluation bottleneck at frequent intervals
- Checkpoint saving can be expensive

### Data Pipeline Performance
- Normalization: ~1k records/sec
- Validation: ~500 records/sec
- Merging: ~2k records/sec
- Export to LLaMA-Factory: ~1k records/sec

## Monitoring and Observability

### Logging
- Script logs: `logs/` directory (if enabled)
- Training logs: Checkpoint directory
- Dataset logs: Stdout + file (configurable)

### Metrics Tracking
- W&B integration (optional, via config)
- TensorBoard logs (automatic in LLaMA-Factory)
- Dataset statistics (via `print_dataset_stats.py`)
- Validation reports (via `validate_canonical_dataset.py`)

### Debugging Tools
- `env_check.py`: Verify environment setup
- `validate_canonical_dataset.py`: Check data quality
- `print_dataset_stats.py`: Analyze distributions
- Training logs: Monitor loss, metrics, checkpoints

---

For detailed information on specific components, see individual documentation files and source code comments.

**Last Updated**: April 2026  
**Architecture Version**: 1.0
