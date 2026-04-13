# OLMo3 Extension and Integration

Comprehensive documentation for the OLMo3 proof-of-concept integration, current status, and roadmap for full support.

## Current Status

**Status**: Experimental Proof-of-Concept (PoC)  
**Version**: 0.1.0  
**Last Updated**: April 2026  
**Maturity**: Early-stage research

### Summary

OLMo3 support is in early proof-of-concept stage. The adapter framework is designed and partially implemented, but actual training and inference awaits the public release of OLMo3 models. This document serves as the integration blueprint and status tracker.

### What's Done

1. **Adapter Framework** ✓
   - Abstract adapter pattern designed and documented
   - QwenAdapter implementation as reference
   - Extension points clearly identified

2. **Model Registry Integration** ✓
   - OLMo3 entry in `src/models/model_registry.py`
   - Configuration template in `configs/models/olmo3_poc.yaml`
   - Training config template in `configs/sft/olmo3_poc_sft.yaml`

3. **Documentation** ✓
   - Complete architecture of adapter pattern
   - TODO list for completion
   - Known differences documented
   - Integration roadmap created

### What's In Progress

1. **Tokenizer Compatibility**
   - Waiting for tokenizer details from OLMo3 release
   - Placeholder code in `src/olmo3_poc/tokenizer_quirks()`

2. **Template Format**
   - OLMo3 chat template format not yet finalized
   - Placeholder in `src/olmo3_poc/default_template`
   - Needs confirmation against real model

3. **Environment Validation**
   - Placeholder validation in `validate_environment()`
   - Actual tests pending model availability

### What's Blocked

1. **Official OLMo3 Release**
   - Primary blocker for real testing
   - Expected: [Check OLMo3 project timeline]
   - Affects: All testing, benchmarking, validation

2. **Tokenizer Documentation**
   - Need detailed tokenizer specification
   - Special tokens, padding behavior, vocab size
   - Required for: Prompt conversion, sequence length handling

3. **Template Format Specification**
   - Exact chat template format not published
   - System prompt handling unclear
   - Role markers undocumented

## Adapter Pattern Explanation

The system uses an **adapter pattern** to support multiple model families with minimal code duplication.

### Abstract Adapter Interface

All model adapters implement this interface:

```python
from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    """Base class for model family adapters."""
    
    @property
    @abstractmethod
    def model_name_or_path(self) -> str:
        """HuggingFace model ID or local path."""
    
    @property
    @abstractmethod
    def template(self) -> str:
        """Chat template name (qwen, olmo, etc.)."""
    
    @property
    @abstractmethod
    def max_length(self) -> int:
        """Default maximum context length."""
    
    @abstractmethod
    def tokenizer_quirks(self) -> dict:
        """Special tokens and tokenizer configuration."""
    
    @abstractmethod
    def validate_environment(self) -> bool:
        """Check if model dependencies are available."""
```

### Qwen3 Adapter (Reference Implementation)

```python
class QwenAdapter(ModelAdapter):
    @property
    def model_name_or_path(self):
        return "Qwen/Qwen3-0.6B"  # or 0.8B variant
    
    @property
    def template(self):
        return "qwen"  # LLaMA-Factory template name
    
    @property
    def max_length(self):
        return 2048
    
    def tokenizer_quirks(self):
        return {
            'bos_token_id': 151857,
            'eos_token_id': 151858,
            'pad_token_id': 151859,
            'padding_side': 'right',
        }
    
    def validate_environment(self):
        import transformers
        # Check Qwen3 is available
        return True
```

### OLMo3 Adapter (To Be Completed)

```python
class OLMoAdapter(ModelAdapter):
    @property
    def model_name_or_path(self):
        # TODO: Confirm actual model ID once released
        return "allenai/OLMo3-base"
    
    @property
    def template(self):
        # TODO: Determine actual template name
        return "olmo"  # Placeholder, to be confirmed
    
    @property
    def max_length(self):
        # TODO: Confirm actual max context length
        return 4096  # Placeholder, to be confirmed
    
    def tokenizer_quirks(self):
        # TODO: Get actual special tokens from OLMo3
        return {
            'bos_token_id': None,  # TODO: Confirm
            'eos_token_id': None,  # TODO: Confirm
            'pad_token_id': None,  # TODO: Confirm
            'padding_side': 'right',
        }
    
    def validate_environment(self):
        # TODO: Test with real OLMo3 model
        try:
            from transformers import AutoModelForCausalLM
            # Try loading model
            return True
        except Exception:
            return False
```

### Adding a New Model Family

To add support for another model:

1. **Create Adapter Class**:
   ```python
   # src/models/custom.py
   from src.models import ModelAdapter
   
   class CustomAdapter(ModelAdapter):
       # Implement all abstract methods
   ```

2. **Register in Model Registry**:
   ```python
   # src/models/model_registry.py
   from src.models.custom import CustomAdapter
   
   MODELS = {
       'qwen3-0.6b': QwenAdapter(),
       'custom-7b': CustomAdapter(),
   }
   ```

3. **Create Config**:
   ```yaml
   # configs/models/custom_7b.yaml
   model_name_or_path: org/custom-model-7b
   template: custom
   family: custom
   ```

4. **Create Training Script**:
   ```bash
   # scripts/train/run_sft_custom.sh
   ```

## Known Differences from Qwen3

### Tokenizer

| Aspect | Qwen3 | OLMo3 | Status |
|--------|-------|-------|--------|
| Implementation | HF Standard | Custom (expected) | To Be Confirmed |
| BOS Token ID | 151857 | [TODO] | Pending |
| EOS Token ID | 151858 | [TODO] | Pending |
| PAD Token ID | 151859 | [TODO] | Pending |
| Padding Side | right | [TODO] | Pending |
| Vocab Size | ~152k | [TODO] | Pending |

### Chat Template

| Aspect | Qwen3 | OLMo3 | Status |
|--------|-------|-------|--------|
| Markers | `<\|im_start\|>/<\|im_end\|>` | [TODO] | Pending |
| Role Format | `<\|im_start\|>user` | [TODO] | Pending |
| System Prompt | Integrated | [TODO] | Pending |
| Message Separator | `<\|im_end\|>\n` | [TODO] | Pending |

### Training Characteristics

| Aspect | Qwen3 | OLMo3 | Status |
|--------|-------|-------|--------|
| Learning Rate | 5e-5 (good) | [TODO] | To Be Benchmarked |
| Warmup Ratio | 0.1 | [TODO] | To Be Tested |
| Context Length | 2048 | [TODO] (4096?) | To Be Confirmed |
| bf16 Stability | Stable | [TODO] | To Be Tested |
| Optimal Batch | 4-16 | [TODO] | To Be Determined |

### Inference Characteristics

| Aspect | Qwen3 | OLMo3 | Status |
|--------|-------|-------|--------|
| Speed | [Baseline] | [TODO] | To Be Benchmarked |
| Quality | [Baseline] | [TODO] | To Be Evaluated |
| Memory | 1.3 GB (fp16) | [TODO] | To Be Measured |

## Integration Roadmap

### Phase 1: Setup and Validation (1-2 weeks)

**Goal**: Get OLMo3 model loading and basic tokenization working.

**Tasks**:
- [ ] Wait for public OLMo3 model release
- [ ] Download OLMo3 model and tokenizer
- [ ] Test model loading: `AutoModelForCausalLM.from_pretrained("allenai/OLMo3-...")`
- [ ] Extract tokenizer properties (special tokens, vocab size)
- [ ] Confirm chat template format from model config
- [ ] Test generation: Generate some sample text
- [ ] Document findings in notes.py

**Deliverables**:
- Working model loading code
- Confirmed special tokens and template
- OLMoAdapter.tokenizer_quirks() updated
- OLMoAdapter.default_template updated

### Phase 2: Integration (1-2 weeks)

**Goal**: Integrate OLMo3 into training pipeline.

**Tasks**:
- [ ] Update `src/olmo3_poc/adapter.py` with real values
- [ ] Update `src/olmo3_poc/conversion.py` with actual conversion logic
- [ ] Update `validate_environment()` to test real OLMo3
- [ ] Create/update LLaMA-Factory config for OLMo3
- [ ] Test basic training on small dataset (1k examples)
- [ ] Debug any training issues
- [ ] Verify model saves correctly

**Deliverables**:
- Completed OLMoAdapter class
- Working training configuration
- Successful training on test dataset
- Checkpoint saving and loading

### Phase 3: Training Benchmarking (2-3 weeks)

**Goal**: Establish baseline performance and optimal hyperparameters.

**Tasks**:
- [ ] Run baseline training (current hyperparameters)
- [ ] Record training time, GPU memory, final metrics
- [ ] Benchmark learning rate (1e-5, 5e-5, 1e-4)
- [ ] Test different warmup strategies
- [ ] Evaluate gradient accumulation impact
- [ ] Test mixed precision (bf16 vs fp16)
- [ ] Determine optimal batch size
- [ ] Compare against Qwen3 baseline

**Deliverables**:
- Recommended hyperparameter settings
- Performance benchmarks vs Qwen3
- Training curves and logs
- Best checkpoint identified

### Phase 4: Evaluation (1-2 weeks)

**Goal**: Assess model quality on IE tasks.

**Tasks**:
- [ ] Run inference on test set
- [ ] Compute entity F1, relation F1, KV accuracy
- [ ] Compare metrics against Qwen3
- [ ] Evaluate speed and throughput
- [ ] Measure inference memory usage
- [ ] Identify any model-specific issues
- [ ] Document quirks and workarounds

**Deliverables**:
- Complete evaluation report
- Metric comparisons
- Inference guide for OLMo3
- Known issues and solutions

### Phase 5: Documentation (1 week)

**Goal**: Update all documentation for OLMo3.

**Tasks**:
- [ ] Update this document (olmo3_extension.md) with findings
- [ ] Update README.md with OLMo3 usage
- [ ] Create OLMo3 troubleshooting guide
- [ ] Add OLMo3 examples to examples/ directory
- [ ] Document performance characteristics
- [ ] Record recommended hardware and settings
- [ ] Update architecture.md with OLMo3 details

**Deliverables**:
- Complete OLMo3 documentation
- Usage examples
- Performance guide
- Troubleshooting guide

## TODO Checklist

High-priority items for OLMo3 completion:

### Critical Path
- [ ] OLMo3 model officially released
- [ ] Download and test model loading
- [ ] Extract tokenizer special tokens
- [ ] Confirm chat template format
- [ ] Complete OLMoAdapter implementation
- [ ] Successful test training

### Recommended Path
- [ ] Run full hyperparameter benchmarking
- [ ] Compare against Qwen3 baseline
- [ ] Document performance characteristics
- [ ] Create OLMo3-specific examples
- [ ] Update all documentation

### Nice-to-Have
- [ ] Quantization support (GPTQ, AWQ)
- [ ] Multi-GPU benchmarking
- [ ] Custom evaluation metrics for OLMo3
- [ ] Integration with evaluation framework

## Effort Estimate

**Total Effort**: 6-8 weeks (calendar time)

**Breakdown**:
- Phase 1 (Setup): 1-2 weeks
- Phase 2 (Integration): 1-2 weeks
- Phase 3 (Benchmarking): 2-3 weeks
- Phase 4 (Evaluation): 1-2 weeks
- Phase 5 (Documentation): 1 week

**Critical Path**: Waiting for model release (not under our control)

**Parallel Work**: Phases 1-2 can start after model availability

## Dependencies and Prerequisites

### Required for OLMo3 Support

1. **Official Model Release**
   - Status: Not yet released (as of April 2026)
   - Impact: Blocks all testing
   - Timeline: Check AI2/OLMo3 project

2. **Transformers Library Update**
   - May need recent version for OLMo3 support
   - Status: Likely available after model release
   - Action: `pip install --upgrade transformers`

3. **CUDA/PyTorch Compatibility**
   - OLMo3 may have specific requirements
   - Status: TBD after release
   - Action: Check model documentation

### Hardware Requirements

**Minimum**:
- GPU with 16GB+ VRAM (similar to Qwen3)
- 32GB system RAM
- 50GB storage for model

**Recommended**:
- GPU with 40GB+ VRAM (for larger variants)
- 64GB system RAM
- Fast SSD storage

## Troubleshooting OLMo3

### Model Loading Fails

```python
from transformers import AutoModelForCausalLM

try:
    model = AutoModelForCausalLM.from_pretrained("allenai/OLMo3-...")
except Exception as e:
    # Check: Does model exist on HuggingFace?
    # Check: Do you have HF_TOKEN set?
    # Check: Is your internet connection working?
    print(f"Error: {e}")
```

**Solutions**:
1. Check model is released and available
2. Set HF_TOKEN: `export HF_TOKEN=<your_token>`
3. Clear cache: `rm -rf ~/.cache/huggingface/`
4. Try with trust_remote_code: `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`

### Training Fails with OLMo3

```bash
# Check if model is properly initialized
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('allenai/OLMo3-...')"

# Check tokenizer
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('allenai/OLMo3-...')"

# Check if LLaMA-Factory recognizes model
llamafactory-cli train configs/sft/olmo3_poc_sft.yaml --model_name_or_path allenai/OLMo3-... --dry_run
```

### Inference Quality Poor

**Possible causes**:
1. Model not properly fine-tuned (check training logs)
2. Different prompt format required
3. Different temperature/generation parameters needed
4. Model architecture differences

**Debug**:
1. Generate with base model (no fine-tuning)
2. Try different temperature (0.1 to 1.0)
3. Check prompt format matches model's expected format
4. Verify schema-conditioned prompt is correct

## Success Criteria

OLMo3 support is considered successful when:

1. **Integration** ✓
   - Model loads and initializes
   - Training completes without errors
   - Checkpoints save and load correctly

2. **Performance** ✓
   - Training time comparable to Qwen3
   - Memory usage acceptable
   - No numerical instabilities

3. **Quality** ✓
   - IE metrics comparable to Qwen3
   - No systematic failures on specific tasks
   - Inference works reliably

4. **Documentation** ✓
   - Complete integration guide
   - Hyperparameter recommendations
   - Troubleshooting guide
   - Example training runs

5. **Community Ready** ✓
   - Easy to use for researchers
   - Clear setup instructions
   - Reproducible results
   - Good performance

## Future Considerations

### OLMo3 Variants

Once OLMo3 is released, there may be multiple sizes:
- OLMo3-1B
- OLMo3-7B
- OLMo3-32B
- OLMo3-1T

**Action**: Repeat integration for each variant as released

### Quantization Support

Consider adding quantization support for OLMo3:
- GPTQ (8-bit quantization)
- AWQ (Activation-aware Quantization)
- Benefits: Faster inference, lower memory

### Mixture of Experts (MoE)

If OLMo3 uses MoE architecture:
- Different training considerations
- Specialized fine-tuning approaches
- Different inference requirements

### Multilingual Support

If OLMo3 supports multiple languages:
- Extend dataset policy for multilingual data
- Add language-specific evaluation
- Support schema-conditioned extraction in multiple languages

## References

- **OLMo3 Project**: https://allenai.org/ollm
- **OLMo3 Models**: https://huggingface.co/allenai (when available)
- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **GoLLIE Framework**: https://arxiv.org/abs/2310.03144

## Contact and Updates

For questions about OLMo3 integration:

- Check `src/olmo3_poc/notes.py` for status
- Review this document for latest updates
- Contact research team for current blockers
- Monitor OLMo3 release timeline

---

**Document Status**: Active, To Be Updated  
**Last Update**: April 2026  
**Next Review**: When OLMo3 is publicly released  
**Maintainer**: Research Team
