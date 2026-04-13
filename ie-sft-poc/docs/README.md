# Documentation Index

Complete documentation for the IE-SFT-PoC (Information Extraction SFT Proof-of-Concept) repository.

## Core Documentation

### [architecture.md](architecture.md)
System design and architectural overview.

**Topics covered:**
- System components and their responsibilities
- Module dependency graph
- Data flow pipeline (from raw data to trained model)
- Design patterns and extension points
- Key architectural decisions and rationale
- Integration points between modules

**For:** Understanding how the system works end-to-end, extending the codebase, debugging complex issues.

### [canonical_schema.md](canonical_schema.md)
Complete specification of the canonical information extraction schema.

**Topics covered:**
- Full JSON schema definition with field descriptions
- Three task types: KV (key-value), Entity, Relation
- Schema-conditioned extraction concept
- Examples for each task type with annotations
- Validation rules and constraints
- How to add new task types or modify schema
- GoLLIE-inspired design rationale

**For:** Understanding data formats, normalizing new datasets, validating data quality, implementing custom extractors.

### [dataset_policy.md](dataset_policy.md)
Dataset licensing, compliance, and onboarding procedures.

**Topics covered:**
- License sensitivity rules and compliance checklist
- Dataset categories: default-enabled, reference-only, restricted
- InstructIE (default): How to download and use
- GoLLIE reference (design inspiration only): What NOT to use for training
- IEPile (optional): Opt-in procedure with license considerations
- Internal datasets: Separate access and onboarding path
- How to add new datasets with proper metadata
- Legal and licensing considerations

**For:** Understanding which datasets can be used, setting up new datasets, ensuring compliance.

### [training_flow.md](training_flow.md)
End-to-end training pipeline walkthrough.

**Topics covered:**
- Prerequisites and environment setup
- Complete data preparation pipeline (step-by-step)
- LLaMA-Factory configuration explained
- Training execution and monitoring
- Evaluation methodology and metrics
- Inference and deployment
- LoRA weight merging for production
- Hyperparameter tuning strategies
- Multi-GPU training setup
- Common issues and troubleshooting

**For:** Running training experiments, understanding training mechanics, tuning hyperparameters, evaluating models.

### [olmo3_extension.md](olmo3_extension.md)
OLMo3 proof-of-concept status and integration roadmap.

**Topics covered:**
- Current status: experimental PoC stage
- What works now vs. what needs to be done
- Implementation checklist (5 phases)
- Known differences from Qwen3 (tokenizer, template, training)
- Adapter pattern explanation and extension approach
- Timeline and effort estimates
- How to add OLMo3 as a full training target
- Blocking dependencies and prerequisites

**For:** Understanding OLMo3 integration status, planning OLMo3 work, implementing model adapters.

## Additional Resources

### README Files

- **[../README.md](../README.md)** - Main project README with quickstart guide
- **[../configs/README.md](../configs/README.md)** - Configuration files explanation
- **[../scripts/README.md](../scripts/README.md)** - Scripts and utility tools
- **[../data/README.md](../data/README.md)** - Data directory structure and pipeline

### Example Files

- **examples/canonical_samples/** - Sample records in canonical format
  - `kv_sample.json` - Key-value extraction examples
  - `entity_relation_sample.json` - Entity and relation extraction examples
  - `unified_sample.json` - Combined multi-task examples

- **examples/prompts/** - Example prompts for different IE tasks

## Quick Navigation by Task

### I want to...

**...understand what this project does**
1. Start with [../README.md](../README.md) "Overview" section
2. Read [architecture.md](architecture.md) for system design
3. Look at examples in `examples/canonical_samples/`

**...download and prepare data for training**
1. Read [../scripts/README.md](../scripts/README.md) "Download Scripts" section
2. Follow pipeline in [../data/README.md](../data/README.md)
3. Validate with examples in [canonical_schema.md](canonical_schema.md)

**...train a model**
1. Follow quickstart in [../README.md](../README.md) sections 1-6
2. Read [training_flow.md](training_flow.md) for detailed walkthrough
3. Check [../configs/README.md](../configs/README.md) for hyperparameter tuning

**...evaluate or do inference**
1. See [training_flow.md](training_flow.md) "Evaluation Methodology" and "Inference" sections
2. Check [../scripts/README.md](../scripts/README.md) training scripts section

**...add a new dataset**
1. Review [dataset_policy.md](dataset_policy.md) for licensing
2. See [../scripts/README.md](../scripts/README.md) for preprocessing scripts
3. Check [../data/README.md](../data/README.md) "Adding a New Dataset"
4. Validate against [canonical_schema.md](canonical_schema.md)

**...add a new model family**
1. Understand adapter pattern in [olmo3_extension.md](olmo3_extension.md)
2. Check model configs in [../configs/README.md](../configs/README.md)
3. See [architecture.md](architecture.md) "Extension Points"

**...work on OLMo3 integration**
1. Read [olmo3_extension.md](olmo3_extension.md) for complete status
2. Check [architecture.md](architecture.md) for module structure
3. Reference OLMo3 adapter code in `src/olmo3_poc/`

**...troubleshoot an issue**
1. Check [training_flow.md](training_flow.md) "Troubleshooting" section
2. Review [../README.md](../README.md) "Troubleshooting"
3. Validate data with [canonical_schema.md](canonical_schema.md) validation rules
4. Check environment with `python scripts/utils/env_check.py`

## Documentation Standards

All documentation follows these conventions:

- **Markdown formatting**: Headers, code blocks, tables, links
- **Code examples**: Bash commands prefixed with `$`, Python with `python`
- **File paths**: Absolute or relative from project root
- **Cross-references**: Use relative markdown links `[text](path.md)`
- **Update frequency**: Docs updated with major feature additions
- **Version**: Last updated dates included in footer

## Contributing Documentation

When adding new features:

1. Update relevant documentation file
2. Add examples to `examples/` if applicable
3. Update index (this file) if creating new doc
4. Ensure all links are correct
5. Check markdown formatting with: `python -m markdown doc.md`
6. Run example commands to verify accuracy

## External References

### Key Resources

- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **HuggingFace Models**: https://huggingface.co/models
- **Qwen Models**: https://huggingface.co/Qwen
- **InstructIE Dataset**: https://huggingface.co/datasets/
- **OLMo3**: https://allenai.org/ollm
- **GoLLIE Framework**: https://arxiv.org/abs/2310.03144

### Relevant Papers

- Information Extraction with LLMs: Recent research in IE with instruction-tuning
- GoLLIE: "Generalizable and Scalable Information Extraction with Code-Savy Language Models"
- Unified Information Extraction: Work on multi-task IE with single models

## Glossary

**Terms used throughout documentation:**

- **Canonical Schema**: Unified JSON format supporting KV, entity, and relation extraction
- **IE**: Information Extraction (extracting structured data from text)
- **KV Extraction**: Key-Value extraction (structured field-value pairs)
- **Entity**: Named entity with type and span location
- **Relation**: Connection between two entities with relation type
- **SFT**: Supervised Fine-Tuning (training with labeled examples)
- **LoRA**: Low-Rank Adaptation (efficient fine-tuning method)
- **Interim**: Normalized canonical format (intermediate in pipeline)
- **Split**: Train/dev/test division of data
- **Checkpoint**: Saved model state during/after training
- **Merged Model**: Model with LoRA weights merged into base model

## Feedback and Updates

Documentation is maintained to reflect current system state. If you find:

- **Outdated information**: Note the date and context
- **Broken links**: Check relative paths and capitalization
- **Missing topics**: Suggest additions via issue or PR
- **Unclear explanations**: Provide specific section and question

---

**Last Updated**: April 2026  
**Documentation Version**: 1.0  
**Scope**: Complete system documentation for IE-SFT-PoC
