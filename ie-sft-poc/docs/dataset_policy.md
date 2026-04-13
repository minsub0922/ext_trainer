# Dataset Policy and Compliance

Guidelines for dataset selection, licensing, compliance, and onboarding procedures for IE-SFT-PoC.

## Overview

IE-SFT-PoC uses datasets from multiple sources, each with distinct licensing and usage terms. This document defines:

1. Which datasets are enabled by default (safe to use)
2. Which datasets are reference-only (design inspiration)
3. Which datasets require explicit opt-in (special licensing)
4. How to add new datasets with proper compliance

**Core Principle**: Respect intellectual property and licensing terms. When in doubt, check with legal/compliance before using a dataset.

## Dataset Categories

### Category 1: Default-Enabled Datasets

Datasets that are publicly available, properly licensed, and safe to download and use for training without additional permissions.

#### InstructIE

**Status**: ✓ Default enabled, actively used

**Description**: Large-scale instruction-tuned dataset for information extraction tasks

**Source**: HuggingFace Hub  
**URL**: https://huggingface.co/datasets/...

**License**: Creative Commons Attribution 4.0 (CC-BY-4.0)

**Size**: ~80,000 examples (approximate)

**Task Types Covered**:
- Key-value extraction
- Named entity recognition
- Relation extraction

**Download**:
```bash
python scripts/download/download_instructie.py
```

**Usage**:
```bash
python scripts/preprocess/normalize_instructie.py
python scripts/preprocess/validate_canonical_dataset.py --input data/interim/instructie/
```

**Why This Dataset**:
- Public, permissive license (CC-BY-4.0)
- Large and diverse (multiple task types)
- Well-maintained on HuggingFace
- Standard benchmark for IE research

**Compliance Notes**:
- ✓ Can be downloaded freely
- ✓ Can be used for research and commercial purposes
- ✓ Safe to include in version control (smaller subsets)
- ✓ No special permissions needed
- Requires attribution (cite dataset in publications)

### Category 2: Reference-Only Datasets

Datasets or frameworks used for design inspiration and schema reference only. NOT to be used as training data.

#### GoLLIE Framework and Schemas

**Status**: Reference only, NOT a training dependency

**Description**: Generalizable and Scalable Information Extraction framework with structured schemas

**Source**: Academic research  
**Paper**: https://arxiv.org/abs/2310.03144

**License**: Varies (check original publication)

**What We Use From GoLLIE**:
- Schema design principles
- Task structure inspiration
- Multi-task extraction concepts
- Schema-conditioned extraction idea

**What We DO NOT Use**:
- GoLLIE training data
- GoLLIE extraction examples (as training data)
- GoLLIE code (we implement our own)

**Why This Policy**:
- GoLLIE is a reference framework, not a dataset
- We adapted the concepts, not the data
- Canonical schema is inspired by, not copied from, GoLLIE
- Our implementation is independent

**Reference Download** (optional):
```bash
python scripts/download/download_reference_gollie_assets.py
```

**Usage**:
- ✓ Reading papers and schema documentation
- ✓ Comparing schema designs
- ✓ Understanding multi-task extraction
- ✗ NOT for training models
- ✗ NOT for testing on GoLLIE data

**Compliance Notes**:
- Reference only, no training data from this source
- Schemas and concepts are publicly published (fair use for reference)
- Proper attribution in documentation (done in README)

### Category 3: Restricted/Opt-In Datasets

Datasets available but requiring explicit configuration and understanding of licensing implications.

#### IEPile (Optional)

**Status**: NOT enabled by default, opt-in only

**Description**: Large-scale collection of IE examples from multiple sources

**Source**: Academic research

**Why Restricted**:
- Composed from multiple sources with varying licenses
- Requires careful licensing review per subset
- May have commercial usage restrictions
- Dataset composition not fully transparent

**How to Enable** (if approved by legal):
1. Obtain dataset and place in `data/raw/iepile/`
2. Create `configs/datasets/iepile.yaml` with licensing notes
3. Create `scripts/preprocess/normalize_iepile.py`
4. Add to `data/metadata/dataset_registry.yaml`:
   ```yaml
   iepile:
     name: "IEPile"
     enabled: false  # Keep disabled by default
     requires_approval: true
     contact: "compliance@internal"
   ```
5. Document licensing in this file
6. Get approval before using

**Compliance Checklist** (before enabling):
- [ ] Legal review completed
- [ ] All sub-dataset licenses documented
- [ ] Attribution plan in place
- [ ] Commercial usage rights confirmed
- [ ] Approval obtained from legal team

## Internal Datasets

Datasets internal to your organization require special handling.

### Onboarding Process for Internal Data

1. **Request Access**:
   - Contact internal data administrators
   - Provide use case and justification
   - Sign data use agreements if required

2. **Dataset Configuration**:
   - Receive dataset location (path or API endpoint)
   - Create `configs/datasets/internal_*.yaml`
   - Document in internal wiki (not in this public doc)

3. **Setup**:
   - Create normalization script in `scripts/preprocess/`
   - Add to `data/metadata/internal_registry.yaml` (separate from main registry)
   - Validate against canonical schema

4. **Audit Trail**:
   - Log access and usage
   - Track model checkpoints trained on internal data
   - Keep internal data separate from public workflows

### Example Internal KV Dataset

**Status**: Example/template configuration (not enabled)

**Files**:
- `configs/datasets/internal_kv_example.yaml`
- `scripts/preprocess/build_internal_kv_template.py`

**Purpose**: Shows how to set up internal datasets

**Enable When**:
- You have approved internal data
- Legal/compliance review completed
- Data isolation procedures in place

## Compliance Checklist

Use this checklist when adding any new dataset:

### For Public Datasets:
- [ ] License identified and documented
- [ ] License is permissive (CC-BY, MIT, Apache 2.0, etc.)
- [ ] License allows research and commercial use
- [ ] Attribution requirements documented
- [ ] No restrictive terms on modifications
- [ ] No trademark/brand restrictions
- [ ] Safe to share with collaborators

### For Academic Datasets:
- [ ] Paper/publication has clear license
- [ ] Academic use explicitly allowed
- [ ] Commercial restrictions (if any) documented
- [ ] Dataset stable and maintained
- [ ] Citation requirements documented
- [ ] No export control restrictions

### For Internal Datasets:
- [ ] Data classification confirmed
- [ ] Access control in place
- [ ] Data use agreement signed
- [ ] Audit logging enabled
- [ ] Isolation procedures documented
- [ ] Compliance review completed
- [ ] Security requirements met

## Adding a New Dataset

### Step 1: License Review

1. Identify dataset source and license
2. Review license terms:
   - Is research use allowed? ✓
   - Is commercial use allowed? ✓
   - Are modifications allowed? ✓
   - Is redistribution restricted? (Check!)
   - Are there trademark restrictions? (Check!)

3. Classify into category (default/reference/opt-in/internal)

4. Document in this file:
   ```markdown
   #### Dataset Name
   
   **Status**: [default enabled / reference only / opt-in / internal]
   **License**: [License identifier]
   **Source**: [URL or internal path]
   **Tasks**: [kv / entity / relation]
   **Size**: [approximate record count]
   **Restrictions**: [Any usage restrictions]
   **Compliance**: [Checklist items]
   ```

### Step 2: Create Dataset Configuration

Create `configs/datasets/new_dataset.yaml`:

```yaml
### New Dataset Configuration
dataset_name: "new_dataset"
source: "https://huggingface.co/datasets/..."
description: "Brief description of dataset"

task_types:
  - kv          # If dataset has KV tasks
  - entity      # If dataset has entity tasks
  - relation    # If dataset has relation tasks

license: "cc-by-4.0"  # Or appropriate license
size_approx: 50000    # Approximate record count

split_ratio:
  train: 0.8
  dev: 0.1
  test: 0.1

format: "original_format"  # instructie, json, csv, etc.
enabled: true             # false for opt-in datasets

notes: |
  Additional notes about dataset:
  - Source details
  - Known issues
  - Processing notes
  - Attribution requirements
```

### Step 3: Create Normalization Script

Create `scripts/preprocess/normalize_new_dataset.py`:

```python
#!/usr/bin/env python
"""Normalize new_dataset to canonical IE format."""

import json
from pathlib import Path
from src.datasets.unified.validator import validate_dataset

def normalize_new_dataset(input_dir: str, output_dir: str) -> None:
    """Convert new_dataset to canonical format.
    
    Args:
        input_dir: Directory with raw dataset files
        output_dir: Output directory for canonical format
    """
    # 1. Load raw data
    # 2. Map to canonical format
    # 3. Write JSONL
    # 4. Validate output
    
    output_path = Path(output_dir) / "train.jsonl"
    report = validate_dataset(output_path, strict=True)
    
    if not report.is_valid:
        raise ValueError(f"Validation failed: {report}")
    
    print(f"Normalized {report.valid} records to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate", action="store_true")
    
    args = parser.parse_args()
    normalize_new_dataset(args.input_dir, args.output_dir)
```

### Step 4: Update Dataset Registry

Add to `data/metadata/dataset_registry.yaml`:

```yaml
datasets:
  # ... existing datasets ...
  
  new_dataset:
    name: "New Dataset"
    source: "https://..."
    task_types: [entity, relation]
    license: "cc-by-4.0"
    size_approx: 50000
    enabled: true  # or false for opt-in
    notes: "Attribution requirements documented above"
```

### Step 5: Document Licensing

1. Add entry to this file (dataset_policy.md)
2. Include in README acknowledgments (if public)
3. Create `data/raw/new_dataset/LICENSE` file
4. Document attribution in `data/raw/new_dataset/ATTRIBUTION.txt`

### Step 6: Test and Validate

```bash
# Test download (if applicable)
python scripts/download/download_new_dataset.py

# Test normalization
python scripts/preprocess/normalize_new_dataset.py \
  --input-dir data/raw/new_dataset/ \
  --output-dir data/interim/new_dataset/ \
  --validate

# Validate output
python scripts/preprocess/validate_canonical_dataset.py \
  --input data/interim/new_dataset/train.jsonl \
  --strict

# Check statistics
python scripts/utils/print_dataset_stats.py \
  --input data/interim/new_dataset/train.jsonl
```

### Step 7: Update Documentation

1. Update `README.md` (if public dataset)
2. Update `docs/dataset_policy.md` (this file)
3. Update `configs/README.md` if needed
4. Add example records if new format

## License Identifiers

### Permissive Licenses (Safe for Training)

- **CC-BY-4.0**: Creative Commons Attribution
  - ✓ Research use
  - ✓ Commercial use
  - ✓ Modifications
  - ✗ Requires attribution

- **CC-BY-SA-4.0**: CC Attribution-ShareAlike
  - ✓ Research use
  - ✓ Commercial use
  - ✓ Modifications
  - ✗ Share-alike requirement (rare in ML)

- **MIT**: MIT License
  - ✓ Research use
  - ✓ Commercial use
  - ✓ Modifications
  - ✓ Permissive

- **Apache-2.0**: Apache License 2.0
  - ✓ Research use
  - ✓ Commercial use
  - ✓ Modifications
  - ✓ Permissive

- **GPL**: GNU Public License
  - ⚠ Check version (v2 vs v3)
  - ✓ Research use
  - ✗ Copyleft may restrict deployment
  - (Usually avoid for ML unless approved)

### Restrictive Licenses (Review Required)

- **CC-BY-NC**: Attribution-NonCommercial
  - ✓ Research use
  - ✗ Commercial use (restricted!)
  - ⚠ Clarify "research" vs "commercial"

- **Custom/Proprietary**: 
  - ⚠ Always review with legal
  - Check terms individually
  - May require licensing agreement

### No License (Contact Required)

- If no explicit license, contact dataset owner
- Ask about usage rights
- Get written permission if needed

## Attribution and Citation

### For Academic Datasets

Include proper citation in publications:

```bibtex
@article{dataset_authors,
  title={Dataset Name},
  author={...},
  journal={...},
  year={...}
}
```

### For HuggingFace Datasets

Include in README:

```markdown
### Datasets

This project uses the following datasets:

- **InstructIE**: [HuggingFace Dataset Card](https://huggingface.co/datasets/)
  - License: CC-BY-4.0
  - Citation: [Original paper](https://...)
```

### For Internal Datasets

Document in internal wiki, not in public documentation.

## Prohibited Datasets

Do NOT use without explicit legal approval:

1. **Copyrighted text without permission**
   - Books, articles, web scrapes
   - Requires explicit license or permission

2. **Personal data without consent**
   - Names, contact info, medical records
   - Requires GDPR/privacy compliance

3. **Proprietary datasets**
   - Competitor data, internal corporate data
   - Requires ownership and permission

4. **Restricted export datasets**
   - May have government export restrictions
   - Check ITAR, EAR classifications

5. **Datasets with unknown license**
   - Always clarify before use
   - Contact dataset owners if unclear

## Troubleshooting Dataset Issues

### Dataset Missing from Registry

```bash
# Check if dataset is in registry
grep "dataset_name" data/metadata/dataset_registry.yaml

# Add to registry if missing
```

### License Questions

- Check dataset's official documentation first
- Look for LICENSE file in dataset repository
- Check HuggingFace dataset card
- Contact dataset owners if unclear
- When in doubt, ask legal/compliance

### Compliance Review Needed

Contact:
- Internal compliance team
- Legal department
- Data governance committee

## Updates and Policy Changes

This policy document is maintained as datasets and regulations evolve.

**Policy Version**: 1.0  
**Last Updated**: April 2026  
**Next Review**: [Schedule regular review dates]

Changes are tracked in:
- Git history (for this file)
- Internal compliance documentation
- Dataset registry updates

---

**For questions about dataset usage, licensing, or compliance, contact the research team or compliance department.**
