#!/usr/bin/env python3
"""Download and generate GoLLIE reference assets.

This script creates reference documentation about GoLLIE (schema-conditioned IE)
without downloading actual GoLLIE models or datasets by default.

It generates a reference document with:
- Links to GoLLIE papers and repositories
- License information
- Usage notes for schema-conditioned prompting
- Marked as "reference only, not default enabled"
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.common.logging_utils import get_logger
from src.common.paths import DATA_RAW

logger = get_logger(__name__)


GOLLIE_REFERENCE_CONTENT = """# GoLLIE Reference - Schema-Conditioned Information Extraction

## Overview

GoLLIE (Generalized Open-ended Language Learning for Information Extraction) is an approach
to information extraction that uses schema definitions to guide extraction tasks.

**Note**: This is a reference-only document. GoLLIE models and datasets are NOT downloaded
by default. This project implements schema-conditioned IE patterns inspired by GoLLIE.

## References

### Papers
- Xie, T., et al. "Unified Structure Generation for Universal Information Extraction as a Byte-level Generation Task"
  - arXiv: https://arxiv.org/abs/2304.07547
  - Published in: Proceedings of ACL 2023

### Official Repository
- GitHub: https://github.com/zjunlp/GoLLIE
- License: Apache 2.0

## Key Concepts

### Schema-Conditioned Extraction
GoLLIE frames IE as a structured prediction task where:
1. A schema definition specifies the extraction targets (fields, entity types, relations)
2. LLMs are prompted with the schema and input text
3. Output is formatted as structured JSON matching the schema

### Schema Definition Format
```json
{
  "kv": ["field1", "field2", ...],
  "entity": ["TYPE1", "TYPE2", ...],
  "relation": ["rel_type1", "rel_type2", ...]
}
```

### Supported Task Types
1. **Key-Value (KV) Extraction**: Extract predefined fields (slot-filling)
2. **Named Entity Recognition (NER)**: Extract entities of specified types
3. **Relation Extraction (RE)**: Extract relations between entities
4. **Unified Extraction**: Combine multiple task types in single output

## Implementation in IE SFT PoC

This project implements schema-conditioned IE using:

### Modules
- `src/datasets/gollie_reference/task_reference.py`: Task definition patterns
- `src/datasets/gollie_reference/schema_patterns.py`: Prompt generation for structured extraction

### Key Features
- Schema-aware prompt building
- Unified extraction across task types
- Instruction-tuning friendly formats
- Support for multi-lingual data

## Usage Examples

### Basic Schema Definition
```python
from src.common.schema import SchemaDefinition

schema = SchemaDefinition(
    kv=["name", "email", "phone"],
    entity=["PERSON", "ORGANIZATION"],
    relation=["works_for"]
)
```

### Generating Extraction Prompts
```python
from src.datasets.gollie_reference.schema_patterns import (
    build_kv_extraction_prompt,
    build_entity_extraction_prompt
)

text = "John Smith works at Google"

# KV extraction prompt
kv_prompt = build_kv_extraction_prompt(
    text,
    ["name", "company"]
)

# Entity extraction prompt
entity_prompt = build_entity_extraction_prompt(
    text,
    ["PERSON", "ORGANIZATION"]
)
```

## Output Format

All extraction outputs follow a consistent JSON structure:

### Key-Value Output
```json
{
  "kv": {
    "field1": "value1",
    "field2": null,
    ...
  }
}
```

### Entity Output
```json
{
  "entity": [
    {
      "text": "entity_text",
      "type": "ENTITY_TYPE",
      "start": 0,
      "end": 11
    },
    ...
  ]
}
```

### Relation Output
```json
{
  "relation": [
    {
      "head": "head_entity",
      "head_type": "ENTITY_TYPE",
      "relation": "RELATION_TYPE",
      "tail": "tail_entity",
      "tail_type": "ENTITY_TYPE"
    },
    ...
  ]
}
```

### Unified Output
Combines all three formats in single structure:
```json
{
  "kv": {...},
  "entity": [...],
  "relation": [...]
}
```

## Citation

If using GoLLIE concepts in research, please cite:

```bibtex
@inproceedings{xie2023gollie,
  title={Unified Structure Generation for Universal Information Extraction as a Byte-level Generation Task},
  author={Xie, Tianze and others},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2023}
}
```

## License

GoLLIE original work: Apache 2.0
This implementation: Follows project license

## Related Resources

- SchemaOracle: https://github.com/liuyuanxu/SchemaOracle
- UniversalIE: https://github.com/universal-ie/UIE
- Information Extraction Surveys: https://github.com/roomylee/information-extraction-papers

---

Generated: {timestamp}
Status: Reference only - not for production use without understanding limitations
"""


def create_reference_document(output_dir: Path, dry_run: bool = False) -> dict[str, str]:
    """Create GoLLIE reference documentation.

    Args:
        output_dir: Directory to save reference document
        dry_run: If True, don't write files, just return content

    Returns:
        Dictionary with output file paths and status
    """
    logger.info(f"Creating GoLLIE reference document (dry_run={dry_run})")

    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Generate content with timestamp (safely replace placeholder without interpreting braces)
    content = GOLLIE_REFERENCE_CONTENT.replace("{timestamp}", datetime.now().isoformat())

    # Create reference document
    doc_path = output_dir / "GOLLIE_REFERENCE.md"
    if not dry_run:
        logger.info(f"Writing reference document to {doc_path}")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        logger.info(f"[DRY RUN] Would write reference document to {doc_path}")

    # Create metadata JSON
    metadata = {
        "name": "gollie_reference",
        "description": "Reference documentation for schema-conditioned IE",
        "status": "reference_only",
        "enabled_by_default": False,
        "created": datetime.now().isoformat(),
        "task_types": ["kv", "entity", "relation", "unified"],
        "languages": ["en", "ko", "zh", "ja"],
        "papers": [
            {
                "title": "Unified Structure Generation for Universal Information Extraction as a Byte-level Generation Task",
                "authors": ["Tianze Xie", "et al."],
                "venue": "ACL 2023",
                "arxiv": "https://arxiv.org/abs/2304.07547"
            }
        ],
        "repositories": [
            {
                "name": "GoLLIE Official",
                "url": "https://github.com/zjunlp/GoLLIE",
                "license": "Apache 2.0"
            }
        ],
        "notes": [
            "This module provides reference patterns inspired by GoLLIE",
            "NOT a full GoLLIE implementation or integration",
            "Useful for schema-conditioned prompt engineering",
            "Support for instruction tuning of IE models"
        ]
    }

    meta_path = output_dir / "gollie_reference_metadata.json"
    if not dry_run:
        logger.info(f"Writing metadata to {meta_path}")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    else:
        logger.info(f"[DRY RUN] Would write metadata to {meta_path}")

    # Create usage guide
    usage_guide = """# Using GoLLIE Reference Patterns

## Quick Start

### 1. Define a Schema
```python
from src.common.schema import SchemaDefinition

schema = SchemaDefinition(
    kv=["name", "position", "company"],
    entity=["PERSON", "ORGANIZATION", "LOCATION"],
    relation=["works_for", "located_in"]
)
```

### 2. Build a Prompt
```python
from src.datasets.gollie_reference.schema_patterns import (
    build_unified_extraction_prompt
)

text = "Alice Johnson is a software engineer at Google, based in Mountain View."
prompt = build_unified_extraction_prompt(text, schema)
print(prompt)
```

### 3. Get Structured Output
The model should return:
```json
{
  "kv": {
    "name": "Alice Johnson",
    "position": "software engineer",
    "company": "Google"
  },
  "entity": [
    {"text": "Alice Johnson", "type": "PERSON", "start": 0, "end": 13},
    {"text": "Google", "type": "ORGANIZATION", "start": 42, "end": 48},
    {"text": "Mountain View", "type": "LOCATION", "start": 59, "end": 72}
  ],
  "relation": [
    {
      "head": "Alice Johnson",
      "head_type": "PERSON",
      "relation": "works_for",
      "tail": "Google",
      "tail_type": "ORGANIZATION"
    },
    {
      "head": "Google",
      "head_type": "ORGANIZATION",
      "relation": "located_in",
      "tail": "Mountain View",
      "tail_type": "LOCATION"
    }
  ]
}
```

## Integration Points

- Use with `src/datasets/internal_kv/` for K/V annotation workflows
- Combine with LLM instruction-tuning for structured extraction
- Export prompts for data collection and annotation
"""

    guide_path = output_dir / "USAGE_GUIDE.md"
    if not dry_run:
        logger.info(f"Writing usage guide to {guide_path}")
        with open(guide_path, "w", encoding="utf-8") as f:
            f.write(usage_guide)
    else:
        logger.info(f"[DRY RUN] Would write usage guide to {guide_path}")

    return {
        "reference_doc": str(doc_path),
        "metadata": str(meta_path),
        "usage_guide": str(guide_path),
        "status": "success",
        "dry_run": dry_run,
    }


def main() -> None:
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Download/generate GoLLIE reference assets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_RAW / "gollie_reference",
        help="Output directory for reference assets (default: data/raw/gollie_reference)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )

    args = parser.parse_args()

    logger.info("Starting GoLLIE reference asset generation")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Dry run: {args.dry_run}")

    try:
        result = create_reference_document(args.output_dir, dry_run=args.dry_run)

        logger.info("GoLLIE reference generation completed successfully")
        logger.info(f"  Generated files:")
        for key, path in result.items():
            if key != "status" and key != "dry_run":
                logger.info(f"    - {key}: {path}")

    except Exception as e:
        logger.error(f"Failed to generate GoLLIE reference: {str(e)}")
        raise


if __name__ == "__main__":
    main()
