# IE SFT Examples Directory

This directory contains example datasets, canonical sample records, and labeling guidelines for the Information Extraction Supervised Fine-Tuning (IE SFT) project.

## Directory Structure

```
examples/
├── canonical_samples/        # Pre-formatted example records in canonical format
│   ├── kv_sample.json       # Key-value extraction examples (Korean)
│   ├── entity_relation_sample.json  # NER and relation extraction examples (English)
│   └── unified_sample.json   # Multi-task examples combining KV, entity, and relation
├── prompts/                  # Annotation guidelines for labelers
│   ├── labeling_prompt_ko.md # Korean annotation guidelines
│   └── labeling_prompt_en.md # English annotation guidelines
└── README.md                # This file
```

## Quick Start

### For Developers
1. Start with `canonical_samples/` to understand the data format
2. Review the schema in `src/common/schema.py` for field definitions
3. Use `src/common/io` utilities to load and validate examples

### For Annotators
1. Read the appropriate labeling guideline:
   - Korean annotators: `prompts/labeling_prompt_ko.md`
   - English annotators: `prompts/labeling_prompt_en.md`
2. Review the canonical samples to see expected output format
3. Follow the guidelines when creating new annotations

## Canonical Samples

### kv_sample.json
Contains 3 examples of key-value extraction from Korean business documents:
- Invoice/business registration information
- Product catalog entry
- Order information

**Use case:** Training models to extract structured information from Korean documents (invoices, receipts, catalogs)

**Schema fields:**
- company_name, business_number, issue_date, amount
- product_name, sku, price, stock_quantity, manufacturer
- order_id, order_datetime, delivery_address, shipping_fee, payment_status

### entity_relation_sample.json
Contains 3 examples of named entity recognition and relation extraction from English news articles:
- Apple Inc. company information
- Elon Musk corporate announcements
- Google company history

**Use case:** Training models to identify entities (PERSON, ORG, LOC, DATE) and relations between them (founded_by, located_in, works_for)

**Entity types:** ORG, PERSON, LOC, DATE

**Relation types:** founded_by, located_in, works_for, announced_at, established_at

### unified_sample.json
Contains 2 examples combining multiple task types:
- Invoice with KV fields, entities, and relations
- Corporate partnership announcement with mixed tasks

**Use case:** Training models to handle multiple extraction tasks simultaneously

**Demonstrates:**
- How KV, entity, and relation tasks coexist in a single record
- Cross-task entity references and consistency
- Mixed language and format handling

## Labeling Guidelines

### labeling_prompt_ko.md
Korean-language annotation guidelines covering:
- KV extraction rules and examples
- Entity types and extraction rules (기본 설명, 예제)
- Relation types and extraction rules
- Common mistakes to avoid
- Quality assurance checklist

### labeling_prompt_en.md
English-language annotation guidelines with:
- Complete task descriptions with examples
- Detailed entity type definitions
- Relation type specifications
- Format specifications for output JSON
- Special cases and edge cases
- Quick reference section

## Data Format

All examples use the canonical format defined in `src/common/schema.py`:

```python
CanonicalIERecord {
    id: str                    # Unique record identifier
    text: str                  # Input text
    lang: str                  # Language code (e.g., "en", "ko")
    source: str                # Data source
    task_types: list[str]      # ["kv", "entity", "relation"]
    schema: SchemaDefinition   # Field/entity/relation type definitions
    answer: Answer             # Extraction results
    meta: MetaInfo             # Metadata
}
```

### CanonicalIERecord Fields

- **id**: Unique identifier (e.g., "kv-sample-001")
- **text**: The input text to extract from
- **lang**: ISO 639-1 language code (e.g., "en", "ko")
- **source**: Source dataset name (e.g., "internal_kv_template")
- **task_types**: List of applicable tasks (["kv"], ["entity", "relation"], etc.)
- **schema**: SchemaDefinition with kv, entity, and relation field names
- **answer**: Answer object containing:
  - **kv**: dict[str, str] - Extracted key-value pairs
  - **entity**: list[EntityAnnotation] - Found entities with type and position
  - **relation**: list[RelationAnnotation] - Relations between entities
- **meta**: MetaInfo with dataset, license, split, and notes

### Answer Structure

**KV Answer:**
```json
{
  "kv": {
    "field_name": "extracted_value",
    "field_name2": "extracted_value2"
  }
}
```

**Entity Answer:**
```json
{
  "entity": [
    {"text": "entity text", "type": "PERSON", "start": 0, "end": 10},
    {"text": "organization", "type": "ORG", "start": 20, "end": 32}
  ]
}
```

**Relation Answer:**
```json
{
  "relation": [
    {
      "head": "entity1",
      "head_type": "PERSON",
      "relation": "works_for",
      "tail": "entity2",
      "tail_type": "ORG"
    }
  ]
}
```

## Validation

To validate example files:

```bash
python -m src.common.schema validate_jsonl_file examples/canonical_samples/kv_sample.json
```

Or use the validation script:

```bash
python scripts/preprocess/validate_canonical_dataset.py examples/canonical_samples/
```

## Using Examples in Training

1. Load examples using utilities from `src/common.io`:
   ```python
   from src.common.io import read_json
   examples = read_json("examples/canonical_samples/kv_sample.json")
   ```

2. Convert to training format using dataset builders in `src/training/`

3. Include in training config for LLaMA-Factory

## Contributing New Examples

When adding new examples:

1. Follow the canonical schema exactly
2. Use appropriate language codes ("en", "ko", etc.)
3. Ensure all task_types have corresponding data in answer
4. Include meaningful metadata (dataset, license, split, notes)
5. Validate using `validate_canonical_dataset.py`
6. Add documentation in this README

### Example Creation Checklist
- [ ] Record has unique ID
- [ ] Text is non-empty and valid
- [ ] Language code is correct
- [ ] task_types match answer content
- [ ] schema defines all used fields/types
- [ ] All extracted values exist in original text (for KV)
- [ ] Entity boundaries are correct (for NER)
- [ ] Relations reference valid entities
- [ ] Metadata is filled in
- [ ] Record validates without errors

## Related Documentation

- **Schema Definition**: `src/common/schema.py`
- **Data I/O Utilities**: `src/common/io.py`
- **Constants**: `src/common/constants.py`
- **Validation Script**: `scripts/preprocess/validate_canonical_dataset.py`
- **Dataset Building**: `src/training/dataset_registry_builder.py`
- **OLMo3 PoC**: `src/olmo3_poc/` (for extending support to new models)

## FAQ

**Q: Can I use examples directly for training?**
A: Yes, after validation. They're in the correct canonical format.

**Q: Do I need to create examples for every language?**
A: No, but having examples in each language helps annotators. At minimum, have guidelines.

**Q: How do I add custom entity/relation types?**
A: Update your dataset's schema definition and regenerate examples. See `SchemaDefinition` in schema.py.

**Q: What if a value appears multiple times in the text?**
A: For KV, extract the most relevant occurrence. For NER, extract all occurrences.

## Support

For questions about the examples or annotation process, refer to:
- The appropriate labeling guideline (Ko or En)
- `src/common/schema.py` for technical details
- Project documentation in the main README

---

Last Updated: March 2024
Project Version: 1.0
