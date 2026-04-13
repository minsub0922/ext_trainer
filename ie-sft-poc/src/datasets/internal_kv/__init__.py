"""Internal K/V dataset onboarding module.

This module handles the creation, parsing, and conversion of internal K/V datasets
for annotation workflows. It supports:

- Template building: Creating empty annotation templates from raw data
- Parsing: Reading annotated K/V data back from JSONL format
- Conversion: Enriching and validating K/V records into canonical format

The workflow typically follows:
1. Start with raw CSV or JSONL data
2. Build annotation templates using template_builder
3. Annotate the templates (done externally)
4. Parse annotated data using parser
5. Convert to canonical format using converter
"""

__version__ = "0.1.0"
__all__ = [
    "KVTemplateConfig",
    "build_kv_template",
    "parse_kv_annotation",
    "parse_kv_file",
    "validate_kv_annotations",
    "enrich_kv_record",
    "convert_kv_dataset",
]
