"""Generate annotation-ready template datasets for internal K/V onboarding.

This module creates empty K/V annotation templates from raw input data (CSV or JSONL).
The templates are structured as CanonicalIERecord with empty answer.kv values,
ready for annotation.
"""

import csv
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.io import read_jsonl, write_jsonl
from src.common.logging_utils import get_logger
from src.common.schema import Answer, CanonicalIERecord, SchemaDefinition

logger = get_logger(__name__)


@dataclass
class KVTemplateConfig:
    """Configuration for building K/V annotation templates.

    Attributes:
        text_column: Name of the column containing the main text to extract from
        id_column: Optional name of the column containing record IDs
        lang_column: Optional name of the column containing language code
        kv_fields: List of field names for K/V extraction
        source_name: Name of the data source
        default_lang: Default language code if not specified (default: 'ko')
    """

    text_column: str
    id_column: str | None = None
    lang_column: str | None = None
    kv_fields: list[str] = None
    source_name: str = "internal"
    default_lang: str = "ko"

    def __post_init__(self) -> None:
        """Validate config after initialization."""
        if self.kv_fields is None:
            self.kv_fields = []
        if not self.text_column:
            raise ValueError("text_column must be specified")
        if not self.kv_fields:
            raise ValueError("kv_fields must not be empty")


def _read_input_data(input_path: Path | str) -> list[dict[str, Any]]:
    """Read input data from CSV or JSONL file.

    Args:
        input_path: Path to input file (CSV or JSONL)

    Returns:
        List of dictionaries representing records

    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file does not exist
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".jsonl" or input_path.suffix.lower() == ".ndjson":
        logger.info(f"Reading JSONL input from {input_path}")
        return read_jsonl(input_path)

    elif input_path.suffix.lower() == ".csv":
        logger.info(f"Reading CSV input from {input_path}")
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records

    else:
        raise ValueError(
            f"Unsupported file format: {input_path.suffix}. "
            "Supported formats: .jsonl, .ndjson, .csv"
        )


def build_kv_template(
    input_path: Path | str,
    config: KVTemplateConfig,
    output_path: Path | str | None = None,
) -> list[CanonicalIERecord]:
    """Build K/V annotation templates from input data.

    Reads raw input data and creates CanonicalIERecord templates with empty
    answer.kv values (skeleton for annotation).

    Args:
        input_path: Path to input CSV or JSONL file
        config: Configuration for template building
        output_path: Optional path to save templates as JSONL

    Returns:
        List of CanonicalIERecord templates ready for annotation

    Raises:
        ValueError: If config is invalid or data is malformed
        FileNotFoundError: If input file does not exist
    """
    logger.info(f"Building K/V templates from {input_path} with config: {config}")

    # Read input data
    records = _read_input_data(input_path)
    logger.info(f"Loaded {len(records)} records from input")

    # Build templates
    templates = []
    errors = []

    for idx, record_dict in enumerate(records):
        try:
            # Extract required fields
            if config.text_column not in record_dict:
                errors.append(f"Record {idx}: Missing text_column '{config.text_column}'")
                continue

            text = record_dict[config.text_column]
            if not text or not str(text).strip():
                errors.append(f"Record {idx}: Empty text")
                continue

            # Generate or extract ID
            if config.id_column and config.id_column in record_dict:
                rec_id = str(record_dict[config.id_column])
            else:
                rec_id = f"{config.source_name}_{idx}_{uuid.uuid4().hex[:8]}"

            # Extract language
            lang = config.default_lang
            if config.lang_column and config.lang_column in record_dict:
                lang = str(record_dict[config.lang_column]).lower()

            # Create template with empty KV values
            kv_template = {field: None for field in config.kv_fields}

            template_record = CanonicalIERecord(
                id=rec_id,
                text=str(text).strip(),
                lang=lang,
                source=config.source_name,
                task_types=["kv"],
                schema_def=SchemaDefinition(kv=config.kv_fields),
                answer=Answer(kv=kv_template),
            )

            templates.append(template_record)

        except Exception as e:
            errors.append(f"Record {idx}: {str(e)}")
            continue

    if errors:
        logger.warning(f"Encountered {len(errors)} errors during template building:")
        for error in errors[:10]:  # Log first 10 errors
            logger.warning(f"  {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")

    logger.info(
        f"Built {len(templates)} templates from {len(records)} input records "
        f"({len(errors)} errors)"
    )

    # Save if output path specified
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records_to_save = [t.to_canonical_dict() for t in templates]
        count = write_jsonl(records_to_save, output_path)
        logger.info(f"Saved {count} templates to {output_path}")

    return templates


def print_template_stats(templates: list[CanonicalIERecord]) -> dict[str, Any]:
    """Print and return statistics about templates.

    Args:
        templates: List of templates to analyze

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total": len(templates),
        "languages": {},
        "fields_with_values": {field: 0 for field in (templates[0].schema_def.kv if templates else [])},
        "avg_text_length": 0,
    }

    if not templates:
        logger.info("No templates to analyze")
        return stats

    total_length = 0

    for template in templates:
        # Count languages
        lang = template.lang
        stats["languages"][lang] = stats["languages"].get(lang, 0) + 1

        # Count text length
        total_length += len(template.text)

        # Count fields with values
        for field, value in template.answer.kv.items():
            if value is not None:
                stats["fields_with_values"][field] += 1

    stats["avg_text_length"] = round(total_length / len(templates), 2)

    logger.info("Template Statistics:")
    logger.info(f"  Total templates: {stats['total']}")
    logger.info(f"  Languages: {stats['languages']}")
    logger.info(f"  Average text length: {stats['avg_text_length']} characters")
    logger.info(f"  Fields with values: {stats['fields_with_values']}")

    return stats
