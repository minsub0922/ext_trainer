"""Convert internal K/V data to canonical format.

This module enriches and validates K/V records, converting them to the canonical
CanonicalIERecord format for downstream processing.
"""

from pathlib import Path
from typing import Any

from src.common.io import write_jsonl
from src.common.logging_utils import get_logger
from src.common.schema import CanonicalIERecord, MetaInfo

from .parser import parse_kv_file, validate_kv_annotations

logger = get_logger(__name__)


def enrich_kv_record(
    record: CanonicalIERecord,
    source_name: str | None = None,
) -> CanonicalIERecord:
    """Enrich a K/V record with additional metadata.

    Args:
        record: Record to enrich
        source_name: Optional source name to set/override

    Returns:
        Enriched CanonicalIERecord
    """
    # Update source if provided
    if source_name:
        record.source = source_name

    # Ensure metadata is set
    if not record.meta.dataset:
        record.meta.dataset = record.source or "internal_kv"

    # Validate consistency
    if "kv" not in record.task_types and record.answer.kv:
        if "kv" not in record.task_types:
            record.task_types.append("kv")
        logger.debug(f"Record {record.id}: Added 'kv' to task_types")

    return record


def convert_kv_dataset(
    input_path: Path | str,
    output_path: Path | str | None = None,
    source_name: str = "internal_kv",
    split: str = "train",
) -> dict[str, Any]:
    """Convert internal K/V dataset to canonical format.

    Reads annotated K/V JSONL, enriches records, and optionally saves to output.

    Args:
        input_path: Path to input JSONL file with annotated K/V records
        output_path: Optional path to save converted records
        source_name: Name of the data source
        split: Data split name (train/dev/test)

    Returns:
        Dictionary with conversion statistics including:
        - total: Number of records processed
        - converted: Number of successfully converted records
        - errors: List of conversion errors
        - validation_stats: Validation statistics from validate_kv_annotations
    """
    logger.info(f"Converting K/V dataset from {input_path}")

    # Parse input file
    try:
        records = parse_kv_file(input_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"Failed to parse input file: {str(e)}")
        raise

    # Enrich records
    converted_records = []
    errors = []

    for record in records:
        try:
            enriched = enrich_kv_record(record, source_name=source_name)

            # Set split and dataset in metadata
            enriched.meta.split = split
            enriched.meta.dataset = source_name

            converted_records.append(enriched)

        except Exception as e:
            errors.append((record.id if hasattr(record, "id") else "unknown", str(e)))
            logger.warning(f"Failed to convert record: {str(e)}")

    logger.info(f"Converted {len(converted_records)} records "
                f"({len(errors)} errors)")

    # Validate converted records
    validation_stats = validate_kv_annotations(converted_records)

    # Save if output path specified
    saved_count = 0
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records_to_save = [r.to_canonical_dict() for r in converted_records]
        saved_count = write_jsonl(records_to_save, output_path)
        logger.info(f"Saved {saved_count} records to {output_path}")

    stats = {
        "total": len(records),
        "converted": len(converted_records),
        "saved": saved_count,
        "errors": len(errors),
        "error_details": errors[:10] if errors else [],  # First 10 errors
        "validation_stats": validation_stats,
    }

    if errors and len(errors) > 10:
        logger.warning(f"... and {len(errors) - 10} more conversion errors")

    return stats
