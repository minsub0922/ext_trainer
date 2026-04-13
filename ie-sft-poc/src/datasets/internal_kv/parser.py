"""Parse annotated K/V datasets back from JSONL.

This module reads and validates K/V annotations from JSONL files.
It provides both single-record and batch parsing with validation.
"""

from pathlib import Path
from typing import Any

from src.common.io import read_jsonl
from src.common.logging_utils import get_logger
from src.common.schema import CanonicalIERecord, validate_record

logger = get_logger(__name__)


def parse_kv_annotation(record_dict: dict[str, Any]) -> CanonicalIERecord:
    """Parse a single K/V annotated record from dictionary.

    Args:
        record_dict: Dictionary representation of an annotated record

    Returns:
        Parsed CanonicalIERecord

    Raises:
        ValueError: If record is invalid or malformed
    """
    try:
        # Validate the record first
        is_valid, errors = validate_record(record_dict)
        if not is_valid:
            raise ValueError(f"Record validation failed: {errors}")

        # Parse as CanonicalIERecord
        record = CanonicalIERecord.from_dict(record_dict)

        # Verify task type is 'kv'
        if "kv" not in record.task_types:
            logger.warning(f"Record {record.id}: task_types does not include 'kv'")

        # Verify answer has KV data
        if not record.answer.kv:
            logger.debug(f"Record {record.id}: has empty KV answers")

        return record

    except Exception as e:
        logger.error(f"Failed to parse K/V annotation: {str(e)}")
        raise


def parse_kv_file(path: Path | str) -> list[CanonicalIERecord]:
    """Parse all K/V annotated records from a JSONL file.

    Args:
        path: Path to JSONL file with annotated K/V records

    Returns:
        List of parsed CanonicalIERecord objects

    Raises:
        FileNotFoundError: If file does not exist
    """
    path = Path(path)
    logger.info(f"Parsing K/V annotations from {path}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records_data = read_jsonl(path)
    parsed_records = []
    errors = []

    for idx, record_dict in enumerate(records_data):
        try:
            record = parse_kv_annotation(record_dict)
            parsed_records.append(record)
        except ValueError as e:
            errors.append((idx, str(e)))
            logger.warning(f"Line {idx}: {str(e)}")

    logger.info(
        f"Parsed {len(parsed_records)} records from {path} "
        f"({len(errors)} errors)"
    )

    if errors:
        logger.warning(f"Encountered {len(errors)} parsing errors")

    return parsed_records


def validate_kv_annotations(
    records: list[CanonicalIERecord],
) -> dict[str, Any]:
    """Validate a collection of K/V annotated records.

    Computes statistics about annotation completeness and consistency.

    Args:
        records: List of CanonicalIERecord objects to validate

    Returns:
        Dictionary with validation statistics including:
        - total: Total number of records
        - valid: Number of valid records
        - complete: Records with all KV fields filled
        - partial: Records with some KV fields filled
        - empty: Records with no KV fields filled
        - missing_values: List of missing fields across all records
        - field_coverage: Dict mapping field names to fill percentages
    """
    logger.info(f"Validating {len(records)} K/V records")

    stats = {
        "total": len(records),
        "valid": 0,
        "complete": 0,
        "partial": 0,
        "empty": 0,
        "missing_values": {},
        "field_coverage": {},
        "errors": [],
    }

    if not records:
        logger.info("No records to validate")
        return stats

    # Collect all fields across records
    all_fields = set()
    for record in records:
        if record.answer.kv:
            all_fields.update(record.answer.kv.keys())

    # Initialize field coverage
    for field in all_fields:
        stats["field_coverage"][field] = {"filled": 0, "empty": 0}
        stats["missing_values"][field] = 0

    # Validate each record
    for record in records:
        try:
            # Check basic validity
            if not record.is_valid():
                stats["errors"].append(
                    f"Record {record.id}: Failed internal validity check"
                )
                continue

            stats["valid"] += 1

            # Check KV completeness
            if not record.answer.kv:
                stats["empty"] += 1
                for field in all_fields:
                    stats["missing_values"][field] += 1
                continue

            # Count filled vs empty fields
            filled = sum(1 for v in record.answer.kv.values() if v is not None)
            total = len(record.answer.kv)

            if filled == total:
                stats["complete"] += 1
            elif filled > 0:
                stats["partial"] += 1
            else:
                stats["empty"] += 1

            # Update field coverage
            for field, value in record.answer.kv.items():
                if value is not None:
                    stats["field_coverage"][field]["filled"] += 1
                else:
                    stats["field_coverage"][field]["empty"] += 1
                    stats["missing_values"][field] += 1

        except Exception as e:
            stats["errors"].append(f"Record {record.id}: {str(e)}")

    # Calculate percentages
    for field, counts in stats["field_coverage"].items():
        total_field = counts["filled"] + counts["empty"]
        if total_field > 0:
            coverage_pct = (counts["filled"] / total_field) * 100
            stats["field_coverage"][field]["coverage_pct"] = round(coverage_pct, 2)

    logger.info(f"Validation complete:")
    logger.info(f"  Valid records: {stats['valid']} / {stats['total']}")
    logger.info(f"  Complete: {stats['complete']}, Partial: {stats['partial']}, "
                f"Empty: {stats['empty']}")
    logger.info(f"  Errors: {len(stats['errors'])}")

    return stats
