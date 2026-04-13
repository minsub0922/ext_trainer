"""Merge multiple canonical JSONL datasets with deduplication and filtering."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.io import read_jsonl, write_jsonl
from src.common.logging_utils import get_logger
from src.common.schema import CanonicalIERecord
from src.common.constants import SUPPORTED_TASK_TYPES

logger = get_logger(__name__)


@dataclass
class MergeStats:
    """Statistics from dataset merging operation.

    Attributes:
        total_input: Total records from all input files
        total_output: Total records in merged output
        duplicates_removed: Number of duplicate records removed
        by_source: Dictionary mapping source name to record count
        by_task_type: Dictionary mapping task type to record count
    """

    total_input: int = 0
    total_output: int = 0
    duplicates_removed: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_task_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "duplicates_removed": self.duplicates_removed,
            "by_source": self.by_source,
            "by_task_type": self.by_task_type,
        }

    def __str__(self) -> str:
        """Return human-readable summary."""
        lines = [
            "Merge Statistics:",
            f"  Total input records: {self.total_input}",
            f"  Total output records: {self.total_output}",
            f"  Duplicates removed: {self.duplicates_removed}",
            f"  Deduplication rate: {self.duplicates_removed / max(self.total_input, 1) * 100:.2f}%",
        ]

        if self.by_source:
            lines.append("  By source:")
            for source, count in sorted(self.by_source.items()):
                lines.append(f"    {source}: {count}")

        if self.by_task_type:
            lines.append("  By task type:")
            for task_type, count in sorted(self.by_task_type.items()):
                lines.append(f"    {task_type}: {count}")

        return "\n".join(lines)


def _compute_text_hash(text: str) -> str:
    """Compute MD5 hash of text for deduplication.

    Args:
        text: Text to hash

    Returns:
        MD5 hexdigest
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def merge_datasets(
    input_paths: list[Path | str],
    output_path: Path | str,
    deduplicate: bool = True,
    task_filter: list[str] | None = None,
) -> MergeStats:
    """Merge multiple canonical JSONL datasets.

    Combines records from multiple input files, optionally deduplicating by
    record ID and/or text content hash. Can filter to specific task types.

    Args:
        input_paths: List of paths to input JSONL files
        output_path: Path to write merged JSONL file
        deduplicate: If True, remove duplicate records (by ID and text hash)
        task_filter: If provided, only include records with these task types

    Returns:
        MergeStats with merge statistics

    Raises:
        FileNotFoundError: If any input file does not exist
        ValueError: If task_filter contains invalid task types
    """
    # Validate inputs
    input_paths = [Path(p) for p in input_paths]
    output_path = Path(output_path)

    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    if task_filter:
        invalid_types = set(task_filter) - set(SUPPORTED_TASK_TYPES)
        if invalid_types:
            raise ValueError(
                f"Invalid task types in filter: {invalid_types}. "
                f"Must be subset of {SUPPORTED_TASK_TYPES}"
            )

    logger.info(f"Merging {len(input_paths)} dataset(s)")
    if task_filter:
        logger.info(f"Filtering to task types: {task_filter}")

    # Track merged records
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    merged_records: list[dict[str, Any]] = []
    stats = MergeStats()

    # Process each input file
    for input_path in input_paths:
        logger.info(f"Processing {input_path}")
        source_count = 0

        try:
            for record_data in read_jsonl(input_path):
                stats.total_input += 1

                # Validate and parse record
                try:
                    record = CanonicalIERecord.from_dict(record_data)
                except Exception as e:
                    logger.warning(f"Skipping invalid record: {e}")
                    continue

                # Apply task filter if specified
                if task_filter:
                    if not any(t in record.task_types for t in task_filter):
                        continue

                # Check for duplicates by ID
                if deduplicate and record.id in seen_ids:
                    stats.duplicates_removed += 1
                    logger.debug(f"Skipping duplicate ID: {record.id}")
                    continue

                # Check for duplicates by text hash
                text_hash = _compute_text_hash(record.text)
                if deduplicate and text_hash in seen_hashes:
                    stats.duplicates_removed += 1
                    logger.debug(f"Skipping duplicate text: {record.id}")
                    continue

                # Record accepted
                if deduplicate:
                    seen_ids.add(record.id)
                    seen_hashes.add(text_hash)

                merged_records.append(record.to_canonical_dict())
                source_count += 1
                stats.total_output += 1

                # Update source stats
                source = record.source or "unknown"
                stats.by_source[source] = stats.by_source.get(source, 0) + 1

                # Update task type stats
                for task_type in record.task_types:
                    stats.by_task_type[task_type] = (
                        stats.by_task_type.get(task_type, 0) + 1
                    )

        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            raise

        logger.info(f"  Processed {source_count} records from {input_path}")

    # Write merged dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(merged_records, output_path)
    logger.info(f"Wrote {len(merged_records)} merged records to {output_path}")

    return stats
