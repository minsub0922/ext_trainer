"""Split unified dataset into train/dev/test splits with stratification."""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.io import read_jsonl, write_jsonl
from src.common.logging_utils import get_logger
from src.common.schema import CanonicalIERecord
from src.common.constants import DEFAULT_SEED

logger = get_logger(__name__)


@dataclass
class SplitStats:
    """Statistics from dataset splitting operation.

    Attributes:
        total: Total records in input dataset
        train_count: Number of records in training split
        dev_count: Number of records in development split
        test_count: Number of records in test split
        by_source: Dictionary mapping split and source to record count
        by_task_type: Dictionary mapping split and task type to record count
    """

    total: int = 0
    train_count: int = 0
    dev_count: int = 0
    test_count: int = 0
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)
    by_task_type: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total": self.total,
            "train_count": self.train_count,
            "dev_count": self.dev_count,
            "test_count": self.test_count,
            "by_source": self.by_source,
            "by_task_type": self.by_task_type,
        }

    def __str__(self) -> str:
        """Return human-readable summary."""
        lines = [
            "Split Statistics:",
            f"  Total records: {self.total}",
            f"  Train: {self.train_count} ({self.train_count/max(self.total, 1)*100:.2f}%)",
            f"  Dev: {self.dev_count} ({self.dev_count/max(self.total, 1)*100:.2f}%)",
            f"  Test: {self.test_count} ({self.test_count/max(self.total, 1)*100:.2f}%)",
        ]

        if self.by_source:
            lines.append("  By source:")
            for split, sources in sorted(self.by_source.items()):
                lines.append(f"    {split}:")
                for source, count in sorted(sources.items()):
                    lines.append(f"      {source}: {count}")

        if self.by_task_type:
            lines.append("  By task type:")
            for split, task_types in sorted(self.by_task_type.items()):
                lines.append(f"    {split}:")
                for task_type, count in sorted(task_types.items()):
                    lines.append(f"      {task_type}: {count}")

        return "\n".join(lines)


def split_dataset(
    input_path: Path | str,
    output_dir: Path | str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = DEFAULT_SEED,
    stratify_by: str = "source",
) -> SplitStats:
    """Split unified dataset into train/dev/test splits.

    Performs stratified splitting to maintain distribution of records across
    splits based on source or task type.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to write split files (train.jsonl, dev.jsonl, test.jsonl)
        train_ratio: Proportion of records for training (default: 0.8)
        dev_ratio: Proportion of records for development (default: 0.1)
        test_ratio: Proportion of records for testing (default: 0.1)
        seed: Random seed for reproducibility (default: DEFAULT_SEED)
        stratify_by: How to stratify ('source', 'task_type', or 'none')

    Returns:
        SplitStats with split statistics

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If ratios don't sum to 1.0 or invalid stratify_by option
    """
    # Validate inputs
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point error
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, dev={dev_ratio}, test={test_ratio})"
        )

    if stratify_by not in ("source", "task_type", "none"):
        raise ValueError(
            f"Invalid stratify_by: {stratify_by}. "
            f"Must be 'source', 'task_type', or 'none'"
        )

    logger.info(f"Splitting dataset from {input_path}")
    logger.info(
        f"Ratios: train={train_ratio}, dev={dev_ratio}, test={test_ratio}"
    )
    logger.info(f"Stratification: {stratify_by}")

    # Set random seed for reproducibility
    random.seed(seed)

    # Load all records
    all_records = []
    for record_data in read_jsonl(input_path):
        try:
            record = CanonicalIERecord.from_dict(record_data)
            all_records.append(record)
        except Exception as e:
            logger.warning(f"Skipping invalid record: {e}")
            continue

    logger.info(f"Loaded {len(all_records)} records")

    # Group records for stratification
    if stratify_by == "source":
        groups = {}
        for record in all_records:
            source = record.source or "unknown"
            if source not in groups:
                groups[source] = []
            groups[source].append(record)
    elif stratify_by == "task_type":
        groups = {}
        for record in all_records:
            # Use first task type as stratification key
            task_type = record.task_types[0] if record.task_types else "unknown"
            if task_type not in groups:
                groups[task_type] = []
            groups[task_type].append(record)
    else:  # none
        groups = {"all": all_records}

    # Split each group
    train_records = []
    dev_records = []
    test_records = []
    stats = SplitStats(total=len(all_records))

    for group_name, group_records in sorted(groups.items()):
        # Shuffle within group
        random.shuffle(group_records)

        # Calculate split points
        n = len(group_records)
        train_idx = int(n * train_ratio)
        dev_idx = train_idx + int(n * dev_ratio)

        # Perform split
        train_group = group_records[:train_idx]
        dev_group = group_records[train_idx:dev_idx]
        test_group = group_records[dev_idx:]

        train_records.extend(train_group)
        dev_records.extend(dev_group)
        test_records.extend(test_group)

        logger.debug(
            f"  {group_name}: train={len(train_group)}, "
            f"dev={len(dev_group)}, test={len(test_group)}"
        )

    stats.train_count = len(train_records)
    stats.dev_count = len(dev_records)
    stats.test_count = len(test_records)

    # Compute detailed stats
    for split_name, split_records in [
        ("train", train_records),
        ("dev", dev_records),
        ("test", test_records),
    ]:
        stats.by_source[split_name] = {}
        stats.by_task_type[split_name] = {}

        for record in split_records:
            source = record.source or "unknown"
            stats.by_source[split_name][source] = (
                stats.by_source[split_name].get(source, 0) + 1
            )

            for task_type in record.task_types:
                stats.by_task_type[split_name][task_type] = (
                    stats.by_task_type[split_name].get(task_type, 0) + 1
                )

    # Write splits
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    test_path = output_dir / "test.jsonl"

    write_jsonl([r.to_canonical_dict() for r in train_records], train_path)
    write_jsonl([r.to_canonical_dict() for r in dev_records], dev_path)
    write_jsonl([r.to_canonical_dict() for r in test_records], test_path)

    logger.info(f"Wrote splits to {output_dir}:")
    logger.info(f"  train.jsonl: {stats.train_count} records")
    logger.info(f"  dev.jsonl: {stats.dev_count} records")
    logger.info(f"  test.jsonl: {stats.test_count} records")

    return stats
