#!/usr/bin/env python3
"""Print statistics for JSONL dataset files.

Analyzes dataset composition, task types, text length distribution, etc.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.common.io import read_jsonl
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def print_dataset_stats(input_path: Path | str) -> int:
    """Print comprehensive statistics for a JSONL dataset.

    Args:
        input_path: Path to JSONL dataset file

    Returns:
        0 on success, 1 on error
    """
    input_path = Path(input_path)

    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return 1

    try:
        logger.info(f"Loading dataset from: {input_path}")
        records = read_jsonl(input_path)
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        return 1

    if not records:
        logger.error("Dataset is empty")
        return 1

    logger.info(f"Loaded {len(records)} samples")
    print()

    # Basic stats
    print("Dataset Statistics")
    print("=" * 70)
    print(f"File: {input_path}")
    print(f"Total samples: {len(records)}")
    print()

    # Field analysis
    print("Field Analysis:")
    print("-" * 70)
    all_fields = set()
    field_counts = Counter()

    for record in records:
        if isinstance(record, dict):
            for field in record.keys():
                all_fields.add(field)
                field_counts[field] += 1

    if all_fields:
        print(f"Total unique fields: {len(all_fields)}")
        print("\nField presence:")
        for field, count in field_counts.most_common():
            percentage = (count / len(records)) * 100
            print(f"  {field}: {count}/{len(records)} ({percentage:.1f}%)")
    else:
        print("No dict fields found")
    print()

    # Text length analysis
    print("Text Content Analysis:")
    print("-" * 70)
    text_fields = ["text", "input", "instruction", "content", "query"]
    text_lengths = []

    for record in records:
        if isinstance(record, dict):
            for field in text_fields:
                if field in record:
                    text = record[field]
                    if isinstance(text, str):
                        text_lengths.append(len(text))

    if text_lengths:
        print(f"Text field samples found: {len(text_lengths)}")
        print(f"  Min length: {min(text_lengths)}")
        print(f"  Max length: {max(text_lengths)}")
        print(f"  Mean length: {sum(text_lengths) / len(text_lengths):.1f}")
        print(f"  Median length: {sorted(text_lengths)[len(text_lengths) // 2]}")
    else:
        print("No recognized text fields found")
    print()

    # Task type analysis
    print("Task Type Analysis:")
    print("-" * 70)
    task_types = Counter()
    task_fields = ["task_type", "task", "type", "label_type"]

    for record in records:
        if isinstance(record, dict):
            for field in task_fields:
                if field in record:
                    task_type = record[field]
                    if isinstance(task_type, (str, list)):
                        if isinstance(task_type, list):
                            for t in task_type:
                                task_types[str(t)] += 1
                        else:
                            task_types[task_type] += 1

    if task_types:
        print(f"Task types found: {len(task_types)}")
        for task_type, count in task_types.most_common():
            percentage = (count / len(records)) * 100
            print(f"  {task_type}: {count} ({percentage:.1f}%)")
    else:
        print("No recognized task type fields found")
    print()

    # Sample preview
    print("Sample Preview:")
    print("-" * 70)
    sample_count = min(3, len(records))
    for i in range(sample_count):
        record = records[i]
        print(f"\nSample {i+1}:")
        if isinstance(record, dict):
            for key, value in record.items():
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    print(f"  {key}: {preview}")
                elif isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value).__name__} ({len(value)} items)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {record}")

    print()
    print("=" * 70)
    print("Statistics complete")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Print statistics for IE SFT datasets"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to JSONL dataset file",
    )

    args = parser.parse_args()

    return print_dataset_stats(args.input)


if __name__ == "__main__":
    sys.exit(main())
