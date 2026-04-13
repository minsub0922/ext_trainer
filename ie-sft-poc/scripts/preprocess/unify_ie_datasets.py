#!/usr/bin/env python3
"""CLI to merge and unify multiple canonical IE datasets."""

import argparse
import json
import sys
from pathlib import Path

from src.common.logging_utils import get_logger
from src.common.paths import DATA_PROCESSED
from src.datasets.unified import merge_datasets, compute_stats

logger = get_logger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge multiple canonical IE datasets with deduplication"
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more paths to canonical JSONL datasets",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_PROCESSED / "unified.jsonl"),
        help="Output path for merged dataset (default: data/processed/unified.jsonl)",
    )

    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Deduplicate records by ID and text hash (default: True)",
    )

    parser.add_argument(
        "--no-deduplicate",
        action="store_false",
        dest="deduplicate",
        help="Disable deduplication",
    )

    parser.add_argument(
        "--task-types",
        nargs="+",
        choices=["kv", "entity", "relation"],
        help="Filter to specific task types",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute and print statistics without merging",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any invalid records",
    )

    args = parser.parse_args()

    # Validate input paths
    input_paths = [Path(p) for p in args.input]
    for path in input_paths:
        if not path.exists():
            logger.error(f"Input file not found: {path}")
            return 1

    output_path = Path(args.output)

    # Check if output exists
    if output_path.exists() and not args.overwrite:
        logger.error(
            f"Output file already exists: {output_path}. "
            f"Use --overwrite to replace."
        )
        return 1

    try:
        # Merge datasets
        logger.info(f"Merging {len(input_paths)} dataset(s)")
        merge_stats = merge_datasets(
            input_paths=input_paths,
            output_path=output_path,
            deduplicate=args.deduplicate,
            task_filter=args.task_types,
        )

        # Print merge statistics
        print("\n" + str(merge_stats))

        # Compute dataset statistics
        logger.info("Computing dataset statistics...")
        dataset_stats = compute_stats(output_path)
        print("\n" + str(dataset_stats))

        # Export stats
        stats_path = output_path.with_suffix(".stats.json")
        stats_dict = {
            "merge": merge_stats.to_dict(),
            "dataset": dataset_stats.to_dict(),
        }
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2)
        logger.info(f"Wrote statistics to {stats_path}")

        logger.info("Merge completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        if args.strict:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
