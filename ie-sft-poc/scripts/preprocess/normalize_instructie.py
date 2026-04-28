#!/usr/bin/env python3
"""
CLI script to preprocess and normalize InstructIE data to canonical format.

Converts raw InstructIE JSONL files to the canonical IE record format.
Handles parsing, validation, and conversion to produce standardized records
suitable for training and evaluation.

Example usage:
    # Convert all splits in the default raw directory
    python scripts/preprocess/normalize_instructie.py

    # Convert specific splits only
    python scripts/preprocess/normalize_instructie.py --split train

    # Convert with custom input/output directories
    python scripts/preprocess/normalize_instructie.py \
        --input-dir /data/raw/instructie \
        --output-dir /data/processed/instructie

    # Show statistics only (don't write files)
    python scripts/preprocess/normalize_instructie.py --stats-only

    # Strict mode (fail on first error)
    python scripts/preprocess/normalize_instructie.py --strict

    # Overwrite existing output files
    python scripts/preprocess/normalize_instructie.py --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging_utils import get_logger, setup_logging
from src.common.paths import DATA_INTERIM, DATA_RAW, ensure_directories_exist
from src.datasets.instructie.converter import convert_dataset, convert_file

logger = get_logger(__name__)


def print_stats(stats: dict) -> None:
    """Print conversion statistics in a formatted way.

    Args:
        stats: Statistics dictionary from conversion.
    """
    print("\n" + "=" * 70)
    print("CONVERSION STATISTICS")
    print("=" * 70)

    if "splits" in stats:
        # Dataset-level statistics
        print(f"\nTotal records across all splits: {stats['total_records']}")
        print(f"Successfully converted: {stats['total_success']}")
        print(f"Failed conversions: {stats['total_failed']}")

        if stats["total_records"] > 0:
            success_rate = (stats["total_success"] / stats["total_records"]) * 100
            print(f"Success rate: {success_rate:.1f}%")

        print("\nPer-split statistics:")
        for split_name, split_stats in stats["splits"].items():
            print(f"\n  {split_name}:")
            print(f"    Total: {split_stats.get('total', 0)}")
            print(f"    Success: {split_stats.get('success', 0)}")
            print(f"    Failed: {split_stats.get('failed', 0)}")

            if split_stats.get("by_task_type"):
                task_stats = split_stats["by_task_type"]
                print(f"    Task type distribution:")
                print(f"      Entity only: {task_stats.get('entity_only', 0)}")
                print(f"      Relation only: {task_stats.get('relation_only', 0)}")
                print(f"      Both: {task_stats.get('both', 0)}")

    print("\n" + "=" * 70)


def main() -> int:
    """Main entry point for the preprocessing script.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Preprocess and normalize InstructIE data to canonical format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all splits (default behavior)
  python scripts/preprocess/normalize_instructie.py

  # Convert only training split
  python scripts/preprocess/normalize_instructie.py --split train

  # Use custom directories
  python scripts/preprocess/normalize_instructie.py \
      --input-dir /custom/raw --output-dir /custom/processed

  # Show statistics without writing output
  python scripts/preprocess/normalize_instructie.py --stats-only

  # Strict mode (fail on first error during conversion)
  python scripts/preprocess/normalize_instructie.py --strict

  # Overwrite existing files
  python scripts/preprocess/normalize_instructie.py --overwrite
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DATA_RAW / "instructie"),
        help=(
            f"Input directory with raw InstructIE data "
            f"(default: {DATA_RAW / 'instructie'})"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_INTERIM / "instructie"),
        help=(
            f"Output directory for normalized data "
            f"(default: {DATA_INTERIM / 'instructie'})"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help=(
            "Specific split(s) to convert (train/dev/test/all). "
            "Use 'all' to convert all available splits. (default: all)"
        ),
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show statistics only, do not write output files",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first parse or conversion error",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Ensure project directories exist
    ensure_directories_exist()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    try:
        # Check input directory
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            logger.info("Please download the InstructIE data first using:")
            logger.info("  python scripts/download/download_instructie.py")
            return 1

        # Check for input files
        input_files = list(input_dir.glob("*.jsonl"))
        if not input_files:
            logger.error(f"No JSONL files found in {input_dir}")
            return 1

        logger.info(f"Found {len(input_files)} input files")

        # Determine splits to process
        if args.split.lower() == "all":
            splits_to_process = None  # Process all found files
            logger.info("Processing all available splits")
        else:
            # Parse split specification (could be "train" or pattern)
            split_patterns = [s.strip() for s in args.split.split(",")]
            splits_to_process = []

            for pattern in split_patterns:
                matching = [f.stem for f in input_files if pattern in f.stem]
                if matching:
                    splits_to_process.extend(matching)
                else:
                    logger.warning(f"No splits found matching pattern: {pattern}")

            if not splits_to_process:
                logger.error(f"No splits found matching: {args.split}")
                return 1

            logger.info(f"Processing splits: {', '.join(splits_to_process)}")

        # Check for existing output files
        if not args.overwrite and output_dir.exists():
            existing_files = list(output_dir.glob("*.jsonl"))
            if existing_files:
                logger.warning(
                    f"Output directory contains {len(existing_files)} existing files"
                )
                logger.warning("Use --overwrite to replace them")
                if splits_to_process is None:
                    logger.error("Cannot proceed without --overwrite")
                    return 1

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert dataset
        logger.info(f"Converting InstructIE data from {input_dir}")
        logger.info(f"Output will be saved to {output_dir}")

        stats = convert_dataset(
            raw_dir=input_dir,
            output_dir=output_dir,
            splits=splits_to_process,
        )

        # Print statistics
        print_stats(stats)

        # Save statistics to file
        stats_file = output_dir / "conversion_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to {stats_file}")

        # Determine exit code based on results
        if stats["total_success"] > 0:
            logger.info("Preprocessing complete!")
            return 0
        else:
            logger.error("No records were successfully converted")
            return 1

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        if args.strict:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
