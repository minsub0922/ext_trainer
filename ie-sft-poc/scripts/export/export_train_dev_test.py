#!/usr/bin/env python3
"""CLI to split unified dataset into train/dev/test splits."""

import argparse
import json
import sys
from pathlib import Path

from src.common.logging_utils import get_logger
from src.common.paths import DATA_PROCESSED
from src.common.constants import DEFAULT_SEED
from src.datasets.unified import split_dataset

logger = get_logger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Split a unified dataset into train/dev/test splits"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to unified JSONL dataset",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_PROCESSED / "splits"),
        help="Output directory for split files (default: data/processed/splits/)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion for training split (default: 0.8)",
    )

    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Proportion for development split (default: 0.1)",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion for test split (default: 0.1)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )

    parser.add_argument(
        "--stratify-by",
        choices=["source", "task_type", "none"],
        default="source",
        help="How to stratify splits (default: source)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they exist",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Validate input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    # Check if output exists
    if output_dir.exists() and not args.overwrite:
        # Check if any split files exist
        if any(
            (output_dir / f"{split}.jsonl").exists()
            for split in ["train", "dev", "test"]
        ):
            logger.error(
                f"Output directory contains existing splits: {output_dir}. "
                f"Use --overwrite to replace."
            )
            return 1

    try:
        logger.info(f"Splitting dataset: {input_path}")
        logger.info(
            f"Ratios: train={args.train_ratio}, "
            f"dev={args.dev_ratio}, test={args.test_ratio}"
        )

        split_stats = split_dataset(
            input_path=input_path,
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            stratify_by=args.stratify_by,
        )

        # Print split statistics
        print("\n" + str(split_stats))

        # Export stats
        stats_path = output_dir / "split_stats.json"
        with open(stats_path, "w") as f:
            json.dump(split_stats.to_dict(), f, indent=2)
        logger.info(f"Wrote split statistics to {stats_path}")

        logger.info("Dataset splitting completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Dataset splitting failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
