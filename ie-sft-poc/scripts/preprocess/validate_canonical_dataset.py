#!/usr/bin/env python3
"""CLI to validate canonical IE datasets."""

import argparse
import json
import sys
from pathlib import Path

from src.common.logging_utils import get_logger
from src.datasets.unified import validate_dataset

logger = get_logger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate a canonical IE dataset JSONL file"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to canonical JSONL dataset to validate",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )

    parser.add_argument(
        "--print-errors",
        action="store_true",
        help="Print all error and warning messages",
    )

    parser.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Maximum number of errors to print (default: 50)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    try:
        logger.info(f"Validating dataset: {input_path}")
        report = validate_dataset(input_path, strict=args.strict)

        # Print report
        print("\n" + str(report) + "\n")

        # Print detailed errors if requested
        if args.print_errors:
            if report.errors:
                print("Errors:")
                for error in report.errors[: args.max_errors]:
                    print(f"  {error}")
                if len(report.errors) > args.max_errors:
                    print(f"  ... and {len(report.errors) - args.max_errors} more")
                print()

            if report.warnings:
                print("Warnings:")
                for warning in report.warnings[: args.max_errors]:
                    print(f"  {warning}")
                if len(report.warnings) > args.max_errors:
                    print(f"  ... and {len(report.warnings) - args.max_errors} more")
                print()

        # Export report
        report_path = input_path.with_suffix(".validation.json")
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Wrote validation report to {report_path}")

        # Return appropriate exit code
        if not report.is_valid:
            logger.error("Validation failed")
            return 1

        if report.has_warnings and args.strict:
            logger.error("Validation failed (strict mode with warnings)")
            return 1

        logger.info("Validation passed")
        return 0

    except Exception as e:
        logger.error(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
