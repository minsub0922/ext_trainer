#!/usr/bin/env python3
"""
CLI script to download the InstructIE dataset from HuggingFace.

Downloads the InstructIE dataset and saves it to the specified directory
with metadata. Supports partial downloads (English or Chinese subsets)
and verification of downloaded data.

Example usage:
    # Download all subsets
    python scripts/download/download_instructie.py

    # Download only English
    python scripts/download/download_instructie.py --subset en

    # Download only Chinese
    python scripts/download/download_instructie.py --subset zh

    # Specify output directory
    python scripts/download/download_instructie.py --output-dir /custom/path

    # Dry run (just check, don't download)
    python scripts/download/download_instructie.py --dry-run

    # Verify existing download
    python scripts/download/download_instructie.py --verify-only --output-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path

from src.common.logging_utils import get_logger, setup_logging
from src.common.paths import DATA_RAW, ensure_directories_exist
from src.datasets.instructie.downloader import InstructIEDownloader

logger = get_logger(__name__)


def main() -> int:
    """Main entry point for the download script.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Download the InstructIE dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all subsets (default)
  python scripts/download/download_instructie.py

  # Download only English split
  python scripts/download/download_instructie.py --subset en

  # Download to custom directory
  python scripts/download/download_instructie.py --output-dir /data/instructie

  # Verify existing download
  python scripts/download/download_instructie.py --verify-only --output-dir /data/instructie

  # Dry run (just check what would be downloaded)
  python scripts/download/download_instructie.py --dry-run
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_RAW / "instructie"),
        help=(
            f"Output directory for downloaded data "
            f"(default: {DATA_RAW / 'instructie'})"
        ),
    )

    parser.add_argument(
        "--subset",
        type=str,
        choices=["en", "zh", "all"],
        default="all",
        help="Subset to download: 'en' (English), 'zh' (Chinese), or 'all' (default)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloaded data (do not download)",
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

    output_dir = Path(args.output_dir)

    try:
        # Create downloader
        logger.info(f"Creating InstructIE downloader (subset={args.subset})")
        downloader = InstructIEDownloader(subset=args.subset)

        logger.info(f"Dataset: {downloader.metadata.name}")
        logger.info(f"Source: {downloader.metadata.source_url}")
        logger.info(f"Task types: {', '.join(downloader.metadata.task_types)}")

        # Handle verify-only mode
        if args.verify_only:
            logger.info("Verification-only mode")
            logger.info(f"Verifying data in {output_dir}...")
            success = downloader.verify(output_dir)

            if success:
                logger.info("Verification passed!")
                return 0
            else:
                logger.error("Verification failed!")
                return 1

        # Handle dry-run mode
        if args.dry_run:
            logger.info("Dry-run mode (no data will be downloaded)")
            logger.info(f"Would download subset: {args.subset}")
            logger.info(f"Would save to: {output_dir}")

            if args.subset == "en":
                subsets_to_download = ["train_en", "dev_en", "test_en"]
            elif args.subset == "zh":
                subsets_to_download = ["train_zh", "dev_zh", "test_zh"]
            else:
                subsets_to_download = [
                    "train_en",
                    "dev_en",
                    "test_en",
                    "train_zh",
                    "dev_zh",
                    "test_zh",
                ]

            logger.info(f"Would download {len(subsets_to_download)} splits:")
            for split in subsets_to_download:
                logger.info(f"  - {split}")

            return 0

        # Normal download mode
        logger.info(f"Downloading to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        success = downloader.download(output_dir)

        if not success:
            logger.error("Download failed!")
            return 1

        # Verify after download
        logger.info("Verifying downloaded data...")
        verified = downloader.verify(output_dir)

        if verified:
            logger.info("Download and verification successful!")
            return 0
        else:
            logger.error("Verification failed!")
            return 1

    except ValueError as e:
        logger.error(f"Invalid argument: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
