#!/usr/bin/env python3
"""CLI script for generating K/V annotation templates from raw data.

This script reads CSV or JSONL input data and creates annotation templates
in the canonical CanonicalIERecord format, ready for manual annotation.
"""

import argparse
from pathlib import Path

from src.common.logging_utils import get_logger
from src.common.paths import DATA_INTERIM
from src.datasets.internal_kv.template_builder import (
    KVTemplateConfig,
    build_kv_template,
    print_template_stats,
)

logger = get_logger(__name__)


def parse_kv_fields(kv_fields_str: str) -> list[str]:
    """Parse comma-separated KV fields string.

    Args:
        kv_fields_str: Comma-separated field names

    Returns:
        List of field names
    """
    if not kv_fields_str:
        raise ValueError("kv_fields cannot be empty")

    fields = [f.strip() for f in kv_fields_str.split(",")]
    fields = [f for f in fields if f]  # Remove empty strings

    if not fields:
        raise ValueError("kv_fields produced no valid fields")

    return fields


def main() -> None:
    """Main entry point for the template generation script."""
    parser = argparse.ArgumentParser(
        description="Generate K/V annotation templates from raw data"
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file path (CSV or JSONL format)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        required=True,
        help="Name of the column containing the main text to extract from",
    )
    parser.add_argument(
        "--kv-fields",
        type=str,
        required=True,
        help="Comma-separated list of field names to extract (e.g., 'name,email,phone')",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_INTERIM / "internal_kv",
        help="Output directory for templates (default: data/interim/internal_kv/)",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Name of the column containing record IDs (optional)",
    )
    parser.add_argument(
        "--lang-column",
        type=str,
        default=None,
        help="Name of the column containing language codes (optional)",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="internal",
        help="Name of the data source (default: 'internal')",
    )
    parser.add_argument(
        "--default-lang",
        type=str,
        default="ko",
        help="Default language code if not in column (default: 'ko')",
    )
    parser.add_argument(
        "--print-example",
        action="store_true",
        help="Print one sample template and exit without saving",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    args = parser.parse_args()

    logger.info("Starting K/V template generation")
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Text column: {args.text_column}")

    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Parse KV fields
    try:
        kv_fields = parse_kv_fields(args.kv_fields)
        logger.info(f"  KV fields: {kv_fields}")
    except ValueError as e:
        logger.error(f"Invalid kv_fields argument: {str(e)}")
        raise

    # Create config
    try:
        config = KVTemplateConfig(
            text_column=args.text_column,
            id_column=args.id_column,
            lang_column=args.lang_column,
            kv_fields=kv_fields,
            source_name=args.source_name,
            default_lang=args.default_lang,
        )
        logger.info("Configuration validated")
    except ValueError as e:
        logger.error(f"Invalid configuration: {str(e)}")
        raise

    try:
        # Build templates
        templates = build_kv_template(
            args.input,
            config,
            output_path=None,  # Don't save yet
        )

        logger.info(f"Built {len(templates)} templates")

        # Print example if requested
        if args.print_example:
            if templates:
                example = templates[0]
                logger.info("Example template:")
                logger.info(example.to_canonical_dict())
            else:
                logger.warning("No templates were built")
            return

        # Prepare output path
        output_dir = Path(args.output)
        output_filename = f"{args.source_name}_kv_templates.jsonl"
        output_path = output_dir / output_filename

        # Check if output file exists
        if output_path.exists() and not args.overwrite:
            logger.error(
                f"Output file already exists: {output_path}\n"
                "Use --overwrite to replace it"
            )
            raise FileExistsError(f"Output file exists: {output_path}")

        # Save templates
        output_dir.mkdir(parents=True, exist_ok=True)
        from src.common.io import write_jsonl

        records_to_save = [t.to_canonical_dict() for t in templates]
        count = write_jsonl(records_to_save, output_path)

        logger.info(f"Saved {count} templates to {output_path}")

        # Print statistics
        stats = print_template_stats(templates)
        logger.info(f"Statistics saved: {stats}")

    except Exception as e:
        logger.error(f"Template generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
