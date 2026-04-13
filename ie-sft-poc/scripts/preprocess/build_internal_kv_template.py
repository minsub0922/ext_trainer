#!/usr/bin/env python3
"""Build K/V annotation templates from configuration.

This script reads a YAML configuration file specifying template building parameters
and generates annotation-ready templates from raw data.

Configuration format:
```yaml
input: path/to/input.csv
output: path/to/output/
source_name: my_dataset
kv_fields:
  - field1
  - field2
  - field3
text_column: description
id_column: record_id
lang_column: language
default_lang: ko
```
"""

import argparse
from pathlib import Path
from typing import Any

from src.common.io import read_yaml, write_jsonl, write_yaml
from src.common.logging_utils import get_logger
from src.datasets.internal_kv.template_builder import (
    KVTemplateConfig,
    build_kv_template,
    print_template_stats,
)

logger = get_logger(__name__)


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")
    config = read_yaml(config_path)

    # Validate required fields
    required_fields = ["input", "text_column", "kv_fields"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Configuration missing required field: {field}")

    return config


def build_config_object(
    config_dict: dict[str, Any],
    input_override: str | None = None,
    output_override: str | None = None,
) -> tuple[KVTemplateConfig, Path | None]:
    """Build KVTemplateConfig from configuration dictionary.

    Args:
        config_dict: Configuration dictionary from YAML
        input_override: Optional override for input path
        output_override: Optional override for output path

    Returns:
        Tuple of (KVTemplateConfig, output_path or None)

    Raises:
        ValueError: If configuration is invalid
    """
    # Handle overrides
    input_path = input_override or config_dict.get("input")
    output_path = output_override or config_dict.get("output")

    if not input_path:
        raise ValueError("No input path specified in config or --input argument")

    input_path = Path(input_path)
    if output_path:
        output_path = Path(output_path)

    # Build config object
    config = KVTemplateConfig(
        text_column=config_dict.get("text_column"),
        id_column=config_dict.get("id_column"),
        lang_column=config_dict.get("lang_column"),
        kv_fields=config_dict.get("kv_fields", []),
        source_name=config_dict.get("source_name", "internal"),
        default_lang=config_dict.get("default_lang", "ko"),
    )

    return config, output_path


def main() -> None:
    """Main entry point for the template building script."""
    parser = argparse.ArgumentParser(
        description="Build K/V annotation templates from YAML configuration"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )

    # Overrides
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Override input path from config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output path from config",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without building templates",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute and print statistics, don't save",
    )

    args = parser.parse_args()

    logger.info("Starting K/V template building from configuration")
    logger.info(f"  Config file: {args.config}")

    # Load and validate configuration
    try:
        config_dict = load_config(args.config)
        logger.info("Configuration loaded and validated")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {str(e)}")
        raise

    # Build configuration object
    try:
        config, output_path = build_config_object(
            config_dict,
            input_override=args.input,
            output_override=args.output,
        )
        logger.info("Configuration object built")
        logger.info(f"  Input: {config_dict.get('input') or args.input}")
        logger.info(f"  Output: {output_path or config_dict.get('output', 'N/A')}")
        logger.info(f"  Text column: {config.text_column}")
        logger.info(f"  KV fields: {config.kv_fields}")
        logger.info(f"  Source: {config.source_name}")

    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise

    # Dry run mode
    if args.dry_run:
        logger.info("[DRY RUN] Would build templates with:")
        logger.info(f"  - Input: {config_dict.get('input') or args.input}")
        logger.info(f"  - Output: {output_path or config_dict.get('output')}")
        logger.info(f"  - Source: {config.source_name}")
        logger.info(f"  - Fields: {config.kv_fields}")
        logger.info("[DRY RUN] No templates generated")
        return

    # Build templates
    try:
        input_path = config_dict.get("input") or args.input
        templates = build_kv_template(
            input_path,
            config,
            output_path=output_path if not args.stats_only else None,
        )

        logger.info(f"Built {len(templates)} templates")

        # Print statistics
        if templates:
            stats = print_template_stats(templates)

            # Save statistics to file if output directory specified
            if output_path and not args.stats_only:
                stats_path = output_path.parent / "template_stats.yaml"
                logger.info(f"Saving statistics to {stats_path}")
                write_yaml(stats, stats_path)

    except Exception as e:
        logger.error(f"Template building failed: {str(e)}")
        raise

    logger.info("K/V template building completed successfully")


if __name__ == "__main__":
    main()
