#!/usr/bin/env python3
"""CLI to convert canonical IE datasets to LLaMA-Factory instruction format."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.common.io import read_jsonl, write_jsonl
from src.common.logging_utils import get_logger
from src.common.paths import DATA_PROCESSED
from src.common.schema import CanonicalIERecord
from src.datasets.gollie_reference.schema_patterns import (
    FORMAT_INSTRUCTIONS,
    build_schema_prompt,
)
from src.training.dataset_registry_builder import build_dataset_info

logger = get_logger(__name__)

# Multimodal placeholder tokens that LLaMA-Factory's mm_plugin interprets
# as media markers. When these appear as literal text (e.g. HTML content
# mentioning "<video>"), they crash the tokenizer. We escape them by
# inserting a zero-width space so they are no longer recognised as tokens.
import re

_MM_TOKEN_RE = re.compile(r"<(video|audio|image)>", re.IGNORECASE)


def _escape_mm_tokens(text: str) -> str:
    """Escape multimodal placeholder tokens in text content.

    Inserts a zero-width space after '<' so that '<video>' becomes
    '<\u200bvideo>' which is visually identical but not parsed as a
    multimodal token by LLaMA-Factory's mm_plugin.
    """
    ZWS = "\u200b"  # actual zero-width space character
    return _MM_TOKEN_RE.sub(rf"<{ZWS}\1>", text)


def build_llamafactory_record(
    record: CanonicalIERecord,
    prompt_mode: str = "unified",
    custom_template: str | None = None,
) -> dict[str, str]:
    """Convert a canonical record to LLaMA-Factory instruction format.

    Args:
        record: Canonical IE record
        prompt_mode: Type of prompt ('kv', 'entity', 'relation', 'unified')
        custom_template: Optional custom prompt template

    Returns:
        Dictionary with 'instruction', 'input', 'output' keys
    """
    # Build instruction (system prompt)
    if custom_template:
        instruction = custom_template
    else:
        base_instruction = (
            "You are an expert information extraction system that extracts structured "
            "information from text. Your task is to identify and extract information "
            "according to the specified schema."
        )

        schema_part = build_schema_prompt(prompt_mode, record.schema_def)
        instruction = f"{base_instruction}\n\n{schema_part}\n\n{FORMAT_INSTRUCTIONS.get(prompt_mode, FORMAT_INSTRUCTIONS['unified'])}"

    # Build input (text + task-specific context)
    # Escape multimodal tokens that may appear as literal text (e.g. HTML articles)
    safe_text = _escape_mm_tokens(record.text)
    input_text = f"Text:\n{safe_text}"

    # Build output (answer as strict JSON). Answer is a plain pydantic
    # BaseModel (only CanonicalIERecord defines to_canonical_dict), so
    # use model_dump directly.
    answer_dict = record.answer.model_dump(exclude_none=False)

    # Filter answer based on prompt mode
    if prompt_mode == "kv":
        answer_dict = {"kv": answer_dict.get("kv", {})}
    elif prompt_mode == "entity":
        answer_dict = {"entity": answer_dict.get("entity", [])}
    elif prompt_mode == "relation":
        answer_dict = {"relation": answer_dict.get("relation", [])}
    # else: unified keeps all

    output = json.dumps(answer_dict, ensure_ascii=False)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def convert_to_llamafactory(
    input_path: Path | str,
    output_path: Path | str,
    prompt_mode: str = "unified",
    custom_template: str | None = None,
    split: str = "all",
) -> dict[str, Any]:
    """Convert canonical dataset to LLaMA-Factory format.

    Args:
        input_path: Path to canonical JSONL file
        output_path: Path to write LLaMA-Factory JSONL
        prompt_mode: Type of prompt to generate
        custom_template: Optional custom prompt template path
        split: Which split to process ('all' for all records)

    Returns:
        Conversion statistics
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load custom template if provided
    template_text = None
    if custom_template:
        template_path = Path(custom_template)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        with open(template_path) as f:
            template_text = f.read()

    logger.info(f"Converting {input_path} to LLaMA-Factory format")
    logger.info(f"Prompt mode: {prompt_mode}")

    output_records = []
    stats = {
        "total_input": 0,
        "total_output": 0,
        "errors": [],
    }

    for record_data in read_jsonl(input_path):
        try:
            record = CanonicalIERecord.from_dict(record_data)
            stats["total_input"] += 1

            # Check split filter
            if split != "all" and record.meta.split != split:
                continue

            # Convert to LLaMA-Factory format
            llama_record = build_llamafactory_record(
                record,
                prompt_mode=prompt_mode,
                custom_template=template_text,
            )

            output_records.append(llama_record)
            stats["total_output"] += 1

        except Exception as e:
            logger.warning(f"Skipping invalid record: {e}")
            stats["errors"].append(str(e))

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_records, output_path)
    logger.info(f"Wrote {len(output_records)} converted records to {output_path}")

    return stats


def process_directory(
    input_dir: Path | str,
    output_dir: Path | str,
    prompt_mode: str = "unified",
    custom_template: str | None = None,
) -> dict[str, Any]:
    """Process directory with train/dev/test.jsonl files.

    Args:
        input_dir: Directory containing train.jsonl, dev.jsonl, test.jsonl
        output_dir: Output directory for converted files
        prompt_mode: Type of prompt to generate
        custom_template: Optional custom prompt template path

    Returns:
        Conversion statistics for all splits
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    all_stats = {}

    for split in ["train", "dev", "test"]:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            logger.warning(f"Split file not found: {input_path}")
            continue

        output_path = output_dir / f"{split}.jsonl"
        # The per-split file IS the authoritative split — records inside
        # carry their original meta.split label (e.g. "train_en"), which
        # wouldn't equal "train". Pass "all" so the converter emits every
        # record in the file.
        stats = convert_to_llamafactory(
            input_path=input_path,
            output_path=output_path,
            prompt_mode=prompt_mode,
            custom_template=custom_template,
            split="all",
        )
        all_stats[split] = stats
        logger.info(
            f"[{split}] wrote {stats['total_output']}/{stats['total_input']} records"
        )

    return all_stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert canonical IE dataset to LLaMA-Factory instruction format"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to canonical JSONL file or directory with train/dev/test.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_PROCESSED / "llamafactory"),
        help="Output directory for converted dataset (default: data/processed/llamafactory/)",
    )

    parser.add_argument(
        "--prompt-mode",
        choices=["kv", "entity", "relation", "unified"],
        default="unified",
        help="Type of prompt to generate (default: unified)",
    )

    parser.add_argument(
        "--template",
        help="Optional path to custom prompt template",
    )

    parser.add_argument(
        "--generate-registry",
        action="store_true",
        default=True,
        help="Generate dataset_info.json for LLaMA-Factory (default: True)",
    )

    parser.add_argument(
        "--no-generate-registry",
        action="store_false",
        dest="generate_registry",
        help="Skip generating dataset_info.json",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ie_sft_unified",
        help="Name for dataset in registry (default: ie_sft_unified)",
    )

    parser.add_argument(
        "--split",
        choices=["train", "dev", "test", "all"],
        default="all",
        help="Which split to process (default: all)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write files, just print what would happen",
    )

    parser.add_argument(
        "--print-example",
        action="store_true",
        help="Print first converted example and exit",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return 1

    try:
        # Check if input is directory or file
        if input_path.is_dir():
            logger.info(f"Processing directory: {input_path}")
            is_directory = True
        else:
            logger.info(f"Processing file: {input_path}")
            is_directory = False

        # Print example if requested
        if args.print_example:
            if is_directory:
                example_file = input_path / "train.jsonl"
            else:
                example_file = input_path

            if not example_file.exists():
                logger.error(f"Cannot find example file: {example_file}")
                return 1

            records = read_jsonl(example_file)
            first_record = next(records)
            record = CanonicalIERecord.from_dict(first_record)
            example = build_llamafactory_record(
                record,
                prompt_mode=args.prompt_mode,
                custom_template=args.template,
            )

            print("\nExample converted record:")
            print(json.dumps(example, indent=2, ensure_ascii=False))
            return 0

        # Convert dataset
        if args.dry_run:
            logger.info("DRY RUN: Not writing files")

        if is_directory:
            all_stats = process_directory(
                input_dir=input_path,
                output_dir=output_dir,
                prompt_mode=args.prompt_mode,
                custom_template=args.template,
            )
            print("\nConversion Statistics:")
            for split, stats in sorted(all_stats.items()):
                print(f"  {split}: {stats['total_output']}/{stats['total_input']} records")
        else:
            stats = convert_to_llamafactory(
                input_path=input_path,
                output_path=output_dir / "dataset.jsonl",
                prompt_mode=args.prompt_mode,
                custom_template=args.template,
                split=args.split,
            )
            print(f"\nConverted {stats['total_output']}/{stats['total_input']} records")

        # Generate dataset_info.json if requested
        if args.generate_registry and not args.dry_run:
            logger.info("Generating dataset_info.json")
            registry = build_dataset_info(
                dataset_name=args.dataset_name,
                output_dir=output_dir,
            )
            registry_path = output_dir / "dataset_info.json"
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Wrote dataset registry to {registry_path}")

        logger.info("Conversion completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
