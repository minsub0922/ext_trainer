"""Converter for InstructIE records to canonical IE format.

Converts parsed InstructIE records to the canonical CanonicalIERecord format
used throughout the IE SFT PoC project.
"""

import json
from pathlib import Path
from typing import Optional

from src.common.io import read_jsonl, write_jsonl
from src.common.logging_utils import get_logger
from src.common.schema import (
    Answer,
    CanonicalIERecord,
    EntityAnnotation,
    MetaInfo,
    RelationAnnotation,
    SchemaDefinition,
)
from src.datasets.instructie.parser import parse_instructie_file

logger = get_logger(__name__)


def convert_record(parsed: dict, split: str = "train") -> CanonicalIERecord:
    """Convert a parsed InstructIE record to canonical format.

    Args:
        parsed: Parsed record dictionary from parse_instructie_record().
        split: Data split (train/dev/test). Defaults to "train".

    Returns:
        CanonicalIERecord instance.

    Raises:
        ValueError: If record cannot be converted.
    """
    try:
        record_id = parsed.get("id", "")
        text = parsed.get("text", "")
        category = parsed.get("category", "")
        entities_dict = parsed.get("entities", {})
        relations = parsed.get("relations", [])

        if not text or not text.strip():
            raise ValueError("Text is empty")

        # Determine language from record (heuristic: check if split name contains language code)
        lang = "en"  # Default to English
        if isinstance(split, str):
            if "zh" in split.lower():
                lang = "zh"
            elif "en" in split.lower():
                lang = "en"

        # Create entity annotations
        entity_annotations = []
        entity_types = set()

        for entity_text, entity_type in entities_dict.items():
            entity_type = entity_type.strip()
            if entity_type:
                entity_types.add(entity_type)
                annotation = EntityAnnotation(
                    text=entity_text,
                    type=entity_type,
                )
                entity_annotations.append(annotation)

        # Create relation annotations
        relation_annotations = []
        relation_types = set()

        for rel in relations:
            relation_type = rel.get("relation", "").strip()
            if relation_type:
                relation_types.add(relation_type)

            annotation = RelationAnnotation(
                head=rel.get("head", ""),
                head_type=rel.get("head_type", ""),
                relation=relation_type,
                tail=rel.get("tail", ""),
                tail_type=rel.get("tail_type", ""),
            )
            relation_annotations.append(annotation)

        # Determine task types
        task_types = []
        if entity_annotations:
            task_types.append("entity")
        if relation_annotations:
            task_types.append("relation")

        # Create schema definition
        schema_def = SchemaDefinition(
            entity=sorted(list(entity_types)),
            relation=sorted(list(relation_types)),
        )

        # Create answer
        answer = Answer(
            entity=entity_annotations,
            relation=relation_annotations,
        )

        # Create metadata
        meta = MetaInfo(
            dataset="instructie",
            license="placeholder - verify before commercial use",
            split=split,
            notes=f"Category: {category}" if category else "",
        )

        # Create canonical record
        record = CanonicalIERecord(
            id=record_id,
            text=text,
            lang=lang,
            source="HuggingFace:KnowLM/InstructIE",
            task_types=task_types,
            schema_def=schema_def,
            answer=answer,
            meta=meta,
        )

        return record

    except Exception as e:
        logger.error(f"Error converting record {parsed.get('id', '?')}: {str(e)}")
        raise ValueError(f"Failed to convert record: {str(e)}")


def convert_file(
    input_path: Path | str,
    output_path: Path | str,
    split: str = "train",
) -> dict:
    """Convert all records in an InstructIE JSONL file to canonical format.

    Args:
        input_path: Path to input InstructIE JSONL file.
        output_path: Path to output canonical JSONL file.
        split: Data split (train/dev/test). Used to infer language if possible.

    Returns:
        Dictionary with conversion statistics:
        - total: Total records processed
        - success: Number of successfully converted records
        - failed: Number of failed records
        - by_task_type: Count by task type (entity, relation, both)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {input_path.name} to canonical format...")

    # Parse input file
    try:
        parsed_records = parse_instructie_file(input_path)
    except Exception as e:
        logger.error(f"Failed to parse input file: {str(e)}")
        raise

    # Convert records
    canonical_records = []
    stats = {
        "total": len(parsed_records),
        "success": 0,
        "failed": 0,
        "by_task_type": {"entity_only": 0, "relation_only": 0, "both": 0},
    }

    for parsed in parsed_records:
        try:
            record = convert_record(parsed, split=split)
            canonical_records.append(record.to_canonical_dict())

            # Update task type stats
            if len(record.task_types) == 1:
                if "entity" in record.task_types:
                    stats["by_task_type"]["entity_only"] += 1
                else:
                    stats["by_task_type"]["relation_only"] += 1
            elif len(record.task_types) == 2:
                stats["by_task_type"]["both"] += 1

            stats["success"] += 1

        except Exception as e:
            logger.debug(f"Failed to convert record: {str(e)}")
            stats["failed"] += 1
            continue

    # Write output file
    if canonical_records:
        try:
            write_jsonl(canonical_records, output_path)
            logger.info(
                f"Converted {len(canonical_records)} records to {output_path}"
            )
        except Exception as e:
            logger.error(f"Failed to write output file: {str(e)}")
            raise

    logger.info(
        f"Conversion complete: {stats['success']}/{stats['total']} successful "
        f"({stats['failed']} failed)"
    )

    return stats


def convert_dataset(
    raw_dir: Path | str,
    output_dir: Path | str,
    splits: Optional[list[str]] = None,
) -> dict:
    """Convert all InstructIE splits in a directory to canonical format.

    Automatically discovers .jsonl files in the input directory and converts
    them, inferring split names from filenames.

    Args:
        raw_dir: Directory containing raw InstructIE JSONL files.
        output_dir: Directory where canonical JSONL files will be saved.
        splits: Optional list of specific splits to convert. If None, converts all
                .jsonl files found in raw_dir.

    Returns:
        Dictionary with overall conversion statistics:
        - splits: Dict mapping split names to conversion results
        - total_records: Total records across all splits
        - total_success: Total successful conversions
        - total_failed: Total failed conversions
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting InstructIE dataset from {raw_dir}...")

    # Find input files
    if splits:
        # Use specified splits
        input_files = []
        for split in splits:
            split_file = raw_dir / f"{split}.jsonl"
            if split_file.exists():
                input_files.append((split, split_file))
            else:
                logger.warning(f"Split file not found: {split_file}")
    else:
        # Find all .jsonl files
        input_files = []
        for file_path in sorted(raw_dir.glob("*.jsonl")):
            # Extract split name from filename (e.g., "train_en.jsonl" -> "train_en")
            split_name = file_path.stem
            input_files.append((split_name, file_path))

    if not input_files:
        logger.warning(f"No .jsonl files found in {raw_dir}")
        return {
            "splits": {},
            "total_records": 0,
            "total_success": 0,
            "total_failed": 0,
        }

    # Convert each split
    overall_stats = {
        "splits": {},
        "total_records": 0,
        "total_success": 0,
        "total_failed": 0,
    }

    for split_name, input_file in input_files:
        try:
            output_file = output_dir / f"{split_name}.jsonl"
            stats = convert_file(input_file, output_file, split=split_name)

            overall_stats["splits"][split_name] = stats
            overall_stats["total_records"] += stats["total"]
            overall_stats["total_success"] += stats["success"]
            overall_stats["total_failed"] += stats["failed"]

        except Exception as e:
            logger.error(f"Failed to convert split {split_name}: {str(e)}")
            overall_stats["splits"][split_name] = {
                "total": 0,
                "success": 0,
                "failed": 0,
                "error": str(e),
            }
            continue

    # Log summary
    logger.info(
        f"Dataset conversion complete:\n"
        f"  Total records: {overall_stats['total_records']}\n"
        f"  Successful: {overall_stats['total_success']}\n"
        f"  Failed: {overall_stats['total_failed']}"
    )

    return overall_stats


if __name__ == "__main__":
    # Example usage and testing
    example_parsed = {
        "id": "instructie-001",
        "text": "Apple Inc. is headquartered in Cupertino, California.",
        "category": "company",
        "entities": {"Apple Inc.": "ORGANIZATION", "Cupertino": "LOCATION"},
        "relations": [
            {
                "head": "Apple Inc.",
                "head_type": "ORGANIZATION",
                "relation": "headquartered_in",
                "tail": "Cupertino",
                "tail_type": "LOCATION",
            }
        ],
    }

    print("Example Conversion:")
    record = convert_record(example_parsed, split="train_en")
    print(json.dumps(record.to_canonical_dict(), indent=2, ensure_ascii=False))
