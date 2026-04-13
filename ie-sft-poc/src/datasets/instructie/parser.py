"""Parser for InstructIE raw data format.

Converts raw InstructIE JSON records into intermediate Python dictionaries
that can be converted to the canonical IE record format.

InstructIE format structure:
- id: Unique record identifier
- cate: Category/domain
- text: Input text
- relation: List of relations with:
  - head: Head entity text
  - head_type: Head entity type
  - relation: Relation type
  - tail: Tail entity text
  - tail_type: Tail entity type
"""

import json
from pathlib import Path
from typing import Any, Optional

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def parse_instructie_record(raw: dict) -> dict:
    """Parse a single InstructIE record from raw format.

    Extracts entities from relation mentions and normalizes the record
    to an intermediate format.

    Args:
        raw: Raw record dictionary from InstructIE.

    Returns:
        Parsed record dictionary with normalized fields.

    Raises:
        ValueError: If record is missing required fields.
    """
    try:
        # Extract required fields
        record_id = raw.get("id")
        text = raw.get("text", "")
        category = raw.get("cate", "")
        relations = raw.get("relation", [])

        if not record_id:
            raise ValueError("Record missing 'id' field")

        if not text or not text.strip():
            raise ValueError(f"Record {record_id}: text is empty")

        # Extract entities from relations
        entities = {}  # Maps entity text to type
        relation_list = []

        for rel in relations:
            if not isinstance(rel, dict):
                logger.warning(f"Record {record_id}: relation is not a dict, skipping")
                continue

            head = rel.get("head", "").strip()
            head_type = rel.get("head_type", "").strip()
            relation_type = rel.get("relation", "").strip()
            tail = rel.get("tail", "").strip()
            tail_type = rel.get("tail_type", "").strip()

            # Validate relation fields
            if not all([head, head_type, relation_type, tail, tail_type]):
                logger.debug(
                    f"Record {record_id}: skipping incomplete relation "
                    f"({head}, {head_type}, {relation_type}, {tail}, {tail_type})"
                )
                continue

            # Track entities
            if head not in entities:
                entities[head] = head_type
            if tail not in entities:
                entities[tail] = tail_type

            # Add relation
            relation_list.append(
                {
                    "head": head,
                    "head_type": head_type,
                    "relation": relation_type,
                    "tail": tail,
                    "tail_type": tail_type,
                }
            )

        return {
            "id": record_id,
            "text": text,
            "category": category,
            "entities": entities,
            "relations": relation_list,
        }

    except ValueError as e:
        logger.warning(f"Parse error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing record: {str(e)}")
        raise ValueError(f"Failed to parse record: {str(e)}")


def parse_instructie_file(path: Path | str) -> list[dict]:
    """Parse a complete InstructIE JSONL file.

    Reads all records from a JSONL file and parses them. Handles both
    English and Chinese splits.

    Args:
        path: Path to the InstructIE JSONL file.

    Returns:
        List of parsed record dictionaries.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    parsed_records = []
    skipped_count = 0
    error_count = 0

    logger.info(f"Parsing InstructIE file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            try:
                # Parse JSON
                raw_record = json.loads(line)

                # Parse record
                try:
                    parsed = parse_instructie_record(raw_record)
                    parsed_records.append(parsed)
                except ValueError:
                    skipped_count += 1
                    continue

            except json.JSONDecodeError as e:
                logger.error(
                    f"Invalid JSON at line {line_num} in {path.name}: {str(e)}"
                )
                error_count += 1
                continue
            except Exception as e:
                logger.error(f"Error at line {line_num}: {str(e)}")
                error_count += 1
                continue

    logger.info(
        f"Parsed {len(parsed_records)} records from {path.name} "
        f"(skipped={skipped_count}, errors={error_count})"
    )

    if error_count > 0:
        logger.warning(f"Encountered {error_count} errors while parsing")

    return parsed_records


if __name__ == "__main__":
    # Example usage and testing
    example_raw = {
        "id": "instructie-001",
        "cate": "news",
        "text": "John Smith works at Google in Mountain View, California.",
        "relation": [
            {
                "head": "John Smith",
                "head_type": "PERSON",
                "relation": "works_for",
                "tail": "Google",
                "tail_type": "ORGANIZATION",
            },
            {
                "head": "Google",
                "head_type": "ORGANIZATION",
                "relation": "located_in",
                "tail": "Mountain View",
                "tail_type": "LOCATION",
            },
        ],
    }

    print("Example Parsing:")
    parsed = parse_instructie_record(example_raw)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    print(f"\nExtracted entities: {parsed['entities']}")
    print(f"Extracted relations: {len(parsed['relations'])}")
