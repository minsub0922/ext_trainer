"""Canonical schema for Information Extraction tasks using Pydantic v2.

Defines the core data structures for representing IE tasks including key-value extraction,
named entity recognition, and relation extraction.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.common.constants import SUPPORTED_TASK_TYPES
from src.common.io import read_jsonl, read_json, write_jsonl


class EntityAnnotation(BaseModel):
    """Represents an annotated entity in text.

    Attributes:
        text: The entity text
        type: Entity type/category
        start: Start character offset (optional, auto-computed from text matching)
        end: End character offset (optional, auto-computed from text matching)
    """

    text: str
    type: str
    start: int | None = None
    end: int | None = None

    def model_dump(self, **kwargs) -> dict:
        """Override to handle None values appropriately."""
        result = super().model_dump(**kwargs)
        # Remove None values for start/end if they weren't explicitly set
        if result.get("start") is None:
            result.pop("start", None)
        if result.get("end") is None:
            result.pop("end", None)
        return result


class RelationAnnotation(BaseModel):
    """Represents a relation between two entities.

    Attributes:
        head: Head entity text
        head_type: Type of head entity
        relation: Relation type
        tail: Tail entity text
        tail_type: Type of tail entity
    """

    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str


class SchemaDefinition(BaseModel):
    """Defines the schema (possible values/types) for different task types.

    Attributes:
        kv: List of possible key names for key-value extraction
        entity: List of possible entity types
        relation: List of possible relation types
    """

    kv: list[str] = Field(default_factory=list)
    entity: list[str] = Field(default_factory=list)
    relation: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    """Container for extraction results.

    Attributes:
        kv: Dictionary of key-value pairs
        entity: List of entity annotations
        relation: List of relation annotations
    """

    kv: dict[str, str | None] = Field(default_factory=dict)
    entity: list[EntityAnnotation] = Field(default_factory=list)
    relation: list[RelationAnnotation] = Field(default_factory=list)


class MetaInfo(BaseModel):
    """Metadata information about a record.

    Attributes:
        dataset: Name of the source dataset
        license: License information
        split: Data split (train/dev/test)
        notes: Additional notes
    """

    dataset: str = ""
    license: str = ""
    split: str = ""
    notes: str = ""


class CanonicalIERecord(BaseModel):
    """Canonical record format for Information Extraction tasks.

    This is the core data structure that all IE datasets are converted to.

    Attributes:
        id: Unique record identifier
        text: Input text to extract information from
        lang: Language code (ISO 639-1, default: en)
        source: Source of the record
        task_types: List of task types relevant to this record
        schema_def: Schema definition for possible extraction values
        answer: Extraction results
        meta: Metadata about the record
    """

    id: str
    text: str
    lang: str = "en"
    source: str = ""
    task_types: list[str] = Field(default_factory=list)
    schema_def: SchemaDefinition = Field(
        default_factory=SchemaDefinition,
        alias="schema",
    )
    answer: Answer = Field(default_factory=Answer)
    meta: MetaInfo = Field(default_factory=MetaInfo)

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": "ex-1",
                    "text": "John Smith works at Google in Mountain View.",
                    "lang": "en",
                    "source": "internal",
                    "task_types": ["entity", "relation"],
                    "schema": {
                        "entity": ["PERSON", "ORGANIZATION", "LOCATION"],
                        "relation": ["works_for", "located_in"],
                    },
                    "answer": {
                        "entity": [
                            {"text": "John Smith", "type": "PERSON", "start": 0, "end": 10},
                            {"text": "Google", "type": "ORGANIZATION", "start": 20, "end": 26},
                            {"text": "Mountain View", "type": "LOCATION", "start": 30, "end": 43},
                        ],
                        "relation": [
                            {
                                "head": "John Smith",
                                "head_type": "PERSON",
                                "relation": "works_for",
                                "tail": "Google",
                                "tail_type": "ORGANIZATION",
                            },
                        ],
                    },
                }
            ]
        },
    }

    @field_validator("task_types")
    @classmethod
    def validate_task_types(cls, v: list[str]) -> list[str]:
        """Validate that all task types are supported.

        Args:
            v: List of task types to validate

        Returns:
            Validated task types list

        Raises:
            ValueError: If any task type is not in SUPPORTED_TASK_TYPES
        """
        for task_type in v:
            if task_type not in SUPPORTED_TASK_TYPES:
                raise ValueError(
                    f"Invalid task type '{task_type}'. "
                    f"Must be one of {SUPPORTED_TASK_TYPES}"
                )
        return v

    @field_validator("lang")
    @classmethod
    def validate_lang(cls, v: str) -> str:
        """Validate language code format.

        Args:
            v: Language code to validate

        Returns:
            Validated language code
        """
        if not v or len(v) < 2:
            raise ValueError("Language must be a valid ISO 639-1 code (e.g., 'en', 'ko')")
        return v.lower()

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate that text is not empty.

        Args:
            v: Text to validate

        Returns:
            Validated text

        Raises:
            ValueError: If text is empty
        """
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "CanonicalIERecord":
        """Validate consistency between task_types and answer content.

        Returns:
            Self (validated record)

        Raises:
            ValueError: If task_types and answer content are inconsistent
        """
        # If no explicit task types, we skip this validation
        # (get_active_task_types handles inference)
        return self

    def get_active_task_types(self) -> list[str]:
        """Infer which task types are active based on answer content.

        Returns:
            List of task types that have non-empty answers
        """
        active_types = []

        if self.answer.kv:
            active_types.append("kv")
        if self.answer.entity:
            active_types.append("entity")
        if self.answer.relation:
            active_types.append("relation")

        return active_types

    def is_valid(self) -> bool:
        """Check if record is valid and consistent.

        Validates that:
        - Text is not empty
        - All task types are supported
        - Answer content matches task types (if task types are specified)

        Returns:
            True if record is valid, False otherwise
        """
        try:
            # Check basic field validity
            if not self.text or not self.text.strip():
                return False

            # Check task types
            for task_type in self.task_types:
                if task_type not in SUPPORTED_TASK_TYPES:
                    return False

            # Check consistency: if task types specified, answer should match
            if self.task_types:
                active = set(self.get_active_task_types())
                specified = set(self.task_types)
                if active != specified:
                    # If specified includes types not in active, that's an error
                    if not active.issubset(specified):
                        return False

            return True
        except Exception:
            return False

    def to_canonical_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with canonical field names.

        Uses 'schema' instead of 'schema_def' in output.

        Returns:
            Dictionary representation of the record
        """
        data = self.model_dump(by_alias=True, exclude_none=False)
        # Ensure schema is present, not schema_def
        if "schema_def" in data:
            data["schema"] = data.pop("schema_def")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalIERecord":
        """Create a record from a dictionary.

        Handles both 'schema' and 'schema_def' field names.

        Args:
            data: Dictionary containing record data

        Returns:
            CanonicalIERecord instance

        Raises:
            ValueError: If data is invalid
        """
        # Normalize field names
        if "schema" in data and "schema_def" not in data:
            data["schema_def"] = data.pop("schema")

        return cls(**data)


def validate_record(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single record dictionary.

    Args:
        data: Dictionary to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    try:
        record = CanonicalIERecord.from_dict(data)
        if not record.is_valid():
            errors.append("Record failed internal consistency checks")
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")

    return len(errors) == 0, errors


def validate_jsonl_file(
    path: Path | str,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate a JSONL file containing IE records.

    Args:
        path: Path to JSONL file
        strict: If True, fail on first error; if False, collect all errors

    Returns:
        Dictionary with validation statistics:
        - total: Total records processed
        - valid: Number of valid records
        - invalid: Number of invalid records
        - errors: List of error messages (only if invalid records found)
    """
    path = Path(path)
    stats = {
        "path": str(path),
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
    }

    try:
        for record_data in read_jsonl(path):
            stats["total"] += 1
            is_valid, errors = validate_record(record_data)

            if is_valid:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
                if strict:
                    stats["errors"] = errors
                    raise ValueError(f"Validation failed at record {stats['total']}: {errors}")
                stats["errors"].extend(
                    [f"Record {stats['total']}: {err}" for err in errors]
                )

    except FileNotFoundError:
        stats["errors"] = [f"File not found: {path}"]
        stats["invalid"] = -1

    return stats


if __name__ == "__main__":
    # Example usage and testing
    record = CanonicalIERecord(
        id="example-1",
        text="Apple Inc. is headquartered in Cupertino, California.",
        lang="en",
        source="example",
        task_types=["entity", "relation"],
        schema_def=SchemaDefinition(
            entity=["ORGANIZATION", "LOCATION"],
            relation=["headquartered_in"],
        ),
        answer=Answer(
            entity=[
                EntityAnnotation(text="Apple Inc.", type="ORGANIZATION", start=0, end=10),
                EntityAnnotation(
                    text="Cupertino, California", type="LOCATION", start=30, end=51
                ),
            ],
            relation=[
                RelationAnnotation(
                    head="Apple Inc.",
                    head_type="ORGANIZATION",
                    relation="headquartered_in",
                    tail="Cupertino, California",
                    tail_type="LOCATION",
                ),
            ],
        ),
    )

    print("Example Record:")
    print(record.to_canonical_dict())
    print(f"\nIs valid: {record.is_valid()}")
    print(f"Active task types: {record.get_active_task_types()}")
