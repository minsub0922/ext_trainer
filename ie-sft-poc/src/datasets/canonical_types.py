"""Type definitions and enums for datasets and canonical IE format.

Provides enumeration types for task types and data splits, and re-exports
schema types for convenience.
"""

from enum import Enum

from src.common.schema import (
    Answer,
    CanonicalIERecord,
    EntityAnnotation,
    MetaInfo,
    RelationAnnotation,
    SchemaDefinition,
)


class TaskType(str, Enum):
    """Enumeration of supported IE task types."""

    KV = "kv"
    ENTITY = "entity"
    RELATION = "relation"

    def __str__(self) -> str:
        """Return the string value."""
        return self.value


class DatasetSplit(str, Enum):
    """Enumeration of standard dataset splits."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    VALIDATION = "validation"

    def __str__(self) -> str:
        """Return the string value."""
        return self.value

    @classmethod
    def from_string(cls, value: str) -> "DatasetSplit":
        """Create from string, normalizing common variants.

        Args:
            value: String representation

        Returns:
            DatasetSplit enum value

        Raises:
            ValueError: If value is not a valid split
        """
        normalized = value.lower().strip()

        # Handle common variants
        if normalized in ("val", "validation"):
            return cls.VALIDATION
        if normalized == "dev":
            return cls.DEV
        if normalized == "train":
            return cls.TRAIN
        if normalized == "test":
            return cls.TEST

        raise ValueError(f"Invalid split value: {value}. Must be one of {list(cls)}")


# Re-export schema types for convenience
__all__ = [
    "TaskType",
    "DatasetSplit",
    # Schema types
    "CanonicalIERecord",
    "EntityAnnotation",
    "RelationAnnotation",
    "SchemaDefinition",
    "Answer",
    "MetaInfo",
]
