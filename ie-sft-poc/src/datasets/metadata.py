"""Dataset metadata and information models.

Defines the structure for storing and accessing dataset metadata including
name, source, supported tasks, and licensing information.
"""

from pydantic import BaseModel, Field

from src.common.constants import SUPPORTED_TASK_TYPES


class DatasetMetadata(BaseModel):
    """Metadata for a dataset in the registry.

    Attributes:
        name: Dataset name (unique identifier)
        source_url: URL to the original dataset source
        task_types: List of IE task types supported by this dataset
        license: License information
        default_enabled: Whether this dataset is enabled by default
        description: Human-readable description of the dataset
        notes: Additional notes and caveats
        version: Dataset version identifier
        citation: Citation information for academic use
        num_records: Expected number of records (optional)
    """

    name: str = Field(..., description="Dataset identifier")
    source_url: str = Field(default="", description="URL to dataset source")
    task_types: list[str] = Field(
        default_factory=lambda: ["entity"],
        description="Supported IE task types",
    )
    license: str = Field(default="", description="License type")
    default_enabled: bool = Field(default=True, description="Enabled by default")
    description: str = Field(default="", description="Dataset description")
    notes: str = Field(default="", description="Additional notes")
    version: str = Field(default="1.0.0", description="Dataset version")
    citation: str = Field(default="", description="Citation information")
    num_records: int | None = Field(default=None, description="Number of records")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "name": "instructie",
                "source_url": "https://github.com/Microsoft/InstructIE",
                "task_types": ["entity", "relation"],
                "license": "MIT",
                "default_enabled": True,
                "description": "Large-scale Information Extraction dataset",
                "notes": "Multi-lingual, multi-domain dataset",
                "version": "1.0",
                "citation": "Luo et al., 2024",
                "num_records": 5000000,
            }
        }

    def validate_task_types(self) -> bool:
        """Validate that all task types are supported.

        Returns:
            True if all task types are valid

        Raises:
            ValueError: If any task type is not supported
        """
        invalid_types = [t for t in self.task_types if t not in SUPPORTED_TASK_TYPES]
        if invalid_types:
            raise ValueError(
                f"Invalid task types: {invalid_types}. "
                f"Must be one of {SUPPORTED_TASK_TYPES}"
            )
        return True

    def to_dict(self) -> dict:
        """Convert metadata to dictionary.

        Returns:
            Dictionary representation of metadata
        """
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetMetadata":
        """Create metadata from dictionary.

        Args:
            data: Dictionary containing metadata fields

        Returns:
            DatasetMetadata instance

        Raises:
            ValueError: If data is invalid
        """
        return cls(**data)


if __name__ == "__main__":
    # Example usage
    instructie_metadata = DatasetMetadata(
        name="instructie",
        source_url="https://github.com/Microsoft/InstructIE",
        task_types=["entity", "relation"],
        license="MIT",
        default_enabled=True,
        description="Large-scale Information Extraction dataset by Microsoft",
        notes="Multi-lingual, covers 12 languages, multi-domain",
        version="1.0",
        citation="Luo et al., InstructIE: A Unified Instruction Set for IE Tasks, 2024",
        num_records=5000000,
    )

    print("Dataset Metadata Example:")
    print(instructie_metadata.model_dump_json(indent=2))
