"""Dataset registry for managing available datasets.

Provides a simple registry pattern for discovering and managing datasets
with support for registration, retrieval, and listing.
"""

from typing import Optional

from src.datasets.metadata import DatasetMetadata


class DatasetRegistry:
    """Registry for managing available datasets."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registry: dict[str, DatasetMetadata] = {}

    def register(self, name: str, metadata: DatasetMetadata) -> None:
        """Register a dataset.

        Args:
            name: Dataset identifier
            metadata: Dataset metadata

        Raises:
            ValueError: If dataset with this name already exists
        """
        if name in self._registry:
            raise ValueError(f"Dataset '{name}' is already registered")

        # Validate that name matches metadata name
        if metadata.name != name:
            metadata.name = name

        self._registry[name] = metadata

    def get(self, name: str) -> Optional[DatasetMetadata]:
        """Get dataset metadata by name.

        Args:
            name: Dataset identifier

        Returns:
            DatasetMetadata if found, None otherwise
        """
        return self._registry.get(name)

    def list(self, enabled_only: bool = False) -> list[str]:
        """List all registered dataset names.

        Args:
            enabled_only: If True, only return enabled datasets

        Returns:
            List of dataset names
        """
        if enabled_only:
            return [
                name
                for name, meta in self._registry.items()
                if meta.default_enabled
            ]
        return list(self._registry.keys())

    def list_metadata(self, enabled_only: bool = False) -> dict[str, DatasetMetadata]:
        """Get all registered datasets with their metadata.

        Args:
            enabled_only: If True, only return enabled datasets

        Returns:
            Dictionary mapping dataset names to metadata
        """
        if enabled_only:
            return {
                name: meta
                for name, meta in self._registry.items()
                if meta.default_enabled
            }
        return self._registry.copy()

    def unregister(self, name: str) -> bool:
        """Unregister a dataset.

        Args:
            name: Dataset identifier

        Returns:
            True if unregistered, False if not found
        """
        if name in self._registry:
            del self._registry[name]
            return True
        return False

    def exists(self, name: str) -> bool:
        """Check if a dataset is registered.

        Args:
            name: Dataset identifier

        Returns:
            True if registered, False otherwise
        """
        return name in self._registry

    def __len__(self) -> int:
        """Get number of registered datasets.

        Returns:
            Number of datasets in registry
        """
        return len(self._registry)

    def __repr__(self) -> str:
        """Get string representation."""
        return f"DatasetRegistry(datasets={self.list()})"


# Global registry instance
_global_registry = DatasetRegistry()


def register_dataset(name: str, metadata: DatasetMetadata) -> None:
    """Register a dataset in the global registry.

    Args:
        name: Dataset identifier
        metadata: Dataset metadata

    Raises:
        ValueError: If dataset with this name already exists
    """
    _global_registry.register(name, metadata)


def get_dataset(name: str) -> Optional[DatasetMetadata]:
    """Get dataset metadata from the global registry.

    Args:
        name: Dataset identifier

    Returns:
        DatasetMetadata if found, None otherwise
    """
    return _global_registry.get(name)


def list_datasets(enabled_only: bool = False) -> list[str]:
    """List all datasets in the global registry.

    Args:
        enabled_only: If True, only return enabled datasets

    Returns:
        List of dataset names
    """
    return _global_registry.list(enabled_only=enabled_only)


def list_datasets_with_metadata(
    enabled_only: bool = False,
) -> dict[str, DatasetMetadata]:
    """Get all datasets with their metadata from the global registry.

    Args:
        enabled_only: If True, only return enabled datasets

    Returns:
        Dictionary mapping dataset names to metadata
    """
    return _global_registry.list_metadata(enabled_only=enabled_only)


def unregister_dataset(name: str) -> bool:
    """Unregister a dataset from the global registry.

    Args:
        name: Dataset identifier

    Returns:
        True if unregistered, False if not found
    """
    return _global_registry.unregister(name)


def dataset_exists(name: str) -> bool:
    """Check if a dataset is registered in the global registry.

    Args:
        name: Dataset identifier

    Returns:
        True if registered, False otherwise
    """
    return _global_registry.exists(name)


# Pre-register built-in datasets
_INSTRUCTIE_METADATA = DatasetMetadata(
    name="instructie",
    source_url="https://github.com/Microsoft/InstructIE",
    task_types=["entity", "relation"],
    license="MIT",
    default_enabled=True,
    description="Large-scale multi-lingual Information Extraction dataset",
    notes="Supports 12 languages, multi-domain IE tasks",
    version="1.0",
    citation="Luo et al., InstructIE: A Unified Instruction Set for IE Tasks, 2024",
    num_records=5000000,
)

_INTERNAL_KV_TEMPLATE_METADATA = DatasetMetadata(
    name="internal_kv_template",
    source_url="",
    task_types=["kv"],
    license="Internal",
    default_enabled=True,
    description="Internal template for key-value extraction tasks",
    notes="Used as a template for creating custom KV extraction datasets",
    version="1.0",
    citation="",
    num_records=0,
)

# Register the pre-built datasets
register_dataset("instructie", _INSTRUCTIE_METADATA)
register_dataset("internal_kv_template", _INTERNAL_KV_TEMPLATE_METADATA)


if __name__ == "__main__":
    # Example usage
    print("Available Datasets:")
    print("=" * 60)

    for name in list_datasets():
        metadata = get_dataset(name)
        if metadata:
            print(f"\nDataset: {metadata.name}")
            print(f"  Description: {metadata.description}")
            print(f"  Task Types: {', '.join(metadata.task_types)}")
            print(f"  License: {metadata.license}")
            print(f"  Enabled: {metadata.default_enabled}")
            if metadata.num_records:
                print(f"  Records: {metadata.num_records:,}")
