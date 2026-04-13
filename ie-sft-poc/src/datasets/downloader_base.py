"""Base class for dataset downloaders.

Provides an abstract interface that specific dataset downloaders should implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from src.datasets.metadata import DatasetMetadata


class DatasetDownloaderBase(ABC):
    """Abstract base class for dataset downloaders.

    Subclasses should implement download, verify, and metadata property.
    """

    @abstractmethod
    def download(self, output_dir: Path | str) -> bool:
        """Download the dataset to the specified directory.

        Args:
            output_dir: Directory where the dataset will be saved

        Returns:
            True if download was successful, False otherwise

        Raises:
            Exception: For network errors, permission issues, etc.
        """
        pass

    @abstractmethod
    def verify(self, output_dir: Path | str) -> bool:
        """Verify that the dataset was downloaded correctly.

        Should check for file integrity, completeness, and format validity.

        Args:
            output_dir: Directory containing the downloaded dataset

        Returns:
            True if dataset is complete and valid, False otherwise
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Get the metadata for this dataset.

        Returns:
            DatasetMetadata instance
        """
        pass

    def download_and_verify(self, output_dir: Path | str) -> tuple[bool, str]:
        """Download and verify the dataset in one call.

        Args:
            output_dir: Directory where the dataset will be saved

        Returns:
            Tuple of (success, message)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            download_success = self.download(output_dir)
            if not download_success:
                return False, "Download failed"

            verify_success = self.verify(output_dir)
            if not verify_success:
                return False, "Verification failed"

            return True, "Download and verification successful"

        except Exception as e:
            return False, f"Error: {str(e)}"

    def __repr__(self) -> str:
        """Get string representation."""
        return f"{self.__class__.__name__}(dataset={self.metadata.name})"


class SimpleDownloader(DatasetDownloaderBase):
    """Simple concrete downloader for demonstration/testing.

    This can be used as a base for actual implementations.
    """

    def __init__(self, metadata: DatasetMetadata) -> None:
        """Initialize with metadata.

        Args:
            metadata: Dataset metadata
        """
        self._metadata = metadata

    @property
    def metadata(self) -> DatasetMetadata:
        """Get metadata."""
        return self._metadata

    def download(self, output_dir: Path | str) -> bool:
        """Download implementation (stub).

        Args:
            output_dir: Target directory

        Returns:
            False (to be overridden by subclasses)
        """
        return False

    def verify(self, output_dir: Path | str) -> bool:
        """Verify implementation (stub).

        Args:
            output_dir: Directory to verify

        Returns:
            False (to be overridden by subclasses)
        """
        return False


if __name__ == "__main__":
    from src.datasets.registry import get_dataset

    # Example of creating a downloader
    instructie_metadata = get_dataset("instructie")

    if instructie_metadata:
        downloader = SimpleDownloader(instructie_metadata)
        print(f"Created downloader: {downloader}")
        print(f"Dataset: {downloader.metadata.name}")
        print(f"Description: {downloader.metadata.description}")
