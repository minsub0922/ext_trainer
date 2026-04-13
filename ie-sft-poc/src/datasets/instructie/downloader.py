"""InstructIE dataset downloader.

Downloads the InstructIE dataset from HuggingFace and saves it locally
with appropriate metadata.
"""

import json
from pathlib import Path
from typing import Optional

from src.common.logging_utils import get_logger
from src.datasets.downloader_base import DatasetDownloaderBase
from src.datasets.metadata import DatasetMetadata

logger = get_logger(__name__)


class InstructIEDownloader(DatasetDownloaderBase):
    """Downloader for the InstructIE dataset from HuggingFace.

    InstructIE is a large-scale, multi-lingual Information Extraction dataset
    hosted at: https://huggingface.co/datasets/KnowLM/InstructIE

    Supports downloading English (en) and Chinese (zh) splits.
    """

    HF_REPO = "KnowLM/InstructIE"
    AVAILABLE_SUBSETS = ["en", "zh", "all"]

    def __init__(self, subset: str = "all") -> None:
        """Initialize the InstructIE downloader.

        Args:
            subset: Which subset to download - "en", "zh", or "all".
                   Defaults to "all".

        Raises:
            ValueError: If subset is not a valid option.
        """
        if subset not in self.AVAILABLE_SUBSETS:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of {self.AVAILABLE_SUBSETS}"
            )
        self.subset = subset
        self._metadata = self._create_metadata()

    def _create_metadata(self) -> DatasetMetadata:
        """Create metadata for the dataset.

        Returns:
            DatasetMetadata instance for InstructIE.
        """
        return DatasetMetadata(
            name="instructie",
            source_url="https://huggingface.co/datasets/KnowLM/InstructIE",
            task_types=["entity", "relation"],
            license="placeholder - verify before commercial use",
            default_enabled=True,
            description=(
                "Large-scale multi-lingual Information Extraction dataset "
                "covering entity and relation extraction tasks across 12 languages "
                "and multiple domains."
            ),
            notes=(
                f"Subset: {self.subset}. Supports English and Chinese. "
                "Multiple domain coverage including news, finance, medicine, etc."
            ),
            version="1.0",
            citation="Luo et al., InstructIE: A Unified Instruction Set for IE Tasks, 2024",
        )

    @property
    def metadata(self) -> DatasetMetadata:
        """Get the metadata for this dataset.

        Returns:
            DatasetMetadata instance.
        """
        return self._metadata

    def download(self, output_dir: Path | str) -> bool:
        """Download the InstructIE dataset from HuggingFace.

        Args:
            output_dir: Directory where the dataset will be saved.

        Returns:
            True if download was successful, False otherwise.

        Raises:
            ImportError: If the datasets library is not installed.
            Exception: For network errors or HuggingFace API issues.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error(
                "The 'datasets' library is required for downloading InstructIE. "
                "Install it with: pip install datasets"
            )
            return False

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading InstructIE dataset (subset={self.subset})...")

            # Determine which splits to download
            splits_to_download = []
            if self.subset == "en":
                splits_to_download = ["train_en", "dev_en", "test_en"]
            elif self.subset == "zh":
                splits_to_download = ["train_zh", "dev_zh", "test_zh"]
            else:  # all
                splits_to_download = [
                    "train_en",
                    "dev_en",
                    "test_en",
                    "train_zh",
                    "dev_zh",
                    "test_zh",
                ]

            total_records = 0
            for split in splits_to_download:
                logger.info(f"  Downloading split: {split}...")
                try:
                    dataset = load_dataset(self.HF_REPO, split=split)
                    split_path = output_dir / f"{split}.jsonl"

                    # Save as JSONL
                    with open(split_path, "w", encoding="utf-8") as f:
                        for record in dataset:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    records_count = len(dataset)
                    total_records += records_count
                    logger.info(f"    Saved {records_count} records to {split_path}")

                except Exception as e:
                    logger.warning(
                        f"    Failed to download split {split}: {str(e)}"
                    )
                    continue

            # Save metadata
            self._save_metadata(output_dir)

            logger.info(
                f"Download complete. Total records: {total_records} "
                f"(in {len(splits_to_download)} splits)"
            )
            return total_records > 0

        except Exception as e:
            logger.error(f"Error downloading InstructIE: {str(e)}")
            return False

    def verify(self, output_dir: Path | str) -> bool:
        """Verify that the dataset was downloaded correctly.

        Checks for:
        - JSONL files exist for expected splits
        - Metadata file exists
        - Files contain valid JSON records
        - Files are non-empty

        Args:
            output_dir: Directory containing the downloaded dataset.

        Returns:
            True if dataset is valid, False otherwise.
        """
        output_dir = Path(output_dir)

        if not output_dir.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            return False

        # Check for metadata
        metadata_file = output_dir / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return False

        # Check for at least one split file
        split_files = list(output_dir.glob("*.jsonl"))
        if not split_files:
            logger.error(f"No JSONL files found in {output_dir}")
            return False

        # Verify each split file
        all_valid = True
        total_records = 0

        for split_file in split_files:
            try:
                record_count = 0
                with open(split_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)
                            record_count += 1
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Invalid JSON on line {line_num} "
                                f"of {split_file.name}: {e}"
                            )
                            all_valid = False
                            break

                if record_count == 0:
                    logger.error(f"File {split_file.name} is empty")
                    all_valid = False
                else:
                    total_records += record_count
                    logger.info(f"Verified {split_file.name}: {record_count} records")

            except Exception as e:
                logger.error(f"Error verifying {split_file.name}: {str(e)}")
                all_valid = False

        if all_valid:
            logger.info(f"Verification passed. Total records: {total_records}")

        return all_valid

    def _save_metadata(self, output_dir: Path) -> None:
        """Save dataset metadata to a JSON file.

        Args:
            output_dir: Directory where metadata will be saved.
        """
        metadata_file = output_dir / "metadata.json"
        metadata_dict = self.metadata.to_dict()
        metadata_dict["source"] = "HuggingFace:KnowLM/InstructIE"
        metadata_dict["task_types"] = ["entity", "relation"]
        metadata_dict["download_timestamp"] = str(Path(__file__).stat().st_mtime)

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    import tempfile

    # Example usage
    with tempfile.TemporaryDirectory() as tmpdir:
        downloader = InstructIEDownloader(subset="all")

        print(f"Downloader: {downloader}")
        print(f"Metadata: {downloader.metadata}")
        print(f"Available subsets: {downloader.AVAILABLE_SUBSETS}")

        # In real usage, this would download the dataset
        # success = downloader.download(tmpdir)
        # if success:
        #     verified = downloader.verify(tmpdir)
        #     print(f"Verification result: {verified}")
