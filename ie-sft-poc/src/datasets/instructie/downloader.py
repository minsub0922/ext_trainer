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

    HF_REPO = "zjunlp/InstructIE"
    AVAILABLE_SUBSETS = ["en", "zh", "all"]

    # The zjunlp/InstructIE repo ships the data as top-level per-language
    # JSON files (e.g. ``train_en.json``, ``valid_en.json``). Its splits
    # have non-uniform columns (the validation file uses ``input`` while
    # train/test use ``text``/``entity``), which makes the standard
    # ``datasets.load_dataset(..., split="train_en")`` path fail mid-way.
    # We therefore download the raw files directly via ``hf_hub_download``
    # and normalize them into JSONL ourselves.
    #
    # Layout: (lang, remote_filename, output_filename)
    # Some older snapshots use ``dev_*`` instead of ``valid_*``, so we
    # list both and pick whichever resolves successfully.
    REMOTE_FILES: list[tuple[str, list[str], str]] = [
        ("en", ["train_en.json"], "train_en.jsonl"),
        ("en", ["valid_en.json", "dev_en.json"], "dev_en.jsonl"),
        ("en", ["test_en.json"], "test_en.jsonl"),
        ("zh", ["train_zh.json"], "train_zh.jsonl"),
        ("zh", ["valid_zh.json", "dev_zh.json"], "dev_zh.jsonl"),
        ("zh", ["test_zh.json"], "test_zh.jsonl"),
    ]

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
            source_url="https://huggingface.co/datasets/zjunlp/InstructIE",
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

        Downloads the raw per-language JSON files from the zjunlp/InstructIE
        repository directly (via ``huggingface_hub.hf_hub_download``) and
        converts them into JSONL for downstream processing. This avoids the
        unreliable split-name auto-detection of ``datasets.load_dataset`` for
        this particular repo layout.

        Args:
            output_dir: Directory where the dataset will be saved.

        Returns:
            True if at least one split was successfully downloaded, else False.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.error(
                "The 'huggingface_hub' library is required for downloading "
                "InstructIE. Install with: pip install huggingface_hub"
            )
            return False

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter to the requested language subset.
        if self.subset == "all":
            files = list(self.REMOTE_FILES)
        else:
            files = [f for f in self.REMOTE_FILES if f[0] == self.subset]

        logger.info(
            f"Downloading InstructIE dataset (subset={self.subset}, "
            f"{len(files)} file(s)) from {self.HF_REPO}..."
        )

        total_records = 0
        any_success = False

        for lang, candidate_names, out_name in files:
            logger.info(
                f"  Fetching {out_name} (lang={lang}, candidates={candidate_names})..."
            )

            local_path: Path | None = None
            resolved_name: str | None = None
            last_error: Exception | None = None
            for candidate in candidate_names:
                try:
                    p = hf_hub_download(
                        repo_id=self.HF_REPO,
                        filename=candidate,
                        repo_type="dataset",
                    )
                    local_path = Path(p)
                    resolved_name = candidate
                    break
                except Exception as e:  # 404 or transient error -> try next
                    last_error = e
                    continue

            if local_path is None:
                logger.warning(
                    f"    Failed to fetch any of {candidate_names}: {last_error}"
                )
                continue

            logger.info(f"    Resolved remote file: {resolved_name}")

            try:
                records = self._read_records(local_path)
            except Exception as e:
                logger.warning(f"    Failed to parse {resolved_name}: {e}")
                continue

            # Tag each record with its language for downstream steps.
            out_path = output_dir / out_name
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in records:
                    if isinstance(rec, dict):
                        rec.setdefault("lang", lang)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n = len(records)
            total_records += n
            any_success = True
            logger.info(f"    Saved {n} records to {out_path}")

        self._save_metadata(output_dir)
        logger.info(f"Download complete. Total records: {total_records}")
        return any_success

    @staticmethod
    def _read_records(path: Path) -> list:
        """Read a file that may be JSON-array or JSONL into a list of records.

        The zjunlp/InstructIE files are distributed as a mix of formats
        depending on the version (JSON array, JSONL, or JSON-with-root-key).
        This helper normalizes all three into a list of dicts.
        """
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []

        # Fast path: JSONL (one JSON object per line).
        if text[0] == "{" and "\n" in text:
            records: list = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # Fall through to JSON-array parsing below.
                    records = []
                    break
            if records:
                return records

        # Full-file JSON (array or object-with-data-key).
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "records", "examples", "train", "test", "valid"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            # Single object -> wrap in list.
            return [data]
        return []

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
        metadata_dict["source"] = "HuggingFace:zjunlp/InstructIE"
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
