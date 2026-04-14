"""Build LLaMA-Factory dataset_info.json registry."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetEntry:
    """Entry for a dataset in LLaMA-Factory registry.

    Attributes:
        name: Dataset name (must be unique in registry)
        file_name: Filename or glob pattern (relative to dataset folder)
        file_sha1: Optional SHA1 hash of the file for verification
        columns: Mapping of column names (must include 'prompt', 'query', 'response')
    """

    name: str
    file_name: str
    columns: dict[str, str]
    file_sha1: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        entry = {
            "file_name": self.file_name,
            "columns": self.columns,
        }
        if self.file_sha1:
            entry["file_sha1"] = self.file_sha1
        return entry


def compute_file_sha1(file_path: Path) -> str:
    """Compute SHA1 hash of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA1 hexdigest
    """
    sha1 = hashlib.sha1()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def build_dataset_info(
    dataset_name: str = "ie_dataset",
    output_dir: Path | str | None = None,
    file_names: list[str] | None = None,
    compute_sha1: bool = False,
) -> dict[str, Any]:
    """Build dataset_info.json for LLaMA-Factory.

    Creates a registry that LLaMA-Factory uses to identify and load datasets.
    Typically includes entries for train.jsonl, dev.jsonl, and test.jsonl.

    Args:
        dataset_name: Name of the dataset in the registry
        output_dir: Directory containing dataset files (for SHA1 computation)
        file_names: List of file names to include (default: ['train.jsonl', 'dev.jsonl', 'test.jsonl'])
        compute_sha1: Whether to compute SHA1 hashes of files

    Returns:
        Dictionary formatted for LLaMA-Factory dataset_info.json
    """
    if file_names is None:
        file_names = ["train.jsonl", "dev.jsonl", "test.jsonl"]

    logger.info(f"Building dataset info for '{dataset_name}'")

    # LLaMA-Factory column mapping for alpaca-style records.
    standard_columns = {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
    }

    # LLaMA-Factory expects a FLAT dataset_info.json:
    #   {
    #     "<name>":      {"file_name": "train.jsonl", "columns": {...}},
    #     "<name>_dev":  {"file_name": "dev.jsonl",   "columns": {...}},
    #     "<name>_test": {"file_name": "test.jsonl",  "columns": {...}}
    #   }
    # The bare `<name>` key is what the YAML's `dataset:` field references.
    registry: dict[str, Any] = {}

    for file_name in file_names:
        fname_low = file_name.lower()
        if "train" in fname_low:
            key = dataset_name  # bare name = train split (YAML default)
        elif "dev" in fname_low or "validation" in fname_low:
            key = f"{dataset_name}_dev"
        elif "test" in fname_low:
            key = f"{dataset_name}_test"
        else:
            key = f"{dataset_name}_{Path(file_name).stem}"

        entry: dict[str, Any] = {
            "file_name": file_name,
            "columns": dict(standard_columns),
        }

        if compute_sha1 and output_dir:
            output_path = Path(output_dir) / file_name
            if output_path.exists():
                entry["file_sha1"] = compute_file_sha1(output_path)
            else:
                logger.warning(f"file not found for sha1: {output_path}")

        registry[key] = entry

    logger.info(f"built flat registry with {len(registry)} entries: {list(registry)}")
    return registry


def write_dataset_info(
    entries: list[DatasetEntry],
    output_path: Path | str,
) -> None:
    """Write dataset entries to LLaMA-Factory dataset_info.json.

    Args:
        entries: List of DatasetEntry objects
        output_path: Path to write JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build registry from entries
    registry = {}

    for entry in entries:
        if entry.name not in registry:
            registry[entry.name] = {
                "hf_hub_url": "",
                "ms_hub_url": "",
                "script_url": "",
            }

        registry[entry.name].update(entry.to_dict())

    # Write to file
    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote dataset info to {output_path}")


if __name__ == "__main__":
    # Example usage
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample files
        for split in ["train", "dev", "test"]:
            filepath = Path(tmpdir) / f"{split}.jsonl"
            filepath.write_text('{"instruction": "test", "input": "test", "output": "test"}\n')

        # Build registry
        registry = build_dataset_info(
            dataset_name="example_dataset",
            output_dir=tmpdir,
            compute_sha1=True,
        )

        print("Generated registry:")
        print(json.dumps(registry, indent=2))
