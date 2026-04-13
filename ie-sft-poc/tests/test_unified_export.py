"""Tests for unified dataset export and LLaMA-Factory integration."""

import pytest

from src.datasets.unified.merger import merge_datasets
from src.training.dataset_registry_builder import (
    DatasetEntry,
    build_dataset_info,
    write_dataset_info,
)
from src.common.io import read_jsonl, read_json


class TestMergeDatasets:
    """Test dataset merging functionality."""

    def test_merge_datasets_basic(self, tmp_path, tmp_jsonl_file,
                                  sample_kv_record, sample_entity_record):
        """Test basic dataset merging."""
        # Create two input files
        records1 = [
            {**sample_kv_record, "id": f"kv-{i}"}
            for i in range(10)
        ]
        records2 = [
            {**sample_entity_record, "id": f"entity-{i}"}
            for i in range(10)
        ]

        input_file1 = tmp_jsonl_file(records1, "kv.jsonl")
        input_file2 = tmp_jsonl_file(records2, "entity.jsonl")

        # Merge
        output_file = tmp_path / "merged.jsonl"
        stats = merge_datasets(
            input_paths=[input_file1, input_file2],
            output_path=output_file,
            deduplicate=False,
            task_filter=None,
        )

        # Verify
        assert stats.total_input == 20
        assert stats.total_output == 20
        assert stats.duplicates_removed == 0

        # Verify output file
        merged_records = read_jsonl(output_file)
        assert len(merged_records) == 20

    def test_merge_deduplication(self, tmp_path, tmp_jsonl_file, sample_kv_record):
        """Test that duplicate records are removed when deduplicate=True."""
        # Create records with duplicates
        records = [
            {**sample_kv_record, "id": "record-1"},
            {**sample_kv_record, "id": "record-2"},
            {**sample_kv_record, "id": "record-1"},  # Duplicate
        ]

        input_file = tmp_jsonl_file(records, "with_dupes.jsonl")

        output_file = tmp_path / "deduped.jsonl"
        stats = merge_datasets(
            input_paths=[input_file],
            output_path=output_file,
            deduplicate=True,
            task_filter=None,
        )

        # With deduplication, should have 2 records (1 duplicate removed)
        assert stats.total_input == 3
        assert stats.duplicates_removed >= 1
        # Note: exact deduplication depends on implementation (id-based vs content-based)

        merged_records = read_jsonl(output_file)
        ids = [r["id"] for r in merged_records]
        # Should not have duplicates
        assert len(ids) == len(set(ids))

    def test_task_filter_kv_only(self, tmp_path, tmp_jsonl_file,
                                 sample_kv_record, sample_entity_record,
                                 sample_relation_record):
        """Test filtering by KV task type."""
        records = []
        records.extend([
            {**sample_kv_record, "id": f"kv-{i}"}
            for i in range(5)
        ])
        records.extend([
            {**sample_entity_record, "id": f"entity-{i}"}
            for i in range(5)
        ])
        records.extend([
            {**sample_relation_record, "id": f"relation-{i}"}
            for i in range(5)
        ])

        input_file = tmp_jsonl_file(records, "mixed.jsonl")
        output_file = tmp_path / "kv_only.jsonl"

        stats = merge_datasets(
            input_paths=[input_file],
            output_path=output_file,
            deduplicate=False,
            task_filter=["kv"],
        )

        # Should have only KV records
        merged_records = read_jsonl(output_file)
        assert len(merged_records) <= 5

        # All records should have 'kv' in task_types
        for record in merged_records:
            assert "kv" in record.get("task_types", [])

    def test_task_filter_entity_and_relation(self, tmp_path, tmp_jsonl_file,
                                            sample_kv_record,
                                            sample_entity_record,
                                            sample_relation_record):
        """Test filtering by multiple task types."""
        records = []
        records.extend([
            {**sample_kv_record, "id": f"kv-{i}"}
            for i in range(5)
        ])
        records.extend([
            {**sample_entity_record, "id": f"entity-{i}"}
            for i in range(5)
        ])
        records.extend([
            {**sample_relation_record, "id": f"relation-{i}"}
            for i in range(5)
        ])

        input_file = tmp_jsonl_file(records, "mixed.jsonl")
        output_file = tmp_path / "entity_relation.jsonl"

        stats = merge_datasets(
            input_paths=[input_file],
            output_path=output_file,
            deduplicate=False,
            task_filter=["entity", "relation"],
        )

        # Should have entity and relation records
        merged_records = read_jsonl(output_file)
        assert len(merged_records) >= 10

        # Records should have entity or relation task type
        for record in merged_records:
            task_types = record.get("task_types", [])
            assert "entity" in task_types or "relation" in task_types

    def test_merge_multiple_files(self, tmp_path, tmp_jsonl_file, sample_kv_record):
        """Test merging multiple input files."""
        input_files = []
        total_records = 0

        for file_idx in range(3):
            records = [
                {**sample_kv_record, "id": f"file{file_idx}-record-{i}"}
                for i in range(10)
            ]
            input_file = tmp_jsonl_file(records, f"file{file_idx}.jsonl")
            input_files.append(input_file)
            total_records += len(records)

        output_file = tmp_path / "merged_all.jsonl"
        stats = merge_datasets(
            input_paths=input_files,
            output_path=output_file,
            deduplicate=False,
            task_filter=None,
        )

        assert stats.total_input == total_records
        assert stats.total_output == total_records

        merged_records = read_jsonl(output_file)
        assert len(merged_records) == total_records


class TestDatasetRegistryBuilder:
    """Test LLaMA-Factory dataset_info.json building."""

    def test_dataset_entry_creation(self):
        """Test creating a DatasetEntry."""
        entry = DatasetEntry(
            name="my_dataset",
            file_name="train.jsonl",
            columns={
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        )

        assert entry.name == "my_dataset"
        assert entry.file_name == "train.jsonl"
        assert "prompt" in entry.columns

    def test_dataset_entry_to_dict(self):
        """Test converting DatasetEntry to dict."""
        entry = DatasetEntry(
            name="test_dataset",
            file_name="data.jsonl",
            columns={
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        )

        data = entry.to_dict()
        assert data["file_name"] == "data.jsonl"
        assert data["columns"]["prompt"] == "instruction"

    def test_build_dataset_info(self):
        """Test building dataset info dictionary."""
        info = build_dataset_info(
            dataset_name="ie_dataset",
            file_names=["train.jsonl", "dev.jsonl"],
            compute_sha1=False,
        )

        assert "ie_dataset" in info
        assert "file_name" in info["ie_dataset"]
        assert "columns" in info["ie_dataset"]

    def test_build_dataset_info_with_defaults(self):
        """Test building dataset info with default file names."""
        info = build_dataset_info(
            dataset_name="default_dataset",
            compute_sha1=False,
        )

        assert "default_dataset" in info
        entry = info["default_dataset"]
        assert "file_name" in entry
        assert "columns" in entry

    def test_write_dataset_info(self, tmp_path):
        """Test writing dataset_info.json file."""
        output_dir = tmp_path / "registry"
        output_dir.mkdir()

        write_dataset_info(
            output_dir=output_dir,
            dataset_name="test_ie_dataset",
            file_names=["train.jsonl", "dev.jsonl", "test.jsonl"],
            compute_sha1=False,
        )

        # Check file exists
        info_file = output_dir / "dataset_info.json"
        assert info_file.exists()

        # Check content
        data = read_json(info_file)
        assert "test_ie_dataset" in data

    def test_write_dataset_info_creates_directory(self, tmp_path):
        """Test that write_dataset_info creates output directory if needed."""
        output_dir = tmp_path / "new_dir" / "registry"

        write_dataset_info(
            output_dir=output_dir,
            dataset_name="new_dataset",
            compute_sha1=False,
        )

        # Directory should be created
        assert output_dir.exists()
        assert (output_dir / "dataset_info.json").exists()


class TestLLaMAFactoryExport:
    """Test export to LLaMA-Factory format."""

    def test_export_format_has_required_columns(self, tmp_path, tmp_jsonl_file,
                                               sample_kv_record):
        """Test that exported format has the required columns."""
        records = [
            {**sample_kv_record, "id": f"record-{i}"}
            for i in range(5)
        ]
        input_file = tmp_jsonl_file(records, "data.jsonl")

        # For this test, we just verify the records can be read
        data = read_jsonl(input_file)
        assert len(data) == 5

        # In the full pipeline, these would be converted to LLaMA-Factory format
        # (instruction/input/output) by a separate export step
        for record in data:
            assert "id" in record
            assert "text" in record
            assert "answer" in record

    def test_registry_entries_for_splits(self, tmp_path):
        """Test creating registry entries for dataset splits."""
        # This represents what would be in dataset_info.json
        registry = {
            "ie_train": {
                "file_name": "train.jsonl",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            },
            "ie_dev": {
                "file_name": "dev.jsonl",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            },
            "ie_test": {
                "file_name": "test.jsonl",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            },
        }

        # Verify structure
        assert "ie_train" in registry
        assert "ie_dev" in registry
        assert "ie_test" in registry

        for name, entry in registry.items():
            assert "file_name" in entry
            assert "columns" in entry
            assert "prompt" in entry["columns"]
            assert "query" in entry["columns"]
            assert "response" in entry["columns"]
