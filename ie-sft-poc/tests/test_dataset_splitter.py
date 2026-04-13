"""Tests for dataset splitting functionality."""

import pytest

from src.datasets.unified.splitter import split_dataset
from src.common.io import read_jsonl


class TestDatasetSplitting:
    """Test dataset splitting into train/dev/test splits."""

    def test_split_ratios(self, tmp_path, tmp_jsonl_file, sample_kv_record, sample_entity_record):
        """Test that split ratios are approximately correct."""
        # Create 100 test records
        records = [
            {**sample_kv_record, "id": f"record-{i}"}
            for i in range(100)
        ]

        # Write to temp file
        input_file = tmp_jsonl_file(records, "unified.jsonl")

        # Split with standard ratios (0.8, 0.1, 0.1)
        output_dir = tmp_path / "splits"
        stats = split_dataset(
            input_path=input_file,
            output_dir=output_dir,
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        # Verify stats
        assert stats.total == 100
        assert stats.train_count == 80
        assert stats.dev_count == 10
        assert stats.test_count == 10

        # Verify files exist
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "dev.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()

        # Verify file contents
        train_records = read_jsonl(output_dir / "train.jsonl")
        dev_records = read_jsonl(output_dir / "dev.jsonl")
        test_records = read_jsonl(output_dir / "test.jsonl")

        assert len(train_records) == 80
        assert len(dev_records) == 10
        assert len(test_records) == 10

    def test_split_reproducibility(self, tmp_path, tmp_jsonl_file, sample_kv_record):
        """Test that same seed produces same split."""
        records = [
            {**sample_kv_record, "id": f"record-{i}"}
            for i in range(100)
        ]
        input_file = tmp_jsonl_file(records, "unified.jsonl")

        # First split with seed=42
        output_dir1 = tmp_path / "splits1"
        split_dataset(
            input_path=input_file,
            output_dir=output_dir1,
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        train1 = read_jsonl(output_dir1 / "train.jsonl")
        train1_ids = {r["id"] for r in train1}

        # Second split with same seed
        output_dir2 = tmp_path / "splits2"
        split_dataset(
            input_path=input_file,
            output_dir=output_dir2,
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        train2 = read_jsonl(output_dir2 / "train.jsonl")
        train2_ids = {r["id"] for r in train2}

        # Same seed should produce same split
        assert train1_ids == train2_ids

    def test_split_different_seeds(self, tmp_path, tmp_jsonl_file, sample_kv_record):
        """Test that different seeds produce different splits."""
        records = [
            {**sample_kv_record, "id": f"record-{i}"}
            for i in range(100)
        ]
        input_file = tmp_jsonl_file(records, "unified.jsonl")

        # Split with seed=42
        output_dir1 = tmp_path / "splits1"
        split_dataset(
            input_path=input_file,
            output_dir=output_dir1,
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )
        train1 = read_jsonl(output_dir1 / "train.jsonl")
        train1_ids = {r["id"] for r in train1}

        # Split with seed=123
        output_dir2 = tmp_path / "splits2"
        split_dataset(
            input_path=input_file,
            output_dir=output_dir2,
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=123,
        )
        train2 = read_jsonl(output_dir2 / "train.jsonl")
        train2_ids = {r["id"] for r in train2}

        # Different seeds should (likely) produce different splits
        # Note: with small datasets there's a chance they could be the same
        # but with 100 records and different seeds, they should differ
        assert train1_ids != train2_ids

    def test_split_writes_files(self, tmp_path, tmp_jsonl_file, sample_kv_record):
        """Test that split creates all required output files."""
        records = [
            {**sample_kv_record, "id": f"record-{i}"}
            for i in range(50)
        ]
        input_file = tmp_jsonl_file(records, "unified.jsonl")
        output_dir = tmp_path / "splits"

        split_dataset(
            input_path=input_file,
            output_dir=output_dir,
            train_ratio=0.7,
            dev_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        # Check files exist
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "dev.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()

        # Check files are not empty
        assert (output_dir / "train.jsonl").stat().st_size > 0
        assert (output_dir / "dev.jsonl").stat().st_size > 0
        assert (output_dir / "test.jsonl").stat().st_size > 0

    def test_split_preserves_all_records(self, tmp_path, tmp_jsonl_file, sample_kv_record):
        """Test that no records are lost during splitting."""
        records = [
            {**sample_kv_record, "id": f"record-{i}"}
            for i in range(100)
        ]
        input_file = tmp_jsonl_file(records, "unified.jsonl")
        output_dir = tmp_path / "splits"

        original_ids = {r["id"] for r in records}

        split_dataset(
            input_path=input_file,
            output_dir=output_dir,
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        # Read all split files
        train_records = read_jsonl(output_dir / "train.jsonl")
        dev_records = read_jsonl(output_dir / "dev.jsonl")
        test_records = read_jsonl(output_dir / "test.jsonl")

        # Collect all IDs from splits
        split_ids = set()
        for record in train_records + dev_records + test_records:
            split_ids.add(record["id"])

        # All original IDs should be in splits
        assert original_ids == split_ids

    def test_split_no_overlap_between_splits(self, tmp_path, tmp_jsonl_file, sample_kv_record):
        """Test that no record appears in multiple splits."""
        records = [
            {**sample_kv_record, "id": f"record-{i}"}
            for i in range(100)
        ]
        input_file = tmp_jsonl_file(records, "unified.jsonl")
        output_dir = tmp_path / "splits"

        split_dataset(
            input_path=input_file,
            output_dir=output_dir,
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        # Read all split files
        train_records = read_jsonl(output_dir / "train.jsonl")
        dev_records = read_jsonl(output_dir / "dev.jsonl")
        test_records = read_jsonl(output_dir / "test.jsonl")

        train_ids = {r["id"] for r in train_records}
        dev_ids = {r["id"] for r in dev_records}
        test_ids = {r["id"] for r in test_records}

        # No overlap between splits
        assert len(train_ids & dev_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(dev_ids & test_ids) == 0

    def test_split_with_custom_ratios(self, tmp_path, tmp_jsonl_file, sample_kv_record):
        """Test splitting with custom ratios."""
        records = [
            {**sample_kv_record, "id": f"record-{i}"}
            for i in range(100)
        ]
        input_file = tmp_jsonl_file(records, "unified.jsonl")
        output_dir = tmp_path / "splits"

        stats = split_dataset(
            input_path=input_file,
            output_dir=output_dir,
            train_ratio=0.6,
            dev_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        assert stats.train_count == 60
        assert stats.dev_count == 20
        assert stats.test_count == 20

    def test_split_single_split(self, tmp_path, tmp_jsonl_file, sample_kv_record):
        """Test edge case: all data in one split."""
        records = [
            {**sample_kv_record, "id": f"record-{i}"}
            for i in range(10)
        ]
        input_file = tmp_jsonl_file(records, "unified.jsonl")
        output_dir = tmp_path / "splits"

        stats = split_dataset(
            input_path=input_file,
            output_dir=output_dir,
            train_ratio=1.0,
            dev_ratio=0.0,
            test_ratio=0.0,
            seed=42,
        )

        assert stats.train_count == 10
        assert stats.dev_count == 0
        assert stats.test_count == 0


class TestSplitWithMultipleTaskTypes:
    """Test splitting with records having different task types."""

    def test_split_mixed_task_types(self, tmp_path, tmp_jsonl_file,
                                     sample_kv_record, sample_entity_record,
                                     sample_relation_record):
        """Test that splitting preserves records with different task types."""
        records = []
        for i in range(10):
            records.append({**sample_kv_record, "id": f"kv-{i}"})
        for i in range(10):
            records.append({**sample_entity_record, "id": f"entity-{i}"})
        for i in range(10):
            records.append({**sample_relation_record, "id": f"relation-{i}"})

        input_file = tmp_jsonl_file(records, "mixed.jsonl")
        output_dir = tmp_path / "splits"

        stats = split_dataset(
            input_path=input_file,
            output_dir=output_dir,
            train_ratio=0.8,
            dev_ratio=0.1,
            test_ratio=0.1,
            seed=42,
        )

        assert stats.total == 30
        assert stats.train_count + stats.dev_count + stats.test_count == 30

        # Verify records are distributed
        train = read_jsonl(output_dir / "train.jsonl")
        dev = read_jsonl(output_dir / "dev.jsonl")
        test = read_jsonl(output_dir / "test.jsonl")

        all_records = train + dev + test
        assert len(all_records) == 30

        # Check we have different task types
        task_types = set()
        for record in all_records:
            task_types.update(record.get("task_types", []))

        assert "kv" in task_types
        assert "entity" in task_types
        assert "relation" in task_types
