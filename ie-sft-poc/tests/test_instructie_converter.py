"""Tests for InstructIE parsing and conversion."""

import pytest

from src.datasets.instructie.parser import parse_instructie_record
from src.datasets.instructie.converter import convert_record
from src.common.schema import CanonicalIERecord


class TestParseInstructIE:
    """Test InstructIE record parsing."""

    def test_parse_basic_record(self, sample_instructie_raw):
        """Test parsing a basic InstructIE record."""
        parsed = parse_instructie_record(sample_instructie_raw)

        assert parsed["id"] == "instructie-001"
        assert parsed["text"] == "Steve Jobs founded Apple Computer Company in 1976."
        assert parsed["category"] == "news"
        assert "Steve Jobs" in parsed["entities"]
        assert "Apple Computer Company" in parsed["entities"]
        assert "1976" in parsed["entities"]
        assert len(parsed["relations"]) == 2

    def test_parse_missing_relations(self):
        """Test parsing record with no relations."""
        raw = {
            "id": "instructie-002",
            "cate": "news",
            "text": "Just some text.",
            "relation": [],
        }
        parsed = parse_instructie_record(raw)

        assert parsed["id"] == "instructie-002"
        assert parsed["text"] == "Just some text."
        assert len(parsed["entities"]) == 0
        assert len(parsed["relations"]) == 0

    def test_parse_missing_id_raises_error(self):
        """Test that missing id raises ValueError."""
        raw = {
            "cate": "news",
            "text": "Some text",
            "relation": [],
        }
        with pytest.raises(ValueError, match="Record missing"):
            parse_instructie_record(raw)

    def test_parse_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        raw = {
            "id": "test-001",
            "cate": "news",
            "text": "",
            "relation": [],
        }
        with pytest.raises(ValueError, match="text is empty"):
            parse_instructie_record(raw)

    def test_parse_incomplete_relations_skipped(self):
        """Test that incomplete relations are skipped."""
        raw = {
            "id": "test-001",
            "cate": "news",
            "text": "Some text",
            "relation": [
                {
                    "head": "A",
                    "head_type": "TYPE_A",
                    "relation": "rel",
                    "tail": "B",
                    "tail_type": "TYPE_B",
                },
                {
                    "head": "C",
                    "head_type": "",  # Missing tail_type
                    "relation": "rel",
                    "tail": "D",
                    "tail_type": "TYPE_D",
                },
            ],
        }
        parsed = parse_instructie_record(raw)

        # Only the complete relation should be included
        assert len(parsed["relations"]) == 1
        assert parsed["relations"][0]["head"] == "A"

    def test_parse_entity_extraction_from_relations(self):
        """Test that entities are correctly extracted from relations."""
        raw = {
            "id": "test-001",
            "cate": "news",
            "text": "A works for B",
            "relation": [
                {
                    "head": "A",
                    "head_type": "PERSON",
                    "relation": "works_for",
                    "tail": "B",
                    "tail_type": "ORGANIZATION",
                },
            ],
        }
        parsed = parse_instructie_record(raw)

        assert "A" in parsed["entities"]
        assert parsed["entities"]["A"] == "PERSON"
        assert "B" in parsed["entities"]
        assert parsed["entities"]["B"] == "ORGANIZATION"


class TestConvertRecord:
    """Test conversion from parsed to canonical format."""

    def test_convert_record_basic(self):
        """Test converting a basic parsed record."""
        parsed = {
            "id": "test-001",
            "text": "Steve Jobs founded Apple.",
            "category": "news",
            "entities": {
                "Steve Jobs": "PERSON",
                "Apple": "ORGANIZATION",
            },
            "relations": [
                {
                    "head": "Steve Jobs",
                    "head_type": "PERSON",
                    "relation": "founded",
                    "tail": "Apple",
                    "tail_type": "ORGANIZATION",
                },
            ],
        }

        record = convert_record(parsed, split="train_en")

        assert isinstance(record, CanonicalIERecord)
        assert record.id == "test-001"
        assert record.text == "Steve Jobs founded Apple."
        assert record.lang == "en"
        assert record.source == "HuggingFace:KnowLM/InstructIE"
        assert "entity" in record.task_types
        assert "relation" in record.task_types

    def test_convert_record_language_detection(self):
        """Test that language is correctly inferred from split name."""
        parsed = {
            "id": "test-zh",
            "text": "某文本",
            "category": "news",
            "entities": {},
            "relations": [],
        }

        # Test English split
        record_en = convert_record(parsed, split="train_en")
        assert record_en.lang == "en"

        # Test Chinese split
        record_zh = convert_record(parsed, split="train_zh")
        assert record_zh.lang == "zh"

    def test_convert_record_entity_only(self):
        """Test converting record with only entities."""
        parsed = {
            "id": "entity-only",
            "text": "Apple Inc.",
            "category": "company",
            "entities": {"Apple Inc.": "ORGANIZATION"},
            "relations": [],
        }

        record = convert_record(parsed)

        assert "entity" in record.task_types
        assert "relation" not in record.task_types
        assert len(record.answer.entity) == 1
        assert len(record.answer.relation) == 0

    def test_convert_record_relation_only(self):
        """Test converting record with only relations."""
        parsed = {
            "id": "relation-only",
            "text": "John works at Google",
            "category": "news",
            "entities": {
                "John": "PERSON",
                "Google": "ORGANIZATION",
            },
            "relations": [
                {
                    "head": "John",
                    "head_type": "PERSON",
                    "relation": "works_for",
                    "tail": "Google",
                    "tail_type": "ORGANIZATION",
                },
            ],
        }

        record = convert_record(parsed)

        assert "relation" in record.task_types
        assert "entity" not in record.task_types
        assert len(record.answer.entity) == 0
        assert len(record.answer.relation) == 1

    def test_convert_record_preserves_metadata(self):
        """Test that metadata is correctly set."""
        parsed = {
            "id": "test-001",
            "text": "Test text",
            "category": "test_category",
            "entities": {},
            "relations": [],
        }

        record = convert_record(parsed, split="dev")

        assert record.meta.dataset == "instructie"
        assert record.meta.split == "dev"
        assert "test_category" in record.meta.notes

    def test_convert_record_schema_definition(self):
        """Test that schema definition is correctly built."""
        parsed = {
            "id": "test-001",
            "text": "A, B, C",
            "category": "test",
            "entities": {
                "A": "TYPE_A",
                "B": "TYPE_B",
                "C": "TYPE_A",  # Duplicate type
            },
            "relations": [
                {
                    "head": "A",
                    "head_type": "TYPE_A",
                    "relation": "rel1",
                    "tail": "B",
                    "tail_type": "TYPE_B",
                },
                {
                    "head": "B",
                    "head_type": "TYPE_B",
                    "relation": "rel2",
                    "tail": "C",
                    "tail_type": "TYPE_A",
                },
            ],
        }

        record = convert_record(parsed)

        # Check entity types are deduplicated and sorted
        assert set(record.schema_def.entity) == {"TYPE_A", "TYPE_B"}
        assert "TYPE_A" in record.schema_def.entity

        # Check relation types are deduplicated and sorted
        assert set(record.schema_def.relation) == {"rel1", "rel2"}

    def test_convert_record_creates_valid_canonical(self, sample_instructie_raw):
        """Test that converted record is a valid CanonicalIERecord."""
        parsed = parse_instructie_record(sample_instructie_raw)
        record = convert_record(parsed)

        # Should be valid
        assert record.is_valid()

        # Should be serializable
        data = record.to_canonical_dict()
        assert data is not None

        # Should be deserializable
        record2 = CanonicalIERecord.from_dict(data)
        assert record2.id == record.id
        assert record2.text == record.text


class TestParseAndConvertFlow:
    """Test the full parse -> convert flow."""

    def test_parse_and_convert_workflow(self, sample_instructie_raw):
        """Test complete workflow: raw -> parsed -> canonical."""
        # Parse
        parsed = parse_instructie_record(sample_instructie_raw)
        assert parsed is not None
        assert "entities" in parsed
        assert "relations" in parsed

        # Convert
        record = convert_record(parsed, split="train_en")
        assert isinstance(record, CanonicalIERecord)

        # Validate
        assert record.is_valid()
        assert record.lang == "en"
        assert record.source == "HuggingFace:KnowLM/InstructIE"

        # Serialize and deserialize
        data = record.to_canonical_dict()
        record2 = CanonicalIERecord.from_dict(data)
        assert record2.id == record.id
        assert record2.text == record.text
        assert len(record2.answer.entity) == len(record.answer.entity)
        assert len(record2.answer.relation) == len(record.answer.relation)
