"""Tests for canonical schema and CanonicalIERecord."""

import pytest

from src.common.schema import (
    CanonicalIERecord,
    EntityAnnotation,
    RelationAnnotation,
    SchemaDefinition,
    Answer,
    MetaInfo,
    validate_record,
    validate_jsonl_file,
)


class TestValidRecords:
    """Test creating and validating various record types."""

    def test_valid_kv_record(self, sample_kv_record):
        """Test that a valid KV record passes validation."""
        record = CanonicalIERecord.from_dict(sample_kv_record)
        assert record.is_valid()
        assert record.id == "kv-001"
        assert "kv" in record.task_types
        assert len(record.answer.kv) == 3
        assert record.answer.kv["name"] == "John Smith"

    def test_valid_entity_record(self, sample_entity_record):
        """Test that a valid entity record passes validation."""
        record = CanonicalIERecord.from_dict(sample_entity_record)
        assert record.is_valid()
        assert record.id == "entity-001"
        assert "entity" in record.task_types
        assert len(record.answer.entity) == 3
        assert record.answer.entity[0].text == "Apple Inc."
        assert record.answer.entity[0].type == "ORGANIZATION"

    def test_valid_relation_record(self, sample_relation_record):
        """Test that a valid relation record passes validation."""
        record = CanonicalIERecord.from_dict(sample_relation_record)
        assert record.is_valid()
        assert record.id == "relation-001"
        assert "relation" in record.task_types
        assert len(record.answer.relation) == 2
        assert record.answer.relation[0].head == "John Smith"
        assert record.answer.relation[0].relation == "works_for"

    def test_valid_unified_record(self, sample_unified_record):
        """Test a record with all three task types."""
        record = CanonicalIERecord.from_dict(sample_unified_record)
        assert record.is_valid()
        assert len(record.task_types) == 3
        assert "kv" in record.task_types
        assert "entity" in record.task_types
        assert "relation" in record.task_types
        assert len(record.answer.kv) == 3
        assert len(record.answer.entity) == 3
        assert len(record.answer.relation) == 1

    def test_empty_answer(self):
        """Test that a record with empty answers is valid."""
        record = CanonicalIERecord(
            id="empty-001",
            text="Some text with no extracted information.",
            task_types=[],
        )
        assert record.is_valid()
        assert len(record.answer.kv) == 0
        assert len(record.answer.entity) == 0
        assert len(record.answer.relation) == 0


class TestValidation:
    """Test validation functions."""

    def test_invalid_task_type(self):
        """Test that invalid task type raises validation error."""
        with pytest.raises(ValueError, match="Invalid task type"):
            CanonicalIERecord(
                id="invalid-001",
                text="Test text",
                task_types=["invalid_type"],
            )

    def test_invalid_lang(self):
        """Test that invalid language code raises validation error."""
        with pytest.raises(ValueError, match="Language"):
            CanonicalIERecord(
                id="invalid-lang",
                text="Test text",
                lang="",
            )

    def test_empty_text(self):
        """Test that empty text raises validation error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            CanonicalIERecord(
                id="empty-text",
                text="",
            )

    def test_whitespace_only_text(self):
        """Test that whitespace-only text raises validation error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            CanonicalIERecord(
                id="whitespace-text",
                text="   ",
            )

    def test_missing_id(self):
        """Test that missing id raises validation error."""
        with pytest.raises(ValueError):
            CanonicalIERecord(
                id="",
                text="Test text",
            )

    def test_validate_record_function(self, sample_kv_record):
        """Test the validate_record function."""
        is_valid, errors = validate_record(sample_kv_record)
        assert is_valid
        assert len(errors) == 0

    def test_validate_record_with_errors(self):
        """Test validate_record with invalid data."""
        invalid_record = {
            "id": "invalid",
            "text": "",  # Empty text should fail
            "task_types": ["kv"],
        }
        is_valid, errors = validate_record(invalid_record)
        assert not is_valid
        assert len(errors) > 0


class TestGetActiveTaskTypes:
    """Test task type inference from answer content."""

    def test_get_active_task_types_kv(self):
        """Test that KV task type is inferred from answer content."""
        record = CanonicalIERecord(
            id="kv-001",
            text="Name: John",
            answer=Answer(kv={"name": "John"}),
        )
        active = record.get_active_task_types()
        assert "kv" in active
        assert "entity" not in active
        assert "relation" not in active

    def test_get_active_task_types_entity(self):
        """Test that entity task type is inferred."""
        record = CanonicalIERecord(
            id="entity-001",
            text="John is a person",
            answer=Answer(
                entity=[EntityAnnotation(text="John", type="PERSON")]
            ),
        )
        active = record.get_active_task_types()
        assert "entity" in active
        assert "kv" not in active
        assert "relation" not in active

    def test_get_active_task_types_relation(self):
        """Test that relation task type is inferred."""
        record = CanonicalIERecord(
            id="rel-001",
            text="John works for Google",
            answer=Answer(
                relation=[
                    RelationAnnotation(
                        head="John",
                        head_type="PERSON",
                        relation="works_for",
                        tail="Google",
                        tail_type="ORGANIZATION",
                    )
                ]
            ),
        )
        active = record.get_active_task_types()
        assert "relation" in active
        assert "kv" not in active
        assert "entity" not in active

    def test_get_active_task_types_combined(self, sample_unified_record):
        """Test that all task types are inferred when present."""
        record = CanonicalIERecord.from_dict(sample_unified_record)
        active = record.get_active_task_types()
        assert len(active) == 3
        assert "kv" in active
        assert "entity" in active
        assert "relation" in active

    def test_get_active_task_types_empty(self):
        """Test that no task types are returned for empty answers."""
        record = CanonicalIERecord(
            id="empty-001",
            text="No information extracted",
        )
        active = record.get_active_task_types()
        assert len(active) == 0


class TestSerialization:
    """Test serialization and deserialization."""

    def test_serialization_roundtrip(self, sample_kv_record):
        """Test that serialize -> deserialize preserves data."""
        # Create from dict
        record1 = CanonicalIERecord.from_dict(sample_kv_record)

        # Serialize
        data = record1.to_canonical_dict()

        # Deserialize
        record2 = CanonicalIERecord.from_dict(data)

        # Compare
        assert record1.id == record2.id
        assert record1.text == record2.text
        assert record1.lang == record2.lang
        assert record1.task_types == record2.task_types
        assert record1.answer.kv == record2.answer.kv

    def test_canonical_dict_uses_schema_alias(self, sample_kv_record):
        """Test that to_canonical_dict uses 'schema' not 'schema_def'."""
        record = CanonicalIERecord.from_dict(sample_kv_record)
        data = record.to_canonical_dict()
        assert "schema" in data
        assert "schema_def" not in data
        assert data["schema"]["kv"] == ["name", "email", "phone"]

    def test_from_dict_accepts_schema_alias(self):
        """Test that from_dict accepts 'schema' as alias for 'schema_def'."""
        data = {
            "id": "test-001",
            "text": "Test",
            "schema": {"kv": ["key"], "entity": [], "relation": []},
        }
        record = CanonicalIERecord.from_dict(data)
        assert record.schema_def.kv == ["key"]

    def test_from_dict_converts_schema_to_schema_def(self):
        """Test that from_dict correctly converts schema to schema_def."""
        data = {
            "id": "test-001",
            "text": "Test text",
            "schema": {
                "kv": ["name"],
                "entity": ["PERSON"],
                "relation": ["works_for"],
            },
        }
        record = CanonicalIERecord.from_dict(data)
        assert isinstance(record.schema_def, SchemaDefinition)
        assert record.schema_def.kv == ["name"]
        assert record.schema_def.entity == ["PERSON"]
        assert record.schema_def.relation == ["works_for"]


class TestMissingFields:
    """Test required fields validation."""

    def test_missing_id_field(self):
        """Test that missing id field raises error."""
        with pytest.raises(ValueError):
            CanonicalIERecord(
                text="Some text",
            )

    def test_missing_text_field(self):
        """Test that missing text field raises error."""
        with pytest.raises(ValueError):
            CanonicalIERecord(
                id="test-001",
            )

    def test_defaults_for_optional_fields(self):
        """Test that optional fields have proper defaults."""
        record = CanonicalIERecord(
            id="test-001",
            text="Test text",
        )
        assert record.lang == "en"
        assert record.source == ""
        assert record.task_types == []
        assert isinstance(record.schema_def, SchemaDefinition)
        assert isinstance(record.answer, Answer)
        assert isinstance(record.meta, MetaInfo)


class TestEntityAnnotation:
    """Test EntityAnnotation model."""

    def test_entity_annotation_with_offsets(self):
        """Test creating entity with start/end offsets."""
        entity = EntityAnnotation(
            text="John Smith",
            type="PERSON",
            start=0,
            end=10,
        )
        assert entity.text == "John Smith"
        assert entity.type == "PERSON"
        assert entity.start == 0
        assert entity.end == 10

    def test_entity_annotation_without_offsets(self):
        """Test creating entity without offsets."""
        entity = EntityAnnotation(
            text="John Smith",
            type="PERSON",
        )
        assert entity.text == "John Smith"
        assert entity.type == "PERSON"
        assert entity.start is None
        assert entity.end is None

    def test_entity_annotation_model_dump(self):
        """Test that model_dump excludes None offsets."""
        entity = EntityAnnotation(
            text="John",
            type="PERSON",
        )
        data = entity.model_dump()
        assert "text" in data
        assert "type" in data
        assert "start" not in data
        assert "end" not in data


class TestRelationAnnotation:
    """Test RelationAnnotation model."""

    def test_relation_annotation_complete(self):
        """Test creating a complete relation annotation."""
        relation = RelationAnnotation(
            head="John",
            head_type="PERSON",
            relation="works_for",
            tail="Google",
            tail_type="ORGANIZATION",
        )
        assert relation.head == "John"
        assert relation.head_type == "PERSON"
        assert relation.relation == "works_for"
        assert relation.tail == "Google"
        assert relation.tail_type == "ORGANIZATION"


class TestSchemaDefinition:
    """Test SchemaDefinition model."""

    def test_schema_definition_defaults(self):
        """Test that SchemaDefinition has empty defaults."""
        schema = SchemaDefinition()
        assert schema.kv == []
        assert schema.entity == []
        assert schema.relation == []

    def test_schema_definition_with_values(self):
        """Test creating SchemaDefinition with values."""
        schema = SchemaDefinition(
            kv=["name", "email"],
            entity=["PERSON", "ORGANIZATION"],
            relation=["works_for"],
        )
        assert schema.kv == ["name", "email"]
        assert schema.entity == ["PERSON", "ORGANIZATION"]
        assert schema.relation == ["works_for"]


class TestMetaInfo:
    """Test MetaInfo model."""

    def test_meta_info_defaults(self):
        """Test MetaInfo default values."""
        meta = MetaInfo()
        assert meta.dataset == ""
        assert meta.license == ""
        assert meta.split == ""
        assert meta.notes == ""

    def test_meta_info_with_values(self):
        """Test MetaInfo with values."""
        meta = MetaInfo(
            dataset="instructie",
            license="MIT",
            split="train",
            notes="Test data",
        )
        assert meta.dataset == "instructie"
        assert meta.license == "MIT"
        assert meta.split == "train"
        assert meta.notes == "Test data"
