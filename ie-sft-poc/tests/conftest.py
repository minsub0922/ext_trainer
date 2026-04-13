"""Shared pytest fixtures for IE SFT PoC tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_kv_record():
    """Create a sample key-value extraction record in canonical format.

    Returns:
        Dictionary representing a valid canonical KV record.
    """
    return {
        "id": "kv-001",
        "text": "Contact: John Smith, Email: john@example.com, Phone: 555-1234",
        "lang": "en",
        "source": "test",
        "task_types": ["kv"],
        "schema": {
            "kv": ["name", "email", "phone"],
            "entity": [],
            "relation": [],
        },
        "answer": {
            "kv": {
                "name": "John Smith",
                "email": "john@example.com",
                "phone": "555-1234",
            },
            "entity": [],
            "relation": [],
        },
        "meta": {
            "dataset": "test",
            "license": "MIT",
            "split": "train",
            "notes": "Test KV record",
        },
    }


@pytest.fixture
def sample_entity_record():
    """Create a sample named entity recognition record in canonical format.

    Returns:
        Dictionary representing a valid canonical entity record.
    """
    return {
        "id": "entity-001",
        "text": "Apple Inc. is headquartered in Cupertino, California.",
        "lang": "en",
        "source": "test",
        "task_types": ["entity"],
        "schema": {
            "kv": [],
            "entity": ["ORGANIZATION", "LOCATION"],
            "relation": [],
        },
        "answer": {
            "kv": {},
            "entity": [
                {
                    "text": "Apple Inc.",
                    "type": "ORGANIZATION",
                    "start": 0,
                    "end": 9,
                },
                {
                    "text": "Cupertino",
                    "type": "LOCATION",
                    "start": 29,
                    "end": 38,
                },
                {
                    "text": "California",
                    "type": "LOCATION",
                    "start": 40,
                    "end": 50,
                },
            ],
            "relation": [],
        },
        "meta": {
            "dataset": "test",
            "license": "MIT",
            "split": "train",
            "notes": "Test entity record",
        },
    }


@pytest.fixture
def sample_relation_record():
    """Create a sample relation extraction record in canonical format.

    Returns:
        Dictionary representing a valid canonical relation record.
    """
    return {
        "id": "relation-001",
        "text": "John Smith works at Google in Mountain View.",
        "lang": "en",
        "source": "test",
        "task_types": ["relation"],
        "schema": {
            "kv": [],
            "entity": ["PERSON", "ORGANIZATION", "LOCATION"],
            "relation": ["works_for", "located_in"],
        },
        "answer": {
            "kv": {},
            "entity": [],
            "relation": [
                {
                    "head": "John Smith",
                    "head_type": "PERSON",
                    "relation": "works_for",
                    "tail": "Google",
                    "tail_type": "ORGANIZATION",
                },
                {
                    "head": "Google",
                    "head_type": "ORGANIZATION",
                    "relation": "located_in",
                    "tail": "Mountain View",
                    "tail_type": "LOCATION",
                },
            ],
        },
        "meta": {
            "dataset": "test",
            "license": "MIT",
            "split": "train",
            "notes": "Test relation record",
        },
    }


@pytest.fixture
def sample_unified_record():
    """Create a sample record with all three task types.

    Returns:
        Dictionary representing a unified canonical record with KV, entity, and relation.
    """
    return {
        "id": "unified-001",
        "text": "CEO Tim Cook announced Apple's Q1 revenue of $100 billion.",
        "lang": "en",
        "source": "test",
        "task_types": ["kv", "entity", "relation"],
        "schema": {
            "kv": ["position", "company", "revenue"],
            "entity": ["PERSON", "ORGANIZATION", "MONEY"],
            "relation": ["works_for", "announced"],
        },
        "answer": {
            "kv": {
                "position": "CEO",
                "company": "Apple",
                "revenue": "$100 billion",
            },
            "entity": [
                {
                    "text": "Tim Cook",
                    "type": "PERSON",
                    "start": 4,
                    "end": 12,
                },
                {
                    "text": "Apple",
                    "type": "ORGANIZATION",
                    "start": 27,
                    "end": 32,
                },
                {
                    "text": "$100 billion",
                    "type": "MONEY",
                    "start": 45,
                    "end": 57,
                },
            ],
            "relation": [
                {
                    "head": "Tim Cook",
                    "head_type": "PERSON",
                    "relation": "works_for",
                    "tail": "Apple",
                    "tail_type": "ORGANIZATION",
                },
            ],
        },
        "meta": {
            "dataset": "test",
            "license": "MIT",
            "split": "train",
            "notes": "Test unified record",
        },
    }


@pytest.fixture
def sample_instructie_raw():
    """Create a sample InstructIE record in raw format (before parsing).

    Returns:
        Dictionary representing a raw InstructIE record.
    """
    return {
        "id": "instructie-001",
        "cate": "news",
        "text": "Steve Jobs founded Apple Computer Company in 1976.",
        "relation": [
            {
                "head": "Steve Jobs",
                "head_type": "PERSON",
                "relation": "founded",
                "tail": "Apple Computer Company",
                "tail_type": "ORGANIZATION",
            },
            {
                "head": "Apple Computer Company",
                "head_type": "ORGANIZATION",
                "relation": "founded_in",
                "tail": "1976",
                "tail_type": "DATE",
            },
        ],
    }


@pytest.fixture
def tmp_jsonl_file(tmp_path):
    """Factory fixture for creating temporary JSONL files with records.

    Args:
        tmp_path: pytest's built-in temporary directory fixture

    Returns:
        Function that takes a list of records and returns the path to a JSONL file.
    """
    def _write_jsonl(records, filename="test.jsonl"):
        """Write records to a JSONL file in the temporary directory.

        Args:
            records: List of dictionaries to write
            filename: Name of the JSONL file (default: test.jsonl)

        Returns:
            Path to the created JSONL file
        """
        file_path = tmp_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return file_path

    return _write_jsonl
