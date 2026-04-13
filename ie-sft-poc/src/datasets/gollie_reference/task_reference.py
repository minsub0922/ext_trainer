"""GoLLIE-style task reference definitions for schema-conditioned IE.

This module defines task references showing how different IE tasks (KV extraction,
entity recognition, relation extraction) map to schema-conditioned extraction patterns.

NOTE: This is reference/inspiration from the GoLLIE approach, NOT a dependency on GoLLIE.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskDefinition:
    """Reference definition for an information extraction task.

    Attributes:
        task_name: Name of the task (e.g., 'kv', 'entity', 'relation', 'unified')
        description: Description of what the task extracts
        schema_fields: List of fields/types in the schema
        input_format: Description of input format
        output_format: Description of expected output format
        examples: List of example (input_text, expected_output) tuples
    """

    task_name: str
    description: str
    schema_fields: list[str]
    input_format: str
    output_format: str
    examples: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def get_kv_task_definition() -> TaskDefinition:
    """Get task reference for key-value extraction.

    Returns:
        TaskDefinition for KV extraction showing how to map slot-filling
        to structured extraction.
    """
    return TaskDefinition(
        task_name="kv",
        description="Extract key-value pairs from text. Keys are predefined slots, "
        "and values are text spans that fill those slots.",
        schema_fields=["name", "email", "phone", "address", "company"],
        input_format="Unstructured text",
        output_format="Dictionary with keys mapping to extracted values",
        examples=[
            (
                "John Smith, john@example.com, works at Google in Mountain View.",
                {
                    "name": "John Smith",
                    "email": "john@example.com",
                    "phone": None,
                    "address": "Mountain View",
                    "company": "Google",
                },
            ),
            (
                "Contact: Alice Johnson (alice.j@org.com, +1-555-1234)",
                {
                    "name": "Alice Johnson",
                    "email": "alice.j@org.com",
                    "phone": "+1-555-1234",
                    "address": None,
                    "company": None,
                },
            ),
        ],
    )


def get_entity_task_definition() -> TaskDefinition:
    """Get task reference for named entity recognition.

    Returns:
        TaskDefinition for NER showing how entity types are predefined
        and entities are extracted as structured annotations.
    """
    return TaskDefinition(
        task_name="entity",
        description="Extract named entities from text. Entity types are predefined, "
        "and each entity is annotated with its type and span.",
        schema_fields=["PERSON", "ORGANIZATION", "LOCATION", "DATE", "PRODUCT"],
        input_format="Unstructured text",
        output_format="List of entities with text, type, and character offsets",
        examples=[
            (
                "Apple Inc. was founded by Steve Jobs in Cupertino on April 1, 1976.",
                [
                    {"text": "Apple Inc.", "type": "ORGANIZATION", "start": 0, "end": 10},
                    {"text": "Steve Jobs", "type": "PERSON", "start": 26, "end": 37},
                    {"text": "Cupertino", "type": "LOCATION", "start": 41, "end": 51},
                    {"text": "April 1, 1976", "type": "DATE", "start": 55, "end": 69},
                ],
            ),
            (
                "Samsung released the Galaxy S21 in South Korea.",
                [
                    {"text": "Samsung", "type": "ORGANIZATION", "start": 0, "end": 7},
                    {"text": "Galaxy S21", "type": "PRODUCT", "start": 20, "end": 31},
                    {"text": "South Korea", "type": "LOCATION", "start": 35, "end": 47},
                ],
            ),
        ],
    )


def get_relation_task_definition() -> TaskDefinition:
    """Get task reference for relation extraction.

    Returns:
        TaskDefinition for RE showing how entities are paired with
        relation types to create structured relation annotations.
    """
    return TaskDefinition(
        task_name="relation",
        description="Extract relations between entities in text. Relations connect "
        "two entities with a predefined relation type.",
        schema_fields=["works_for", "located_in", "founded", "owner_of", "parent_of"],
        input_format="Unstructured text (entities typically identified first)",
        output_format="List of relations with head entity, relation type, tail entity",
        examples=[
            (
                "John Smith works for Google. Google is located in Mountain View.",
                [
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
            ),
            (
                "Elon Musk founded Tesla in 2003.",
                [
                    {
                        "head": "Elon Musk",
                        "head_type": "PERSON",
                        "relation": "founded",
                        "tail": "Tesla",
                        "tail_type": "ORGANIZATION",
                    },
                ],
            ),
        ],
    )


def get_unified_task_definition() -> TaskDefinition:
    """Get task reference for unified extraction combining multiple task types.

    Returns:
        TaskDefinition showing how KV, entity, and relation extraction
        can be unified in a single structured output.
    """
    return TaskDefinition(
        task_name="unified",
        description="Unified information extraction combining KV extraction, "
        "entity recognition, and relation extraction in a single output.",
        schema_fields=[
            "kv: [name, email, phone]",
            "entity: [PERSON, ORGANIZATION, LOCATION]",
            "relation: [works_for, located_in]",
        ],
        input_format="Unstructured text",
        output_format="Single structure with keys for 'kv', 'entity', and 'relation'",
        examples=[
            (
                "John Smith (john@company.com) works for Google in Mountain View.",
                {
                    "kv": {
                        "name": "John Smith",
                        "email": "john@company.com",
                        "phone": None,
                    },
                    "entity": [
                        {
                            "text": "John Smith",
                            "type": "PERSON",
                            "start": 0,
                            "end": 10,
                        },
                        {
                            "text": "Google",
                            "type": "ORGANIZATION",
                            "start": 33,
                            "end": 39,
                        },
                        {
                            "text": "Mountain View",
                            "type": "LOCATION",
                            "start": 43,
                            "end": 57,
                        },
                    ],
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
            ),
        ],
    )


def get_all_task_definitions() -> dict[str, TaskDefinition]:
    """Get all task definitions.

    Returns:
        Dictionary mapping task names to their definitions
    """
    return {
        "kv": get_kv_task_definition(),
        "entity": get_entity_task_definition(),
        "relation": get_relation_task_definition(),
        "unified": get_unified_task_definition(),
    }
