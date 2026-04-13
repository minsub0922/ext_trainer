"""Schema-conditioned prompting patterns inspired by GoLLIE.

This module generates prompts for instruction-tuning that include schema definitions
and structured output format specifications. The patterns guide LLMs to perform
structured extraction in a consistent, predictable manner.

NOTE: This is inspired by GoLLIE's approach to schema-conditioned IE, but implements
patterns specifically for this project's task types and canonical schema format.
"""

from src.common.schema import SchemaDefinition


# Format instruction templates for each task type
FORMAT_INSTRUCTIONS = {
    "kv": """Output format: Return a JSON object with the following structure:
{
  "kv": {
    "field_name_1": "extracted_value_or_null",
    "field_name_2": "extracted_value_or_null",
    ...
  }
}

Rules:
- Only include keys that are in the schema
- Return null for fields where no value is found in the text
- Values should be text spans directly from the input
- Do not paraphrase or infer values not explicitly stated""",
    "entity": """Output format: Return a JSON object with the following structure:
{
  "entity": [
    {
      "text": "extracted_entity_text",
      "type": "ENTITY_TYPE",
      "start": character_offset_start,
      "end": character_offset_end
    },
    ...
  ]
}

Rules:
- Only extract entities matching the defined entity types
- Provide character offsets for each entity (0-indexed, end is exclusive)
- Entities should be exact text spans from the input
- If no entities are found, return an empty list
- Entities should not overlap""",
    "relation": """Output format: Return a JSON object with the following structure:
{
  "relation": [
    {
      "head": "head_entity_text",
      "head_type": "HEAD_ENTITY_TYPE",
      "relation": "RELATION_TYPE",
      "tail": "tail_entity_text",
      "tail_type": "TAIL_ENTITY_TYPE"
    },
    ...
  ]
}

Rules:
- Only extract relations where both head and tail entities exist in the text
- Relation type must be from the defined relation schema
- Head and tail entity types must match their respective entity types
- If no relations are found, return an empty list
- Each relation must connect exactly two entities""",
    "unified": """Output format: Return a JSON object with the following structure:
{
  "kv": {
    "field_name": "value_or_null",
    ...
  },
  "entity": [
    {
      "text": "entity_text",
      "type": "ENTITY_TYPE",
      "start": offset_start,
      "end": offset_end
    },
    ...
  ],
  "relation": [
    {
      "head": "head_text",
      "head_type": "HEAD_TYPE",
      "relation": "RELATION_TYPE",
      "tail": "tail_text",
      "tail_type": "TAIL_TYPE"
    },
    ...
  ]
}

Rules:
- Include only sections with applicable results (kv, entity, relation)
- Follow all rules from individual KV, Entity, and Relation formats
- Entities should be complete and consistent across KV, Entity, and Relation sections
- Empty lists should be returned for entity/relation if none are found""",
}


def build_schema_prompt(task_type: str, schema_def: SchemaDefinition) -> str:
    """Build a schema description prompt for the given task type and schema.

    Args:
        task_type: Type of task ('kv', 'entity', 'relation', 'unified')
        schema_def: Schema definition containing field/type lists

    Returns:
        Schema description prompt text
    """
    if task_type == "kv":
        fields_list = "\n".join(f"  - {field}" for field in schema_def.kv)
        return f"""Task: Key-Value Extraction
Description: Extract key-value pairs from the input text.

Schema (keys to extract):
{fields_list}"""

    elif task_type == "entity":
        types_list = "\n".join(f"  - {etype}" for etype in schema_def.entity)
        return f"""Task: Named Entity Recognition
Description: Extract named entities from the input text.

Entity Types:
{types_list}"""

    elif task_type == "relation":
        rel_types_list = "\n".join(f"  - {rtype}" for rtype in schema_def.relation)
        ent_types_list = "\n".join(f"  - {etype}" for etype in schema_def.entity)
        return f"""Task: Relation Extraction
Description: Extract relations between entities in the input text.

Relation Types:
{rel_types_list}

Entity Types (for head and tail):
{ent_types_list}"""

    elif task_type == "unified":
        kv_fields = "\n".join(f"  - {field}" for field in schema_def.kv)
        ent_types = "\n".join(f"  - {etype}" for etype in schema_def.entity)
        rel_types = "\n".join(f"  - {rtype}" for rtype in schema_def.relation)
        return f"""Task: Unified Information Extraction
Description: Extract key-value pairs, entities, and relations from the input text.

KV Fields:
{kv_fields if kv_fields else "  (none)"}

Entity Types:
{ent_types if ent_types else "  (none)"}

Relation Types:
{rel_types if rel_types else "  (none)"}"""

    else:
        return f"Unknown task type: {task_type}"


def build_kv_extraction_prompt(text: str, kv_fields: list[str]) -> str:
    """Build a complete prompt for KV extraction.

    Args:
        text: Input text to extract from
        kv_fields: List of field names to extract

    Returns:
        Complete instruction prompt for KV extraction
    """
    fields_desc = "\n".join(f"  - {field}" for field in kv_fields)

    return f"""You are an expert information extraction system.

Task: Key-Value Extraction
Extract the following fields from the text:
{fields_desc}

{FORMAT_INSTRUCTIONS["kv"]}

Input text:
{text}

Output:"""


def build_entity_extraction_prompt(text: str, entity_types: list[str]) -> str:
    """Build a complete prompt for entity extraction.

    Args:
        text: Input text to extract from
        entity_types: List of entity types to recognize

    Returns:
        Complete instruction prompt for entity extraction
    """
    types_desc = "\n".join(f"  - {etype}" for etype in entity_types)

    return f"""You are an expert information extraction system.

Task: Named Entity Recognition
Extract entities of the following types from the text:
{types_desc}

{FORMAT_INSTRUCTIONS["entity"]}

Input text:
{text}

Output:"""


def build_relation_extraction_prompt(
    text: str,
    relation_types: list[str],
    entity_types: list[str],
) -> str:
    """Build a complete prompt for relation extraction.

    Args:
        text: Input text to extract from
        relation_types: List of relation types
        entity_types: List of entity types (for head/tail validation)

    Returns:
        Complete instruction prompt for relation extraction
    """
    rel_desc = "\n".join(f"  - {rtype}" for rtype in relation_types)
    ent_desc = "\n".join(f"  - {etype}" for etype in entity_types)

    return f"""You are an expert information extraction system.

Task: Relation Extraction
Extract relations between entities in the text.

Relation Types:
{rel_desc}

Valid Entity Types for head and tail:
{ent_desc}

{FORMAT_INSTRUCTIONS["relation"]}

Input text:
{text}

Output:"""


def build_unified_extraction_prompt(text: str, schema_def: SchemaDefinition) -> str:
    """Build a complete prompt for unified extraction.

    Args:
        text: Input text to extract from
        schema_def: Schema definition with all fields/types

    Returns:
        Complete instruction prompt for unified extraction
    """
    kv_desc = ""
    if schema_def.kv:
        kv_fields = "\n".join(f"  - {field}" for field in schema_def.kv)
        kv_desc = f"\nKV Fields:\n{kv_fields}"

    ent_desc = ""
    if schema_def.entity:
        ent_types = "\n".join(f"  - {etype}" for etype in schema_def.entity)
        ent_desc = f"\nEntity Types:\n{ent_types}"

    rel_desc = ""
    if schema_def.relation:
        rel_types = "\n".join(f"  - {rtype}" for rtype in schema_def.relation)
        rel_desc = f"\nRelation Types:\n{rel_types}"

    schema_section = f"""Schema Definition:{kv_desc}{ent_desc}{rel_desc}"""

    return f"""You are an expert information extraction system.

Task: Unified Information Extraction
Extract key-value pairs, named entities, and relations from the input text.

{schema_section}

{FORMAT_INSTRUCTIONS["unified"]}

Input text:
{text}

Output:"""


def build_schema_prompt_from_schema(
    task_type: str,
    schema_def: SchemaDefinition,
    input_text: str,
) -> str:
    """Build a complete task prompt using schema definition and input text.

    This is a convenience function that combines schema building and prompt generation.

    Args:
        task_type: Type of task ('kv', 'entity', 'relation', 'unified')
        schema_def: Schema definition
        input_text: Input text to extract from

    Returns:
        Complete instruction prompt
    """
    if task_type == "kv":
        return build_kv_extraction_prompt(input_text, schema_def.kv)
    elif task_type == "entity":
        return build_entity_extraction_prompt(input_text, schema_def.entity)
    elif task_type == "relation":
        return build_relation_extraction_prompt(
            input_text,
            schema_def.relation,
            schema_def.entity,
        )
    elif task_type == "unified":
        return build_unified_extraction_prompt(input_text, schema_def)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
