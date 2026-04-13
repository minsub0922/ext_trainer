# Canonical Information Extraction Schema

Complete specification of the unified schema used throughout IE-SFT-PoC for all information extraction tasks.

## Overview

The canonical schema is a single JSON format that supports three core information extraction task types:

1. **Key-Value (KV) Extraction**: Structured field-value pairs (e.g., product attributes)
2. **Entity Extraction**: Named entity recognition with type and span information
3. **Relation Extraction**: Relationships between entities with relation type

This unified format enables:
- Training a single model on multiple task types
- Composing datasets with heterogeneous tasks
- Consistent validation and evaluation across tasks
- Flexible multi-task learning approaches

## Full Schema Definition

### Record Structure (Root Level)

```json
{
  "id": "string",                    // Unique record identifier (required)
  "text": "string",                  // Input text to extract from (required)
  "lang": "string",                  // Language code, e.g., "en" (optional, default: "en")
  "source": "string",                // Data source identifier (optional)
  "task_types": ["string"],          // Active task types: ["kv", "entity", "relation"] (required)
  "schema": {                        // Schema definitions (required)
    "kv": ["string"],               // KV field names
    "entity": ["string"],           // Entity type labels
    "relation": ["string"]          // Relation type labels
  },
  "answer": {                        // Extraction results (required)
    "kv": { "string": "string" },   // Key-value pairs
    "entity": [...],                // Entity annotations
    "relation": [...]               // Relation annotations
  },
  "meta": {                          // Metadata (optional)
    "dataset": "string",            // Dataset source
    "license": "string",            // License identifier
    "split": "string",              // Data split (train/dev/test)
    "notes": "string"               // Additional notes
  }
}
```

### Field Descriptions

#### Record-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for this record. Used in validation reports and deduplication. Format: `{dataset}_{split}_{index}` recommended. |
| `text` | string | Yes | Raw input text from which to extract information. Length: 1-10,000 characters. |
| `lang` | string | No | ISO 639-1 language code (en, zh, ja, etc.). Default: "en". Used for multilingual datasets. |
| `source` | string | No | Original data source identifier. Helps track data provenance. Examples: "news_corpus", "wiki", "instructie". |
| `task_types` | array | Yes | List of active extraction task types. Must be subset of: ["kv", "entity", "relation"]. At least one required. |
| `schema` | object | Yes | Task-specific schema definitions (see Schema Object below). |
| `answer` | object | Yes | Extraction results organized by task type (see Answer Object below). |
| `meta` | object | No | Metadata about the record for tracking and analysis. |

#### Schema Object

Defines the schema/ontology for each active task type:

```json
{
  "schema": {
    "kv": ["field1", "field2", "price"],           // Field names for KV extraction
    "entity": ["PERSON", "ORG", "LOCATION"],       // Entity type labels
    "relation": ["founded_by", "located_in"]       // Relation type labels
  }
}
```

- **kv** (array of strings): Field names to extract in KV task
  - Examples: "price", "manufacturer", "model_number"
  - Order doesn't matter; used for validation

- **entity** (array of strings): Valid entity type labels
  - Examples: "PERSON", "ORG", "LOC", "DATE", "PRODUCT"
  - Entities in answer must match one of these types
  - Case-sensitive

- **relation** (array of strings): Valid relation type labels
  - Examples: "founded_by", "works_for", "located_in"
  - Relations in answer must match one of these types
  - Case-sensitive

#### Answer Object

Contains extraction results organized by task type:

```json
{
  "answer": {
    "kv": {
      "price": "$499.99",
      "brand": "Sony",
      "model": "WH-1000XM4"
    },
    "entity": [
      {
        "text": "Sony",
        "type": "ORG",
        "start": 10,
        "end": 14
      }
    ],
    "relation": [
      {
        "head": "Sony",
        "head_type": "ORG",
        "relation": "manufactures",
        "tail": "WH-1000XM4",
        "tail_type": "PRODUCT"
      }
    ]
  }
}
```

##### KV Extraction (answer.kv)

- Type: Object (dictionary)
- Keys: Field names from schema.kv
- Values: Extracted values as strings
- Empty KV: `{}`

Example:
```json
{
  "kv": {
    "product_name": "iPhone 15 Pro",
    "price": "$999",
    "brand": "Apple",
    "color": "Space Black"
  }
}
```

Validation Rules:
- All keys must be in schema.kv
- Values are strings (no nested objects)
- Empty values allowed (""), but null not allowed
- No extra keys beyond schema definition

##### Entity Extraction (answer.entity)

- Type: Array of entity objects
- Each entity has: text, type, start (char offset), end (char offset)

Entity Object Structure:
```json
{
  "text": "string",      // Entity text (must match text[start:end])
  "type": "string",     // Entity type (must be in schema.entity)
  "start": "integer",   // Character offset (inclusive)
  "end": "integer"      // Character offset (exclusive)
}
```

Example:
```json
{
  "entity": [
    {
      "text": "Apple Inc.",
      "type": "ORG",
      "start": 0,
      "end": 10
    },
    {
      "text": "Steve Jobs",
      "type": "PERSON",
      "start": 25,
      "end": 35
    },
    {
      "text": "California",
      "type": "LOC",
      "start": 39,
      "end": 49
    }
  ]
}
```

Validation Rules:
- `text` must exactly match `record.text[start:end]`
- `type` must be in `schema.entity`
- `start` must be < `end`
- `end` must be <= `len(record.text)`
- Overlapping spans allowed (for multi-type annotations)
- Empty entity list `[]` allowed

##### Relation Extraction (answer.relation)

- Type: Array of relation objects
- Each relation connects head and tail entities

Relation Object Structure:
```json
{
  "head": "string",           // Head entity text
  "head_type": "string",      // Head entity type
  "relation": "string",       // Relation type
  "tail": "string",           // Tail entity text
  "tail_type": "string"       // Tail entity type
}
```

Example:
```json
{
  "relation": [
    {
      "head": "Apple Inc.",
      "head_type": "ORG",
      "relation": "founded_by",
      "tail": "Steve Jobs",
      "tail_type": "PERSON"
    },
    {
      "head": "Apple Inc.",
      "head_type": "ORG",
      "relation": "located_in",
      "tail": "California",
      "tail_type": "LOC"
    }
  ]
}
```

Validation Rules:
- `head`, `head_type`, `tail`, `tail_type`, `relation` all required (non-empty)
- `head_type` must be in `schema.entity`
- `tail_type` must be in `schema.entity`
- `relation` must be in `schema.relation`
- Entity text doesn't need to match exactly (can be normalized)
- Circular relations allowed (A → B and B → A)
- Self-relations allowed (A → A)
- Empty relation list `[]` allowed

#### Metadata Object

Optional metadata for tracking and analysis:

```json
{
  "meta": {
    "dataset": "instructie",         // Dataset source
    "license": "cc-by-4.0",          // License identifier
    "split": "train",                // Data split
    "source_id": "instr_001",        // Original ID from source dataset
    "notes": "Product review excerpt", // Additional notes
    "annotators": ["human1"]         // Who annotated (optional)
  }
}
```

Common fields:
- **dataset**: Name of dataset (instructie, internal_kv, gollie, etc.)
- **license**: License (cc-by-4.0, mit, custom, etc.)
- **split**: Data split (train, dev, test)
- **source_id**: Original ID in source dataset
- **notes**: Free-form notes about the record

## Task Types Explained

### Key-Value Extraction

Extracting structured field-value pairs from text.

**Use Cases**:
- Product specifications (price, brand, model)
- Document metadata (author, date, title)
- Contact information (name, email, phone)
- Transaction details (amount, date, account)

**Example**:
```json
{
  "id": "product_001",
  "text": "Sony WH-1000XM4 wireless headphones, priced at $348.00, available in black and silver. Features active noise cancellation.",
  "task_types": ["kv"],
  "schema": {
    "kv": ["product_name", "price", "brand", "colors", "features"],
    "entity": [],
    "relation": []
  },
  "answer": {
    "kv": {
      "product_name": "WH-1000XM4",
      "price": "$348.00",
      "brand": "Sony",
      "colors": "black, silver",
      "features": "active noise cancellation"
    },
    "entity": [],
    "relation": []
  }
}
```

**Schema Considerations**:
- Fields should be exhaustive but not overwhelming
- Order in schema matches extraction order
- Consider field normalization (e.g., "price" vs "price_usd")

### Entity Extraction

Identifying and categorizing named entities with span information.

**Use Cases**:
- Named entity recognition (person, organization, location)
- Product identification in text
- Event detection
- Temporal expression extraction

**Entity Types** (common):
- `PERSON`: Individual human
- `ORG`: Organization, company
- `LOC`: Geographic location
- `DATE`: Temporal expression
- `PRODUCT`: Product or service name
- `GPE`: Geopolitical entity (country, city)
- `EVENT`: Named event

**Example**:
```json
{
  "id": "entity_001",
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976.",
  "task_types": ["entity"],
  "schema": {
    "kv": [],
    "entity": ["ORG", "PERSON", "LOC", "DATE"],
    "relation": []
  },
  "answer": {
    "kv": {},
    "entity": [
      {
        "text": "Apple Inc.",
        "type": "ORG",
        "start": 0,
        "end": 9
      },
      {
        "text": "Steve Jobs",
        "type": "PERSON",
        "start": 27,
        "end": 37
      },
      {
        "text": "Cupertino, California",
        "type": "LOC",
        "start": 41,
        "end": 62
      },
      {
        "text": "April 1, 1976",
        "type": "DATE",
        "start": 66,
        "end": 79
      }
    ],
    "relation": []
  }
}
```

**Span Accuracy**:
- Spans must be exact character offsets
- Include punctuation if part of entity
- Exclude leading/trailing whitespace
- Use string slicing: `text[start:end]`

### Relation Extraction

Identifying relationships between entities with relation type.

**Use Cases**:
- Organizational relationships (works_for, founded_by)
- Geographic relationships (located_in)
- Product relationships (manufactures, produces)
- Event participation (attends, organizes)

**Common Relations**:
- `founded_by`: Organization founded by person
- `located_in`: Entity located in place
- `works_for`: Person works for organization
- `manages`: Person manages other person/team
- `manufactures`: Organization manufactures product

**Example**:
```json
{
  "id": "relation_001",
  "text": "Elon Musk founded Tesla and SpaceX. Tesla is headquartered in Austin, Texas.",
  "task_types": ["entity", "relation"],
  "schema": {
    "kv": [],
    "entity": ["ORG", "PERSON", "LOC"],
    "relation": ["founded_by", "headquartered_in"]
  },
  "answer": {
    "kv": {},
    "entity": [
      {"text": "Elon Musk", "type": "PERSON", "start": 0, "end": 9},
      {"text": "Tesla", "type": "ORG", "start": 18, "end": 23},
      {"text": "SpaceX", "type": "ORG", "start": 28, "end": 34},
      {"text": "Austin, Texas", "type": "LOC", "start": 63, "end": 76}
    ],
    "relation": [
      {
        "head": "Tesla",
        "head_type": "ORG",
        "relation": "founded_by",
        "tail": "Elon Musk",
        "tail_type": "PERSON"
      },
      {
        "head": "SpaceX",
        "head_type": "ORG",
        "relation": "founded_by",
        "tail": "Elon Musk",
        "tail_type": "PERSON"
      },
      {
        "head": "Tesla",
        "head_type": "ORG",
        "relation": "headquartered_in",
        "tail": "Austin, Texas",
        "tail_type": "LOC"
      }
    ]
  }
}
```

**Relation Design**:
- Relations are typically directed (A → B)
- Include inverse if both directions are valid
- Relation types should be semantically clear
- Consider both explicit and implicit relations

### Multi-Task Records

Records can contain multiple task types simultaneously:

**Example**:
```json
{
  "id": "multi_task_001",
  "text": "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976. The company is now led by Tim Cook.",
  "task_types": ["kv", "entity", "relation"],
  "schema": {
    "kv": ["company_name", "founding_year", "current_leader"],
    "entity": ["ORG", "PERSON", "LOC", "DATE"],
    "relation": ["founded_by", "located_in", "led_by"]
  },
  "answer": {
    "kv": {
      "company_name": "Apple Inc.",
      "founding_year": "1976",
      "current_leader": "Tim Cook"
    },
    "entity": [
      {"text": "Apple Inc.", "type": "ORG", "start": 0, "end": 10},
      {"text": "Steve Jobs", "type": "PERSON", "start": 28, "end": 38},
      // ... more entities ...
    ],
    "relation": [
      {
        "head": "Apple Inc.",
        "head_type": "ORG",
        "relation": "founded_by",
        "tail": "Steve Jobs",
        "tail_type": "PERSON"
      },
      // ... more relations ...
    ]
  }
}
```

## Validation Rules

### Schema Consistency

1. **Task Type Declaration**:
   - At least one task type required in `task_types`
   - `task_types` must be subset of ["kv", "entity", "relation"]

2. **Schema Definition**:
   - schema.kv must be non-empty array if "kv" in task_types
   - schema.entity must be non-empty array if "entity"/"relation" in task_types
   - schema.relation must be non-empty array if "relation" in task_types
   - All array elements must be non-empty strings

3. **Answer-Schema Match**:
   - If "kv" in task_types, all keys in answer.kv must be in schema.kv
   - If "entity"/"relation" in task_types, all entity types must be in schema.entity
   - If "relation" in task_types, all relation types must be in schema.relation

### Data Integrity

1. **Required Fields**:
   - `id`, `text`, `task_types`, `schema`, `answer` all required
   - All fields must be non-null

2. **Text Constraints**:
   - Length: 1-10,000 characters
   - Must be valid UTF-8
   - No control characters (except newline)

3. **Entity Span Validation**:
   - `start` and `end` must be valid character offsets
   - 0 <= start < end <= len(text)
   - `text[start:end]` must equal entity.text exactly

4. **Relation Validation**:
   - All required fields (head, head_type, tail, tail_type, relation) non-empty
   - head_type in schema.entity
   - tail_type in schema.entity
   - relation in schema.relation

### Quality Checks

1. **Completeness**:
   - Record shouldn't have empty answer for active task types
   - At least one annotation in active tasks

2. **Consistency**:
   - Entity types in relations must exist in entity list (or be valid schema types)
   - No duplicate entities in entity list (same text, type, span)
   - No duplicate relations

3. **Span Accuracy**:
   - Entity text must match original text exactly
   - No overlapping entity spans within same type (usually)
   - Valid UTF-8 character boundaries

## Schema-Conditioned Extraction

The canonical schema enables **schema-conditioned extraction**: models receive task schema as part of the input prompt.

**Concept**:
Instead of learning fixed entity/relation types, models learn to:
1. Accept dynamic schema definitions in the prompt
2. Extract entities/relations matching the provided schema
3. Generalize to unseen entity/relation types

**Example Prompt**:
```
Extract entities and relations from the text according to the schema.

Text: "Apple Inc. was founded by Steve Jobs in California."

Schema:
- Entity types: ORG, PERSON, LOC
- Relation types: founded_by, located_in

Output:
{
  "entity": [...],
  "relation": [...]
}
```

**Benefits**:
- Single model handles multiple ontologies
- Generalization to new entity/relation types
- Flexible schema at inference time
- Enables zero-shot extraction with custom schemas

## Adding New Task Types

To extend the canonical schema with new task types:

1. **Define Task Structure**:
   - Determine what data this task extracts
   - Design JSON representation
   - Document schema and validation rules

2. **Update Record Schema**:
   ```json
   {
     "schema": {
       "kv": [...],
       "entity": [...],
       "relation": [...],
       "new_task": [...]      // Add new task type
     },
     "answer": {
       "kv": {},
       "entity": [],
       "relation": [],
       "new_task": [...]      // Add corresponding answer field
     }
   }
   ```

3. **Update Validation**:
   - Add validation logic in `src/datasets/unified/validator.py`
   - Update constants in `src/common/constants.py`

4. **Update Documentation**:
   - Document new task type in this file
   - Update examples
   - Update validation rules

5. **Test**:
   - Add test records with new task type
   - Validate with updated validator
   - Update statistics generation

## Examples

Complete example records are provided in `examples/canonical_samples/`:
- `kv_sample.json`: Key-value extraction examples
- `entity_relation_sample.json`: Entity and relation extraction
- `unified_sample.json`: Multi-task examples

See those files for ready-to-use examples.

## Design Rationale

### Single Unified Format vs. Task-Specific Formats

**Chosen**: Single unified format

**Rationale**:
- Simplifies pipeline (one validator, one format)
- Enables multi-task training on single dataset
- Easier to compose datasets
- Flexible schema adaptation at inference time

**Alternative**: Task-specific formats
- Would require separate normalization per task
- Harder to mix tasks in training
- More complex validation

### Relation Without Entity List Requirement

**Choice**: Relations reference entities by text, not by ID

**Rationale**:
- More human-readable
- Doesn't require entity deduplication
- Handles repeated entity mentions naturally
- Matches how humans describe relations

**Trade-off**: Requires entity text matching rules

### Character Offsets vs. Token Offsets

**Chosen**: Character offsets

**Rationale**:
- Tokenization is model-dependent
- Character offsets more reproducible
- Can reconstruct tokens from characters
- Matches standard NER conventions

## Performance Tips

1. **Validation Speed**:
   - Use `validate_dataset(..., sample=1000)` for quick checks
   - Parallel validation for large files

2. **Storage Efficiency**:
   - Character offsets are compact (integer)
   - Canonical format similar size to originals
   - Compression possible (gzip): ~70% reduction

3. **Processing**:
   - Batch processing for normalization
   - Streaming readers for large files
   - Deduplication at merge stage

---

**Last Updated**: April 2026  
**Schema Version**: 1.0
