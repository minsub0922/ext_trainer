"""GoLLIE-inspired reference module for schema-conditioned IE patterns.

This module provides reference patterns and utilities inspired by GoLLIE (Generalized
Open-ended Language Learning for Information Extraction) for structuring information
extraction tasks. It is NOT a full GoLLIE integration but rather a reference implementation
showing how GoLLIE-style schema-conditioned prompting can be applied to different IE task types.

The module demonstrates:
- Task reference definitions (task names, descriptions, schemas, examples)
- Schema-conditioned prompt generation for structured extraction
- Support for KV extraction, entity recognition, and relation extraction
- Unified extraction combining multiple task types

References:
- GoLLIE: Xie et al., "Unified Structure Generation for Universal Information Extraction"
  (This module is inspired by but not dependent on the original GoLLIE codebase)
"""

__version__ = "0.1.0"
__all__ = [
    "TaskDefinition",
    "get_kv_task_definition",
    "get_entity_task_definition",
    "get_relation_task_definition",
    "get_unified_task_definition",
    "build_schema_prompt",
    "build_kv_extraction_prompt",
    "build_entity_extraction_prompt",
    "build_relation_extraction_prompt",
    "build_unified_extraction_prompt",
]
