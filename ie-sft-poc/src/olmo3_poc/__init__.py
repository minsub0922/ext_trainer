"""OLMo3 PoC extension layer for IE SFT.

This package contains experimental support and utilities for adapting the Information
Extraction Supervised Fine-Tuning pipeline to the OLMo3 model family. It serves as an
extensibility layer demonstrating how to integrate new model families beyond Qwen3.

Key components:
- adapter: ModelFamilyAdapter abstraction and concrete implementations
- conversion: OLMo3-specific data formatting and configuration utilities
- notes: Status tracking, TODO items, and known differences documentation

Note: OLMo3 support is currently in development. Some features may be incomplete
or require manual configuration adjustments.
"""

__version__ = "0.1.0"

from src.olmo3_poc.adapter import (
    ADAPTER_REGISTRY,
    ModelFamilyAdapter,
    OLMoAdapter,
    QwenAdapter,
    get_adapter,
)
from src.olmo3_poc.conversion import (
    adapt_training_config_for_olmo,
    convert_prompt_for_olmo,
)
from src.olmo3_poc.notes import (
    OLMO3_KNOWN_DIFFERENCES,
    OLMO3_STATUS,
    OLMO3_TODO_LIST,
    print_olmo3_status,
)

__all__ = [
    "ModelFamilyAdapter",
    "QwenAdapter",
    "OLMoAdapter",
    "get_adapter",
    "ADAPTER_REGISTRY",
    "convert_prompt_for_olmo",
    "adapt_training_config_for_olmo",
    "OLMO3_STATUS",
    "OLMO3_TODO_LIST",
    "OLMO3_KNOWN_DIFFERENCES",
    "print_olmo3_status",
]
