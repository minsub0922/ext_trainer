"""Model registry module for IE SFT PoC.

Provides centralized management of supported model configurations, including
registration, retrieval, and validation of model metadata for training workflows.
"""

from .model_registry import (
    ModelConfig,
    get_model,
    list_models,
    register_model,
)

__all__ = [
    "ModelConfig",
    "register_model",
    "get_model",
    "list_models",
]
