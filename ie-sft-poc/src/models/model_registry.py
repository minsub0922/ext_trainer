"""Model registry for managing supported model configurations.

Provides a centralized registry for model metadata including HuggingFace paths,
template configurations, LoRA settings, and training recommendations.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.common.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration metadata for a registered model.

    Attributes:
        name: Unique identifier for the model (e.g., "qwen3-0.6b")
        family: Model family (e.g., "qwen3", "olmo3")
        model_name_or_path: HuggingFace model path or identifier
        template: Chat template name for LLaMA-Factory
        default_lora_target: Default LoRA target modules (e.g., "all" or specific module names)
        default_max_length: Default context length / cutoff
        bf16: Whether to use bfloat16 training by default
        recommended_batch_size: Recommended batch size for typical setups
        notes: Additional notes or warnings about the model
    """

    name: str
    family: str
    model_name_or_path: str
    template: str
    default_lora_target: str
    default_max_length: int = 2048
    bf16: bool = True
    recommended_batch_size: int = 4
    notes: str = ""


# Global model registry
MODEL_REGISTRY: dict[str, ModelConfig] = {}


def register_model(config: ModelConfig) -> None:
    """Register a model configuration in the global registry.

    Args:
        config: ModelConfig instance to register

    Raises:
        ValueError: If model name is already registered
    """
    if config.name in MODEL_REGISTRY:
        logger.warning(f"Model '{config.name}' already registered, overwriting")
    MODEL_REGISTRY[config.name] = config
    logger.debug(f"Registered model: {config.name} (family: {config.family})")


def get_model(name: str) -> ModelConfig:
    """Retrieve a model configuration by name.

    Args:
        name: Model name to retrieve

    Returns:
        ModelConfig for the requested model

    Raises:
        KeyError: If model name not found in registry
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Model '{name}' not found in registry. Available models: {available}"
        )
    return MODEL_REGISTRY[name]


def list_models() -> list[str]:
    """List all registered model names.

    Returns:
        Sorted list of registered model names
    """
    return sorted(MODEL_REGISTRY.keys())


# Pre-register models
_QWEN3_0_6B = ModelConfig(
    name="qwen3-0.6b",
    family="qwen3",
    model_name_or_path="Qwen/Qwen3-0.6B",
    template="qwen",
    default_lora_target="all",
    default_max_length=2048,
    bf16=True,
    recommended_batch_size=4,
    notes="Qwen3 0.6B base model optimized for IE tasks",
)

_QWEN3_4B = ModelConfig(
    name="qwen3-4b",
    family="qwen3",
    model_name_or_path="Qwen/Qwen3-4B",
    template="qwen",
    default_lora_target="all",
    default_max_length=2048,
    bf16=True,
    recommended_batch_size=2,
    notes="Qwen3 4B model for IE tasks — mid-size quality/cost tradeoff",
)

_QWEN3_5_0_8B = ModelConfig(
    name="qwen3.5-0.8b",
    family="qwen3",
    model_name_or_path="Qwen/Qwen3.5-0.8B",
    template="qwen3_5_nothink",
    default_lora_target="all",
    default_max_length=2048,
    bf16=True,
    recommended_batch_size=4,
    notes="Qwen3.5 0.8B base model optimized for IE tasks",
)

_OLMO3_1B_POC = ModelConfig(
    name="olmo3-1b-poc",
    family="olmo3",
    model_name_or_path="allenai/OLMo-2-0325-32B-Instruct",
    template="default",
    default_lora_target="all",
    default_max_length=2048,
    bf16=True,
    recommended_batch_size=4,
    notes="OLMo3 POC placeholder - update model path for actual model",
)

# Register pre-defined models
register_model(_QWEN3_0_6B)
register_model(_QWEN3_4B)
register_model(_QWEN3_5_0_8B)
register_model(_OLMO3_1B_POC)


if __name__ == "__main__":
    # Example usage and registry inspection
    print("Registered Models:")
    print("=" * 60)
    for model_name in list_models():
        config = get_model(model_name)
        print(f"\n{model_name}:")
        print(f"  Family: {config.family}")
        print(f"  Path: {config.model_name_or_path}")
        print(f"  Template: {config.template}")
        print(f"  LoRA Target: {config.default_lora_target}")
        print(f"  Max Length: {config.default_max_length}")
        print(f"  BF16: {config.bf16}")
        print(f"  Recommended Batch Size: {config.recommended_batch_size}")
        if config.notes:
            print(f"  Notes: {config.notes}")
