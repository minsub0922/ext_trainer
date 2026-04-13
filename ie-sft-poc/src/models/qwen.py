"""Qwen-specific model utilities and configurations.

Provides Qwen model family utilities including training parameter recommendations,
template configurations, and compatibility validation.
"""

from typing import Any

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# LoRA target modules for Qwen models
QWEN_LORA_TARGETS = [
    "all",  # Target all linear layers
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_qwen3_config(variant: str = "0.6b") -> dict[str, Any]:
    """Get recommended training parameters for Qwen3 variants.

    Args:
        variant: Qwen3 variant identifier ("0.6b" or "0.8b")

    Returns:
        Dictionary of recommended training parameters

    Raises:
        ValueError: If variant is not supported
    """
    variant = variant.lower().strip()

    # Base parameters common to all Qwen3 variants
    base_params = {
        "learning_rate": 2.0e-4,
        "num_train_epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
    }

    if variant == "0.6b":
        return {
            **base_params,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "max_length": 2048,
            "notes": "Optimized for 4x V100 or similar",
        }
    elif variant in ("0.8b", "3.5-0.8b"):
        return {
            **base_params,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "max_length": 2048,
            "notes": "Qwen3.5 variant, optimized for 4x V100 or similar",
        }
    else:
        raise ValueError(
            f"Unsupported Qwen3 variant: {variant}. "
            f"Supported: 0.6b, 0.8b, 3.5-0.8b"
        )


def get_qwen_template_name() -> str:
    """Get the LLaMA-Factory template name for Qwen models.

    Returns:
        Template identifier for use with LLaMA-Factory
    """
    return "qwen"


def validate_qwen_compatibility(config: dict[str, Any]) -> list[str]:
    """Validate Qwen model configuration for common issues.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of warning messages (empty if no issues found)
    """
    warnings = []

    # Check LoRA target validity
    lora_target = config.get("lora_target", "all")
    if lora_target != "all" and lora_target not in QWEN_LORA_TARGETS:
        warnings.append(
            f"Unknown LoRA target '{lora_target}'. "
            f"Consider using 'all' or one of: {', '.join(QWEN_LORA_TARGETS)}"
        )

    # Check batch size reasonableness
    batch_size = config.get("per_device_train_batch_size", 4)
    if batch_size > 16:
        warnings.append(
            f"Large batch size ({batch_size}) may cause OOM on typical GPUs. "
            f"Recommended: 4-8 with gradient accumulation."
        )

    # Check learning rate
    lr = config.get("learning_rate", 2.0e-4)
    if lr > 1e-3:
        warnings.append(
            f"High learning rate ({lr}). "
            f"Recommended for Qwen3: 1e-5 to 5e-4"
        )
    elif lr < 1e-5:
        warnings.append(
            f"Very low learning rate ({lr}). "
            f"Training may be very slow."
        )

    # Check max length
    max_len = config.get("max_length", 2048)
    if max_len > 4096:
        warnings.append(
            f"Very long context ({max_len}). "
            f"May cause memory issues. Qwen3 typically trained to 2048."
        )

    return warnings


if __name__ == "__main__":
    # Example usage
    print("Qwen3 0.6B Configuration:")
    print("=" * 50)
    config_06b = get_qwen3_config("0.6b")
    for key, value in config_06b.items():
        print(f"  {key}: {value}")

    print("\nQwen3.5 0.8B Configuration:")
    print("=" * 50)
    config_08b = get_qwen3_config("0.8b")
    for key, value in config_08b.items():
        print(f"  {key}: {value}")

    print(f"\nTemplate Name: {get_qwen_template_name()}")
    print(f"\nAvailable LoRA Targets: {', '.join(QWEN_LORA_TARGETS)}")

    # Validation example
    print("\nValidation Example:")
    test_config = {
        "lora_target": "all",
        "per_device_train_batch_size": 8,
        "learning_rate": 2.0e-4,
        "max_length": 2048,
    }
    warnings = validate_qwen_compatibility(test_config)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No compatibility warnings.")
