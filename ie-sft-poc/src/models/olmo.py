"""OLMo-specific model utilities and configurations.

Provides OLMo model family utilities including training parameter recommendations,
template configurations, and compatibility validation.

TODO: Update with actual OLMo3 1B model path once available
TODO: Handle OLMo-specific tokenizer quirks and special tokens
TODO: Add special token handling for OLMo conversation format
"""

from typing import Any

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# LoRA target modules for OLMo models
# TODO: Verify actual layer names in OLMo3 implementation
OLMO_LORA_TARGETS = [
    "all",  # Target all linear layers
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_olmo3_config() -> dict[str, Any]:
    """Get recommended training parameters for OLMo3 1B POC model.

    Returns:
        Dictionary of recommended training parameters

    Note:
        These are placeholder recommendations. Adjust based on actual
        model characteristics and hardware specifications.
    """
    return {
        "learning_rate": 2.0e-4,
        "num_train_epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "max_length": 2048,
        "notes": "OLMo3 POC configuration - verify with actual model",
    }


def get_olmo_template_name() -> str:
    """Get the LLaMA-Factory template name for OLMo models.

    Returns:
        Template identifier for use with LLaMA-Factory

    TODO: Verify correct template name for OLMo3
    """
    return "default"


def validate_olmo_compatibility(config: dict[str, Any]) -> list[str]:
    """Validate OLMo model configuration for common issues.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of warning messages (empty if no issues found)

    TODO: Add OLMo-specific validation rules
    TODO: Check for tokenizer compatibility
    TODO: Verify special token handling
    """
    warnings = []

    # Check LoRA target validity
    lora_target = config.get("lora_target", "all")
    if lora_target != "all" and lora_target not in OLMO_LORA_TARGETS:
        warnings.append(
            f"Unknown LoRA target '{lora_target}'. "
            f"Consider using 'all' or one of: {', '.join(OLMO_LORA_TARGETS)}"
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
            f"Recommended for OLMo3: 1e-5 to 5e-4"
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
            f"May cause memory issues. OLMo3 context length TBD."
        )

    # POC warning
    warnings.append(
        "This is a POC configuration. Verify all settings with actual OLMo3 model."
    )

    return warnings


if __name__ == "__main__":
    # Example usage
    print("OLMo3 POC Configuration:")
    print("=" * 50)
    config = get_olmo3_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\nTemplate Name: {get_olmo_template_name()}")
    print(f"\nAvailable LoRA Targets: {', '.join(OLMO_LORA_TARGETS)}")

    # Validation example
    print("\nValidation Example:")
    test_config = {
        "lora_target": "all",
        "per_device_train_batch_size": 8,
        "learning_rate": 2.0e-4,
        "max_length": 2048,
    }
    warnings = validate_olmo_compatibility(test_config)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No compatibility warnings.")
