"""Build and manage LLaMA-Factory SFT training configurations programmatically.

Provides tools for constructing, validating, and exporting SFT training configs
compatible with LLaMA-Factory, supporting model registry lookup and override patterns.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from src.common.io import write_yaml
from src.common.logging_utils import get_logger
from src.models.model_registry import get_model

logger = get_logger(__name__)


@dataclass
class SFTConfig:
    """Complete SFT training configuration for LLaMA-Factory.

    Maps to LLaMA-Factory training arguments. All fields are optional with sensible defaults.
    """

    # Model configuration
    model_name_or_path: str
    template: str

    # Dataset configuration
    dataset: str
    dataset_dir: str
    output_dir: str

    # Training method
    stage: str = "sft"
    do_train: bool = True
    finetuning_type: str = "lora"
    lora_target: str = "all"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training hyperparameters
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    num_train_epochs: int = 3
    max_length: int = 2048  # cutoff_len in LLaMA-Factory
    cutoff_len: int = field(default=0, init=False)  # Auto-set from max_length

    # Optimizer and scheduling
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Precision and computation
    bf16: bool = True
    fp16: bool = False
    preprocessing_num_workers: int = 4

    # Evaluation and checkpointing
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 10

    # Other settings
    ddp_timeout: int = 180000000
    seed: int = 42
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])

    def __post_init__(self) -> None:
        """Post-initialization: Set derived fields and validate."""
        # Set cutoff_len from max_length for LLaMA-Factory compatibility
        self.cutoff_len = self.max_length

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for YAML export.

        Returns:
            Dictionary representation suitable for LLaMA-Factory YAML
        """
        config_dict = asdict(self)
        # Remove cutoff_len since it's derived, but ensure we export max_length as cutoff_len
        config_dict.pop("cutoff_len", None)
        config_dict["cutoff_len"] = self.max_length
        return config_dict


def build_sft_config(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    **overrides: Any,
) -> SFTConfig:
    """Build an SFT configuration by looking up model in registry and applying overrides.

    Args:
        model_name: Registered model name (e.g., "qwen3-0.6b")
        dataset_path: Path to dataset directory
        output_dir: Path to output directory
        **overrides: Keyword arguments to override defaults

    Returns:
        Configured SFTConfig instance

    Raises:
        KeyError: If model not found in registry
    """
    # Look up model in registry
    model_config = get_model(model_name)

    logger.info(f"Building SFT config for model: {model_name}")
    logger.info(f"  Model family: {model_config.family}")
    logger.info(f"  Model path: {model_config.model_name_or_path}")

    # Create base config from model
    config_dict = {
        "model_name_or_path": model_config.model_name_or_path,
        "template": model_config.template,
        "dataset": "ie_sft_unified",
        "dataset_dir": str(dataset_path),
        "output_dir": str(output_dir),
        "lora_target": model_config.default_lora_target,
        "max_length": model_config.default_max_length,
        "bf16": model_config.bf16,
        "per_device_train_batch_size": model_config.recommended_batch_size,
    }

    # Apply overrides
    config_dict.update(overrides)

    logger.debug(f"Applying {len(overrides)} config overrides")

    # Create and return config
    config = SFTConfig(**config_dict)

    logger.info(f"SFT config built successfully")
    logger.debug(f"Config: {config}")

    return config


def export_sft_yaml(config: SFTConfig, path: Path | str) -> None:
    """Export SFT configuration to LLaMA-Factory YAML format.

    Args:
        config: SFTConfig instance to export
        path: Output path for YAML file

    Raises:
        IOError: If file cannot be written
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict()

    write_yaml(config_dict, path)
    logger.info(f"Exported SFT config to: {path}")


def validate_sft_config(config: SFTConfig) -> list[str]:
    """Validate SFT configuration for common issues.

    Args:
        config: SFTConfig to validate

    Returns:
        List of issue messages (empty if valid)
    """
    issues = []

    # Validate paths exist or are reasonable
    dataset_path = Path(config.dataset_dir)
    if not dataset_path.exists():
        issues.append(f"Dataset directory does not exist: {config.dataset_dir}")

    # Check batch size and gradient accumulation
    total_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    if total_batch > 128:
        issues.append(
            f"Large effective batch size ({total_batch}). "
            f"Actual GPU memory usage may be high."
        )

    # Check learning rate
    if config.learning_rate > 1e-3:
        issues.append(
            f"High learning rate ({config.learning_rate}). "
            f"May cause training instability."
        )
    elif config.learning_rate < 1e-6:
        issues.append(
            f"Very low learning rate ({config.learning_rate}). "
            f"Training may be ineffective."
        )

    # Check max length
    if config.max_length > 4096:
        issues.append(
            f"Very long sequence length ({config.max_length}). "
            f"May cause memory issues."
        )
    elif config.max_length < 256:
        issues.append(
            f"Very short sequence length ({config.max_length}). "
            f"May limit IE task quality."
        )

    # Check eval settings
    if config.eval_steps > config.save_steps:
        issues.append(
            f"eval_steps ({config.eval_steps}) > save_steps ({config.save_steps}). "
            f"Evaluation will not happen before saves."
        )

    # Check precision flags
    if config.bf16 and config.fp16:
        issues.append("Both bf16 and fp16 are enabled. Only one should be used.")

    # Check num_train_epochs reasonableness
    if config.num_train_epochs > 20:
        issues.append(
            f"Very high epoch count ({config.num_train_epochs}). "
            f"May cause overfitting."
        )

    return issues


if __name__ == "__main__":
    # Example usage
    print("Example SFT Config Building:")
    print("=" * 60)

    # Build config for Qwen3-0.6B
    try:
        config = build_sft_config(
            model_name="qwen3-0.6b",
            dataset_path="data/processed/llamafactory",
            output_dir="outputs/qwen3-0.6b-ie-lora",
            num_train_epochs=3,
            lora_rank=16,
        )

        print("Built config:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")

        # Validate
        print("\nValidation Results:")
        issues = validate_sft_config(config)
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No validation issues found.")

        # Export example
        print("\nWould export to YAML (example path only):")
        print("  /tmp/qwen3_sft_config.yaml")

    except KeyError as e:
        print(f"Error: {e}")
