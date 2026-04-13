"""Conversion utilities for OLMo3-specific data formatting.

This module handles the conversion of prompts and configurations from
the standard format to OLMo3-specific formats. It serves as a bridge
between the generic IE pipeline and OLMo3's unique requirements.
"""

from typing import Any


def convert_prompt_for_olmo(prompt: str, template: str) -> str:
    """Convert a generic prompt to OLMo3-specific format.

    OLMo3 may use different prompt delimiters and structural markers
    compared to Qwen. This function applies those transformations.

    Args:
        prompt: The input prompt in standard format
        template: The template name being used

    Returns:
        Prompt formatted for OLMo3

    Example:
        Standard format might use <|im_start|> and <|im_end|> markers.
        OLMo3 might use different markers or require different escaping.

    TODO: Once OLMo3 final format is known, implement actual conversions:
    - Map Qwen template markers to OLMo markers if different
    - Handle special token differences
    - Adjust whitespace/newline handling if needed
    - Test with actual OLMo3 model to ensure compatibility
    """
    if template == "olmo":
        # Placeholder for OLMo-specific formatting
        # This is where we would apply OLMo-specific transformations
        # For now, return the prompt unchanged pending actual OLMo template details

        # TODO: Implement actual conversion logic:
        # 1. Replace Qwen's role markers with OLMo equivalents
        # 2. Handle special token escaping
        # 3. Adjust formatting based on OLMo tokenizer requirements
        # 4. Test token count to ensure prompt fits within context

        return prompt
    else:
        # For other templates, return unchanged
        return prompt


def adapt_training_config_for_olmo(base_config: dict[str, Any]) -> dict[str, Any]:
    """Adapt a base training config for OLMo3-specific needs.

    OLMo3 may require different configuration values than other models,
    particularly for:
    - Learning rate scheduling
    - Gradient handling
    - Warmup strategies
    - Memory management

    Args:
        base_config: Base training configuration dictionary

    Returns:
        OLMo3-adapted configuration dictionary

    Example:
        base_config might have learning_rate=5e-5, but OLMo3 might
        need different values based on its architecture and training
        characteristics.

    TODO: Once we have more data on OLMo3 training behavior:
    - Benchmark different learning rates and document results
    - Determine optimal warmup strategies
    - Test gradient accumulation impact on convergence
    - Validate mixed precision (bf16) behavior
    - Measure max batch size before OOM
    """
    adapted_config = base_config.copy()

    # OLMo3-specific configuration overrides
    # These are placeholder values pending actual tuning

    # TODO: Replace these with actual values based on benchmarks:
    # Current thinking:
    # - OLMo3 may be more sensitive to learning rate than Qwen3
    # - Start with 1e-4 or 5e-5 depending on model size
    # - Warmup might need to be shorter (0.05) or longer (0.15)
    # - Gradient accumulation could affect convergence behavior

    olmo_overrides = {
        "learning_rate": 5e-5,  # TODO: Benchmark different values
        "warmup_ratio": 0.1,  # TODO: Test 0.05, 0.1, 0.15
        "gradient_accumulation_steps": 2,  # TODO: Test 1, 2, 4
        "max_grad_norm": 1.0,  # TODO: Verify clipping threshold
        "weight_decay": 0.01,  # TODO: Test 0.0, 0.01, 0.1
    }

    # Merge OLMo-specific settings into the config
    adapted_config.update(olmo_overrides)

    # Handle logging configuration for OLMo3
    if "logging_steps" in adapted_config:
        # TODO: Adjust logging frequency if needed
        pass

    if "save_steps" in adapted_config:
        # TODO: Adjust checkpoint saving frequency if needed
        pass

    # Ensure bf16 is appropriate for OLMo3
    # TODO: Verify that bf16 training is stable with OLMo3
    adapted_config.setdefault("bf16", True)

    # Document the differences made
    adapted_config["_note_olmo3_adapted"] = (
        "Config has been adapted for OLMo3. Some values are placeholders "
        "pending actual benchmarking. See conversion.py TODO items."
    )

    return adapted_config


def document_template_differences() -> dict[str, Any]:
    """Document known differences between Qwen and OLMo3 templates.

    Returns:
        Dictionary describing the differences

    This is informational and helps understand why certain conversions
    are necessary when switching between model families.
    """
    return {
        "qwen_markers": {
            "user": "<|im_start|>user",
            "assistant": "<|im_start|>assistant",
            "end": "<|im_end|>",
        },
        "olmo_markers": {
            "user": "[TODO: Determine OLMo marker]",
            "assistant": "[TODO: Determine OLMo marker]",
            "end": "[TODO: Determine OLMo marker]",
        },
        "special_tokens": {
            "qwen": {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
            },
            "olmo3": {
                "bos_token": "[TODO: Verify]",
                "eos_token": "[TODO: Verify]",
                "pad_token": "[TODO: Verify]",
            },
        },
        "whitespace_handling": {
            "qwen": "Preserves newlines in role markers",
            "olmo3": "[TODO: Determine behavior]",
        },
        "context_length": {
            "qwen": 2048,  # or higher depending on version
            "olmo3": "[TODO: Confirm maximum context]",
        },
        "notes": [
            "These differences impact how prompts must be formatted",
            "Each difference may require a conversion step in convert_prompt_for_olmo",
            "Actual differences should be validated with real OLMo3 checkpoint",
            "See adapter.py for more context on OLMo3 status",
        ],
    }
