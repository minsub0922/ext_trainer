"""Model family adapter abstraction for extensible model integration.

This module provides an abstraction layer for integrating different model families
into the IE SFT pipeline. Each model family may have unique tokenizer quirks, template
formats, and configuration requirements. The adapter pattern allows us to handle these
differences in a centralized, composable way.
"""

from abc import ABC, abstractmethod
from typing import Any


class ModelFamilyAdapter(ABC):
    """Abstract base class for model family adapters.

    Each adapter encapsulates the specific knowledge needed to work with a particular
    model family (e.g., Qwen3, OLMo3, LLaMA). Adapters handle differences in:
    - Tokenizer behavior and special token handling
    - Prompt template formatting
    - Configuration overrides for training
    - Environment validation

    Subclasses should implement all abstract methods to provide family-specific
    customizations.
    """

    @property
    @abstractmethod
    def family_name(self) -> str:
        """Return the model family name (e.g., 'qwen3', 'olmo3').

        Returns:
            String identifier for this model family
        """
        pass

    @property
    @abstractmethod
    def default_template(self) -> str:
        """Return the default chat template name for this family.

        The template name is used by LLaMA-Factory to apply the correct prompt
        formatting during training and inference.

        Returns:
            Template name (e.g., 'qwen', 'olmo', 'llama2')
        """
        pass

    @abstractmethod
    def tokenizer_quirks(self) -> dict[str, Any]:
        """Return tokenizer-specific quirks and configurations.

        This method documents and returns any non-standard tokenizer behaviors
        specific to this model family, such as:
        - Special token definitions that differ from standard
        - Padding side preferences (left vs right)
        - Trust_remote_code requirements
        - Custom token merging strategies

        Returns:
            Dictionary of tokenizer quirks and their values
        """
        pass

    @abstractmethod
    def recommended_prompt_style(self) -> str:
        """Return the recommended prompt formatting style for this family.

        Different models may prefer different ways of structuring prompts:
        - Instruction + Response format
        - System + User + Assistant
        - Custom delimiters or markers

        Returns:
            String describing the recommended prompt style
        """
        pass

    @abstractmethod
    def training_config_overrides(self) -> dict[str, Any]:
        """Return training configuration overrides for this family.

        Some model families require specific training settings to work well.
        These might include:
        - Learning rate adjustments
        - Warmup strategies
        - Gradient accumulation settings
        - Loss scaling configurations

        Returns:
            Dictionary of config keys and override values
        """
        pass

    @abstractmethod
    def validate_environment(self) -> list[str]:
        """Validate that the environment is properly set up for this family.

        Checks for required dependencies, models being available, etc.
        Should return a list of issues found (empty if all checks pass).

        Returns:
            List of validation issue strings (empty if all checks pass)
        """
        pass


class QwenAdapter(ModelFamilyAdapter):
    """Adapter for the Qwen3 model family.

    Implements Qwen3-specific handling for tokenizer, templates, and training.
    Qwen3 uses a standard chat template with clear role delineation.
    """

    @property
    def family_name(self) -> str:
        """Return 'qwen3' as the family name."""
        return "qwen3"

    @property
    def default_template(self) -> str:
        """Return the Qwen chat template name."""
        return "qwen"

    def tokenizer_quirks(self) -> dict[str, Any]:
        """Return Qwen3 tokenizer quirks.

        Qwen3 uses standard tokenizer behavior with minimal special handling.
        The tokenizer is available on HuggingFace and requires no special flags.
        """
        return {
            "padding_side": "right",
            "trust_remote_code": False,
            "use_fast": True,
        }

    def recommended_prompt_style(self) -> str:
        """Return Qwen3 recommended prompt style.

        Qwen3 works well with the standard instruction + response format,
        with clear delimiters between user and assistant turns.
        """
        return "instruction_response"

    def training_config_overrides(self) -> dict[str, Any]:
        """Return Qwen3 training configuration overrides.

        Qwen3 models train well with standard settings. No overrides needed
        for typical IE fine-tuning tasks.
        """
        return {
            "learning_rate": 5e-5,
            "warmup_ratio": 0.1,
            "gradient_accumulation_steps": 2,
        }

    def validate_environment(self) -> list[str]:
        """Validate Qwen3 environment setup.

        Checks that Qwen tokenizer can be imported successfully.
        """
        issues = []
        try:
            from transformers import AutoTokenizer

            # Try to load a Qwen tokenizer to verify it's available
            _ = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=False)
        except Exception as e:
            issues.append(f"Cannot load Qwen tokenizer: {str(e)}")
        return issues


class OLMoAdapter(ModelFamilyAdapter):
    """Adapter for the OLMo3 model family.

    This adapter provides experimental support for OLMo3 models. OLMo3 is an
    open-source model from AI2 with some differences from Qwen3 that require
    special handling.

    Known Limitations:
    - Custom tokenizer handling may be required
    - Special token definitions differ from Qwen
    - Template format not yet fully tested with IE tasks
    - Some training config tuning may be needed

    TODO Items:
    - Verify tokenizer behavior with actual OLMo3 checkpoint
    - Test and validate chat template format
    - Benchmark training performance and adjust learning rates
    - Evaluate inference quality on IE tasks
    """

    @property
    def family_name(self) -> str:
        """Return 'olmo3' as the family name."""
        return "olmo3"

    @property
    def default_template(self) -> str:
        """Return the OLMo3 chat template name.

        TODO: Verify this is the correct template name for OLMo3.
        May need to use 'olmo' or a custom template depending on final model format.
        """
        return "olmo"

    def tokenizer_quirks(self) -> dict[str, Any]:
        """Return OLMo3 tokenizer quirks.

        OLMo3 uses a custom tokenizer that may require special handling.

        TODO: Verify the following assumptions with actual OLMo3 checkpoint:
        - Whether trust_remote_code is actually required
        - Actual padding side preference
        - Whether use_fast tokenizer is compatible
        - Special token handling (e.g., BOS, EOS, PAD tokens)
        """
        return {
            "padding_side": "right",
            "trust_remote_code": True,  # OLMo3 may use custom tokenizer code
            "use_fast": False,  # TODO: Test if fast tokenizer works
            "add_bos_token": True,  # TODO: Verify BOS token handling
            "add_eos_token": True,  # TODO: Verify EOS token handling
        }

    def recommended_prompt_style(self) -> str:
        """Return OLMo3 recommended prompt style.

        TODO: Determine if OLMo3 prefers a different prompt style than Qwen3.
        The format may differ in:
        - Role delimiters (e.g., <|im_start|> vs custom markers)
        - System vs user message ordering
        - Special tokens for conversation structure
        """
        return "instruction_response"

    def training_config_overrides(self) -> dict[str, Any]:
        """Return OLMo3 training configuration overrides.

        OLMo3 may require different training settings than Qwen3.

        TODO: Benchmark and adjust these values:
        - Learning rate: Start with 5e-5 but may need 1e-4 or 1e-5
        - Warmup ratio: Experiment with 0.05 to 0.15
        - Gradient accumulation: Test with values 1-4
        - Weight decay: May need tuning
        - LR scheduler type: Consider cosine vs linear
        """
        return {
            "learning_rate": 5e-5,  # TODO: Benchmark and adjust
            "warmup_ratio": 0.1,  # TODO: Test different warmup strategies
            "gradient_accumulation_steps": 2,  # TODO: Adjust based on VRAM
            "weight_decay": 0.01,  # TODO: Tune this value
            "lr_scheduler_type": "cosine",  # TODO: Compare with linear
        }

    def validate_environment(self) -> list[str]:
        """Validate OLMo3 environment setup.

        Checks that OLMo3 tokenizer and model files can be accessed.

        TODO: Update with actual OLMo3 model paths and requirements
        once the final model is released.
        """
        issues = []

        try:
            from transformers import AutoTokenizer

            # TODO: Use the actual OLMo3 model identifier
            # This is a placeholder that will fail until OLMo3 is released
            _ = AutoTokenizer.from_pretrained("allenai/OLMo3-base", trust_remote_code=True)
        except Exception as e:
            issues.append(
                f"Cannot load OLMo3 tokenizer. "
                f"Make sure OLMo3 model is available. Error: {str(e)}"
            )

        return issues


# Registry mapping family names to adapter instances
ADAPTER_REGISTRY: dict[str, ModelFamilyAdapter] = {
    "qwen3": QwenAdapter(),
    "olmo3": OLMoAdapter(),
}


def get_adapter(family: str) -> ModelFamilyAdapter:
    """Retrieve an adapter for the specified model family.

    Args:
        family: Model family name (e.g., 'qwen3', 'olmo3')

    Returns:
        ModelFamilyAdapter instance for the requested family

    Raises:
        ValueError: If the family is not in the adapter registry
    """
    if family not in ADAPTER_REGISTRY:
        available = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model family '{family}'. "
            f"Available adapters: {available}"
        )
    return ADAPTER_REGISTRY[family]
