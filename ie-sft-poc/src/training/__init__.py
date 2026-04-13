"""Training utilities for IE SFT PoC.

Provides tools for training data preparation, dataset registry management,
SFT config building, and LLaMA-Factory runner wrappers.
"""

from .config_builder import (
    SFTConfig,
    build_sft_config,
    export_sft_yaml,
    validate_sft_config,
)
from .dataset_registry_builder import (
    DatasetEntry,
    build_dataset_info,
    compute_file_sha1,
    write_dataset_info,
)
from .eval_runner import run_eval
from .inference_runner import run_batch_inference, run_inference
from .llamafactory_runner import run_sft

__all__ = [
    # Dataset registry
    "build_dataset_info",
    "write_dataset_info",
    "DatasetEntry",
    "compute_file_sha1",
    # SFT config building
    "SFTConfig",
    "build_sft_config",
    "export_sft_yaml",
    "validate_sft_config",
    # Runners
    "run_sft",
    "run_eval",
    "run_inference",
    "run_batch_inference",
]
