"""Central path management for the IE SFT PoC project.

Provides centralized management of all project directories with support for
environment variable overrides. Uses pathlib.Path for cross-platform compatibility.
"""

import os
from pathlib import Path

# Project root - inferred as the parent directory of 'src'
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Allow override via environment variable
if "IESFT_PROJECT_ROOT" in os.environ:
    PROJECT_ROOT = Path(os.environ["IESFT_PROJECT_ROOT"]).resolve()


def _get_path(env_var: str, default: Path) -> Path:
    """
    Get a directory path with optional environment variable override.

    Args:
        env_var: Environment variable name to check
        default: Default path relative to PROJECT_ROOT

    Returns:
        Resolved Path object
    """
    if env_var in os.environ:
        return Path(os.environ[env_var]).resolve()
    return (PROJECT_ROOT / default).resolve()


# Data directories
DATA_DIR = _get_path("IESFT_DATA_DIR", Path("data"))
DATA_RAW = _get_path("IESFT_DATA_RAW", DATA_DIR / "raw")
DATA_INTERIM = _get_path("IESFT_DATA_INTERIM", DATA_DIR / "interim")
DATA_PROCESSED = _get_path("IESFT_DATA_PROCESSED", DATA_DIR / "processed")
DATA_METADATA = _get_path("IESFT_DATA_METADATA", DATA_DIR / "metadata")

# Configuration and model directories
CONFIGS_DIR = _get_path("IESFT_CONFIGS_DIR", Path("configs"))
MODELS_DIR = _get_path("IESFT_MODELS_DIR", PROJECT_ROOT / "models")
CHECKPOINTS_DIR = _get_path("IESFT_CHECKPOINTS_DIR", MODELS_DIR / "checkpoints")

# Output and logs
OUTPUTS_DIR = _get_path("IESFT_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
LOGS_DIR = _get_path("IESFT_LOGS_DIR", OUTPUTS_DIR / "logs")
RESULTS_DIR = _get_path("IESFT_RESULTS_DIR", OUTPUTS_DIR / "results")

# Source code
SRC_DIR = _get_path("IESFT_SRC_DIR", Path("src"))


def ensure_directories_exist() -> None:
    """Create all necessary project directories if they don't exist."""
    directories = [
        DATA_RAW,
        DATA_INTERIM,
        DATA_PROCESSED,
        DATA_METADATA,
        CONFIGS_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR,
        OUTPUTS_DIR,
        LOGS_DIR,
        RESULTS_DIR,
        SRC_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_path(config_name: str) -> Path:
    """
    Get the full path to a configuration file.

    Args:
        config_name: Name of the config file (with or without .yaml extension)

    Returns:
        Path to the configuration file
    """
    if not config_name.endswith((".yaml", ".yml", ".json")):
        config_name = f"{config_name}.yaml"

    return CONFIGS_DIR / config_name


def get_data_path(dataset_name: str, split: str = "train", filetype: str = "jsonl") -> Path:
    """
    Get the standard path for a processed dataset file.

    Args:
        dataset_name: Name of the dataset
        split: Data split (train, dev, test)
        filetype: File extension (jsonl, json, etc.)

    Returns:
        Path to the dataset file
    """
    return DATA_PROCESSED / dataset_name / f"{split}.{filetype}"


def get_model_checkpoint_path(model_name: str, checkpoint: str = "best") -> Path:
    """
    Get the path to a model checkpoint.

    Args:
        model_name: Name of the model
        checkpoint: Checkpoint identifier (best, latest, etc.) or epoch number

    Returns:
        Path to the checkpoint directory
    """
    return CHECKPOINTS_DIR / model_name / checkpoint


if __name__ == "__main__":
    # Print all configured paths for debugging
    print("Project Configuration Paths")
    print("=" * 50)
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_RAW: {DATA_RAW}")
    print(f"DATA_INTERIM: {DATA_INTERIM}")
    print(f"DATA_PROCESSED: {DATA_PROCESSED}")
    print(f"DATA_METADATA: {DATA_METADATA}")
    print(f"CONFIGS_DIR: {CONFIGS_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"CHECKPOINTS_DIR: {CHECKPOINTS_DIR}")
    print(f"OUTPUTS_DIR: {OUTPUTS_DIR}")
    print(f"LOGS_DIR: {LOGS_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"SRC_DIR: {SRC_DIR}")
