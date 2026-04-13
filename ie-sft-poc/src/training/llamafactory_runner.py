"""Python wrapper for running LLaMA-Factory training jobs.

Provides subprocess-based execution of LLaMA-Factory CLI commands with
configuration validation and logging.
"""

import subprocess
import sys
from pathlib import Path

from src.common.io import read_yaml
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def _check_llamafactory_installed() -> bool:
    """Check if llamafactory-cli is available in the current environment.

    Returns:
        True if llamafactory-cli is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ["llamafactory-cli", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger.debug(f"llamafactory-cli version: {result.stdout.strip()}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def run_sft(config_path: str | Path, dry_run: bool = False) -> int:
    """Run SFT training with LLaMA-Factory.

    Validates that the config file exists, resolves the configuration,
    and executes the training job via subprocess.

    Args:
        config_path: Path to LLaMA-Factory training YAML config
        dry_run: If True, only print resolved config without running

    Returns:
        Exit code from subprocess (0 for success, non-zero for failure)

    Raises:
        FileNotFoundError: If config file does not exist
        ValueError: If llamafactory-cli is not installed
    """
    config_path = Path(config_path)

    # Validate config exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading SFT config from: {config_path}")

    # Read and print config
    try:
        config = read_yaml(config_path)
    except Exception as e:
        logger.error(f"Failed to read config file: {e}")
        raise

    logger.info("Resolved SFT Configuration:")
    logger.info("=" * 60)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    # Check LLaMA-Factory installation
    if not _check_llamafactory_installed():
        raise ValueError(
            "llamafactory-cli not found. "
            "Install with: pip install llamafactory"
        )

    # Dry run: just print resolved config
    if dry_run:
        logger.info("DRY RUN: Would execute training with above config")
        return 0

    # Run training
    logger.info("Starting SFT training...")
    try:
        cmd = ["llamafactory-cli", "train", str(config_path)]
        logger.debug(f"Running command: {' '.join(cmd)}")

        process = subprocess.run(cmd, check=False)
        exit_code = process.returncode

        if exit_code == 0:
            logger.info("SFT training completed successfully")
        else:
            logger.error(f"SFT training failed with exit code {exit_code}")

        return exit_code

    except Exception as e:
        logger.error(f"Error running SFT training: {e}")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLaMA-Factory SFT training")
    parser.add_argument("config_path", help="Path to training config YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config without training",
    )

    args = parser.parse_args()

    try:
        exit_code = run_sft(args.config_path, dry_run=args.dry_run)
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
