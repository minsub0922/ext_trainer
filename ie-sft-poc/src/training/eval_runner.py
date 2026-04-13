"""Python wrapper for running LLaMA-Factory evaluation jobs.

Provides subprocess-based execution of LLaMA-Factory evaluation with
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


def run_eval(config_path: str | Path, dry_run: bool = False) -> int:
    """Run evaluation with LLaMA-Factory.

    Validates that the config file exists, resolves the configuration,
    and executes the evaluation job via subprocess.

    Args:
        config_path: Path to LLaMA-Factory evaluation YAML config
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

    logger.info(f"Loading eval config from: {config_path}")

    # Read and print config
    try:
        config = read_yaml(config_path)
    except Exception as e:
        logger.error(f"Failed to read config file: {e}")
        raise

    logger.info("Resolved Eval Configuration:")
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
        logger.info("DRY RUN: Would execute evaluation with above config")
        return 0

    # Run evaluation
    logger.info("Starting evaluation...")
    try:
        cmd = ["llamafactory-cli", "eval", str(config_path)]
        logger.debug(f"Running command: {' '.join(cmd)}")

        process = subprocess.run(cmd, check=False)
        exit_code = process.returncode

        if exit_code == 0:
            logger.info("Evaluation completed successfully")
        else:
            logger.error(f"Evaluation failed with exit code {exit_code}")

        return exit_code

    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLaMA-Factory evaluation")
    parser.add_argument("config_path", help="Path to evaluation config YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config without evaluation",
    )

    args = parser.parse_args()

    try:
        exit_code = run_eval(args.config_path, dry_run=args.dry_run)
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
