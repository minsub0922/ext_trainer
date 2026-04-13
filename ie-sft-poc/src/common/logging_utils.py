"""Logging utilities for the IE SFT PoC.

Provides simple but robust logging configuration with timestamps and
configurable log levels.
"""

import logging
import logging.handlers
from pathlib import Path

from src.common.paths import LOGS_DIR


def get_logger(
    name: str,
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Get or create a logger with standard configuration.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
        log_file: Optional path to log file. If provided, logs to both console and file.
        format_string: Custom format string. If None, uses default format with timestamps.

    Returns:
        Configured logger instance
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger already configured
    if logger.hasHandlers():
        return logger

    # Default format with timestamp
    if format_string is None:
        format_string = (
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        )

    formatter = logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_logging(
    level: int | str = logging.INFO,
    log_dir: Path | str | None = None,
    log_file_name: str = "ie_sft.log",
) -> logging.Logger:
    """
    Set up root logger with console and file handlers.

    Args:
        level: Root logging level
        log_dir: Directory for log files (default: LOGS_DIR)
        log_file_name: Name of the log file

    Returns:
        Root logger instance
    """
    if log_dir is None:
        log_dir = LOGS_DIR

    log_dir = Path(log_dir)
    log_file = log_dir / log_file_name

    return get_logger(
        "ie_sft",
        level=level,
        log_file=log_file,
    )


def disable_logging(logger_name: str | None = None) -> None:
    """
    Disable logging for a specific logger or root logger.

    Args:
        logger_name: Name of logger to disable. If None, disables root logger.
    """
    if logger_name is None:
        logging.getLogger().disabled = True
    else:
        logging.getLogger(logger_name).disabled = True


def enable_logging(logger_name: str | None = None) -> None:
    """
    Enable logging for a specific logger or root logger.

    Args:
        logger_name: Name of logger to enable. If None, enables root logger.
    """
    if logger_name is None:
        logging.getLogger().disabled = False
    else:
        logging.getLogger(logger_name).disabled = False


if __name__ == "__main__":
    # Example usage
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up logging with file output
        logger = setup_logging(
            level=logging.DEBUG,
            log_dir=tmpdir,
            log_file_name="test.log",
        )

        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")

        # Check that log file was created
        log_file = Path(tmpdir) / "test.log"
        if log_file.exists():
            print(f"Log file created at {log_file}")
            print(f"File size: {log_file.stat().st_size} bytes")
        else:
            print("Error: Log file was not created")
