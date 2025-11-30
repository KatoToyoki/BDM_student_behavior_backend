"""
Logging configuration module.

Provides centralized logging functionality with both file and console output.
"""

import logging
from datetime import datetime
from pathlib import Path

# Global logger instance
_logger: logging.Logger | None = None


def setup_logger(
    name: str = "behavior_analysis",
    log_dir: str = "/home/jovyan/workspace/artifacts/logs",
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    Set up and configure the application logger.

    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to console

    Returns:
        logging.Logger: Configured logger instance
    """
    global _logger

    # Return existing logger if already configured
    if _logger is not None:
        return _logger

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = str(log_path / f"behavior_analysis_{timestamp}.log")

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
    )

    # File handler (detailed format)
    file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler (simple format) - optional
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Log initial message
    logger.info(f"Logger initialized. Log file: {log_filename}")

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Get the global logger instance.

    If logger hasn't been set up yet, initializes it with default settings.

    Returns:
        logging.Logger: The logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def set_log_level(level: int) -> None:
    """
    Change the logging level for all handlers.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logger = get_logger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def log_exception(exc: Exception, message: str = "An exception occurred") -> None:
    """
    Log an exception with traceback.

    Args:
        exc: The exception to log
        message: Additional context message
    """
    logger = get_logger()
    logger.error(f"{message}: {exc}", exc_info=True)


def log_progress(current: int, total: int, item_name: str = "items") -> None:
    """
    Log progress information.

    Args:
        current: Current progress count
        total: Total count
        item_name: Name of items being processed
    """
    logger = get_logger()
    percentage = (current / total * 100) if total > 0 else 0
    logger.info(f"Progress: {current}/{total} {item_name} ({percentage:.1f}%)")
