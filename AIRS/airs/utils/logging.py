"""
Logging configuration for AIRS.

Provides structured logging with consistent formatting across the application.
"""

import logging
import sys
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler

# Console for rich output
console = Console()

# Log format for file handlers
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_file: str | None = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for file logging
    """
    log_level = getattr(logging, level.upper())

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Rich console handler for development
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    rich_handler.setLevel(log_level)
    root_logger.addHandler(rich_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT, DATE_FORMAT))
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for logging with additional context."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation

    def __enter__(self) -> "LogContext":
        self.logger.info(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.logger.error(f"Failed: {self.operation} - {exc_val}")
        else:
            self.logger.info(f"Completed: {self.operation}")
        return False
