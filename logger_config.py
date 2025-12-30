"""
Logging configuration for the image translation pipeline.
Provides structured logging with file and console output.
"""

import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "image_translation",
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. If no logger exists, creates one with defaults.
    
    Args:
        name: Logger name (default: "image_translation")
        
    Returns:
        Logger instance
    """
    if name is None:
        name = "image_translation"
    
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up default configuration
    if not logger.handlers:
        setup_logger(name)
    
    return logger

