"""
Utility functions for reproducibility, logging, and runtime helpers.

Provides:
    - seed_everything: deterministic seeding for all RNGs
    - setup_logging: console + file logging configuration
    - get_available_gpus: enumerate CUDA devices
    - format_bytes / format_time: human-readable formatting
    - checkpoint_exists: validate checkpoint files
"""

import os
import random
import logging
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 2.0+ reproducibility settings
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Configure logging with both console and file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('discordance')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always capture everything in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_available_gpus() -> list:
    """
    Get list of available GPU IDs.

    Returns:
        List of GPU IDs
    """
    if not torch.cuda.is_available():
        return []

    return list(range(torch.cuda.device_count()))


def format_bytes(num_bytes: int) -> str:
    """
    Format bytes as human-readable string.

    Args:
        num_bytes: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_time(seconds: float) -> str:
    """
    Format seconds as human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def checkpoint_exists(checkpoint_path: str) -> bool:
    """
    Check if a checkpoint file exists and is valid.

    Args:
        checkpoint_path: Path to checkpoint

    Returns:
        True if checkpoint exists and is valid
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return False

    # Check if file is not empty
    if path.stat().st_size == 0:
        return False

    return True
