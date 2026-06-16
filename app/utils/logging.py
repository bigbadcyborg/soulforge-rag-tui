"""File logging setup for SoulForge."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import AppConfig, resolve_path

_CONFIGURED = False


def setup_logging(config: AppConfig) -> logging.Logger:
    """Configure root logger with optional file and console handlers."""
    global _CONFIGURED
    log_cfg = config.logging
    log_path = resolve_path(log_cfg.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level_name = log_cfg.level.upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("soulforge")
    root.setLevel(level)

    if _CONFIGURED:
        return root

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    if log_cfg.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root.addHandler(console_handler)

    _CONFIGURED = True
    root.info("Logging initialized at %s", log_path)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the soulforge namespace."""
    return logging.getLogger(f"soulforge.{name}")
