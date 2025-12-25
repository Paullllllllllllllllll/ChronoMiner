"""Centralized logger configuration."""

import logging
from pathlib import Path

from modules.config.loader import get_config_loader
from modules.core.path_utils import ensure_path_safe


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_logs_dir() -> Path:
    try:
        loader = get_config_loader()
        paths_config = loader.get_paths_config() or {}
        general = paths_config.get("general", {}) or {}
        raw_logs_dir = general.get("logs_dir")
        if raw_logs_dir:
            logs_dir = Path(raw_logs_dir)
            if not logs_dir.is_absolute():
                return (PROJECT_ROOT / logs_dir).resolve()
            return logs_dir
    except Exception:
        pass
    return PROJECT_ROOT / "logs"


def setup_logger(name: str) -> logging.Logger:
    """Set up and return a logger with file + console handlers."""

    logs_dir = ensure_path_safe(_resolve_logs_dir())
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = ensure_path_safe(logs_dir / "application.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger
