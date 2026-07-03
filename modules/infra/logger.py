"""Centralized logger configuration.

Resolves ``logs_dir`` from ``paths_config.yaml`` (falling back to
``PROJECT_ROOT/logs``) and attaches a file handler plus a WARNING-level
console handler. Note: any module that calls ``setup_logger`` at import
time triggers config loading; see :mod:`modules.config.loader`.

Handler placement (CM-9): handlers are attached to TOP-LEVEL loggers only
(``modules``, ``main``, and the top-level ancestor of whatever name is
passed to ``setup_logger``), never to dotted child loggers. Children reach
the handlers via standard propagation. This makes records from modules that
use plain ``logging.getLogger(__name__)`` (e.g. ``modules.config.context``,
``modules.line_ranges.readjuster``) land in ``application.log`` exactly like
``setup_logger`` users, while guaranteeing each record is emitted once.
"""

import logging
from pathlib import Path

from modules.config.loader import get_config_loader
from modules.infra.paths import ensure_path_safe

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Top-level logger namespaces that must always be configured so that any
# module logging under them is visible, regardless of import order.
_BASE_NAMESPACES = ("modules", "main")


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


def _configure_base_logger(base: logging.Logger, log_file: Path) -> None:
    """Attach the shared file + console handlers to a top-level logger once."""
    base.setLevel(logging.INFO)
    if base.handlers:
        return

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    base.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    base.addHandler(console_handler)


def setup_logger(name: str) -> logging.Logger:
    """Set up and return a logger whose records reach file + console handlers.

    Handlers live on the top-level ancestor of *name* (and on the ``modules``
    and ``main`` namespaces); the returned logger relies on propagation, so a
    dotted name never carries its own handlers and records are emitted
    exactly once.
    """

    logs_dir = ensure_path_safe(_resolve_logs_dir())
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = ensure_path_safe(logs_dir / "application.log")

    # Always configure the shared package namespaces so plain
    # logging.getLogger(__name__) loggers under modules/ and main/ are
    # visible no matter which entry point ran setup_logger first.
    for namespace in _BASE_NAMESPACES:
        _configure_base_logger(logging.getLogger(namespace), log_file)

    top_level_name = name.split(".", 1)[0]
    _configure_base_logger(logging.getLogger(top_level_name), log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
