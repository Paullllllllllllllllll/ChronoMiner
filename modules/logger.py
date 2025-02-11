# modules/logger.py

import logging
from pathlib import Path
from modules.config_loader import ConfigLoader
from typing import Any

def setup_logger(name: str) -> logging.Logger:
    """
    Set up and return a logger configured with a file handler.

    :param name: The name of the logger.
    :return: A configured logger instance.
    """
    config_loader: ConfigLoader = ConfigLoader()
    config_loader.load_configs()
    paths_config: Any = config_loader.get_paths_config()
    logs_dir: Path = Path(paths_config.get('general', {}).get('logs_dir', 'logs'))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file: Path = logs_dir / "application.log"
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
