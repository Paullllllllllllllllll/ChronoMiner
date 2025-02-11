# modules/config_loader.py

import yaml
from pathlib import Path
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    A class for loading and validating YAML configuration files.

    It loads:
      - paths_config.yaml
      - model_config.yaml
      - concurrency_config.yaml
    """
    REQUIRED_PATHS_KEYS = ['general', 'schemas_paths']
    REQUIRED_MODEL_CONFIG = ['extraction_model']
    REQUIRED_CONCURRENCY = ['concurrency']

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        if config_dir is None:
            self.config_dir: Path = Path(__file__).resolve().parent.parent / 'config'
        else:
            self.config_dir = config_dir
        self.paths_config: Optional[Dict[str, Any]] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.concurrency_config: Optional[Dict[str, Any]] = None

    def load_configs(self) -> None:
        """
        Load and validate the configuration files.
        """
        self.paths_config = self._load_yaml('paths_config.yaml')
        self.model_config = self._load_yaml('model_config.yaml')
        self.concurrency_config = self._load_yaml('concurrency_config.yaml')

        self._validate_paths_config(self.paths_config)
        self._validate_model_config(self.model_config)
        self._validate_concurrency_config(self.concurrency_config)
        logger.info("All configurations loaded and validated successfully.")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML file and return its content as a dictionary.

        :param filename: The name of the YAML file.
        :return: Parsed YAML content.
        :raises FileNotFoundError: If the file is missing.
        """
        config_path: Path = self.config_dir / filename
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Missing configuration file: {config_path}")
        with config_path.open('r', encoding='utf-8') as f:
            content = f.read().replace("\\", "/")
            try:
                return yaml.safe_load(content)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {filename}: {e}")
                raise

    def _validate_paths_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the paths configuration for required keys.

        :param config: The paths configuration dictionary.
        :raises KeyError: If a required key is missing.
        """
        for key in self.REQUIRED_PATHS_KEYS:
            if key not in config:
                logger.error(f"Missing '{key}' in paths_config.yaml")
                raise KeyError(f"'{key}' is required in paths_config.yaml")

    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the model configuration for required keys.

        :param config: The model configuration dictionary.
        :raises KeyError: If a required key is missing.
        """
        for key in self.REQUIRED_MODEL_CONFIG:
            if key not in config:
                logger.error(f"Missing '{key}' in model_config.yaml")
                raise KeyError(f"'{key}' is required in model_config.yaml")

    def _validate_concurrency_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the concurrency configuration for required keys.

        :param config: The concurrency configuration dictionary.
        :raises KeyError: If a required key is missing.
        """
        for key in self.REQUIRED_CONCURRENCY:
            if key not in config:
                logger.error(f"Missing '{key}' in concurrency_config.yaml")
                raise KeyError(f"'{key}' is required in concurrency_config.yaml")

    def get_paths_config(self) -> Dict[str, Any]:
        """
        Get the loaded paths configuration.

        :return: The paths configuration dictionary.
        """
        return self.paths_config  # type: ignore

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the loaded model configuration.

        :return: The model configuration dictionary.
        """
        return self.model_config  # type: ignore

    def get_concurrency_config(self) -> Dict[str, Any]:
        """
        Get the loaded concurrency configuration.

        :return: The concurrency configuration dictionary.
        """
        return self.concurrency_config  # type: ignore

    def get_schemas_paths(self) -> Dict[str, Any]:
        """
        Get the schema-specific paths from the configuration.

        :return: A dictionary mapping schema names to their paths.
        """
        return self.paths_config.get("schemas_paths", {})  # type: ignore
