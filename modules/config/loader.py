# modules/config/loader.py

"""Configuration loading and validation for ChronoMiner."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Module-level cache for loaded configurations
_config_cache: Optional["ConfigLoader"] = None


class ConfigLoader:
    """
    Loads and validates YAML configuration files for ChronoMiner.

    Manages:
      - paths_config.yaml
      - model_config.yaml
      - chunking_and_context.yaml
      - concurrency_config.yaml
    """
    _REQUIRED_KEYS: Dict[str, list] = {
        "paths_config.yaml": ["general", "schemas_paths"],
        "model_config.yaml": ["extraction_model"],
        "concurrency_config.yaml": ["concurrency"],
        "chunking_and_context.yaml": ["chunking"],
    }

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        self.config_dir = config_dir or Path(__file__).resolve().parents[2] / 'config'
        self.paths_config: Optional[Dict[str, Any]] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.concurrency_config: Optional[Dict[str, Any]] = None
        self.chunking_and_context_config: Optional[Dict[str, Any]] = None
        self._image_processing_config: Optional[Dict[str, Any]] = None

    def load_configs(self) -> None:
        """
        Load and validate the configuration files.
        """
        self.paths_config = self._load_yaml('paths_config.yaml')
        self.model_config = self._load_yaml('model_config.yaml')
        self.concurrency_config = self._load_yaml('concurrency_config.yaml')
        self.chunking_and_context_config = self._load_yaml(
            'chunking_and_context.yaml')

        configs = {
            "paths_config.yaml": self.paths_config,
            "model_config.yaml": self.model_config,
            "concurrency_config.yaml": self.concurrency_config,
            "chunking_and_context.yaml": self.chunking_and_context_config,
        }
        for filename, config in configs.items():
            self._validate_config(
                config, self._REQUIRED_KEYS[filename], filename)

        self._resolve_paths(self.paths_config)

        # Validate image support if expects_image_inputs is set
        tm = self.model_config.get("extraction_model", {})
        if tm.get("expects_image_inputs", False):
            self._ensure_image_support(tm.get("name", ""), True)

        logger.info("All configurations loaded and validated successfully.")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load a YAML file and return its content as a dictionary.

        :param filename: The name of the YAML file.
        :return: Parsed YAML content.
        :raises FileNotFoundError: If the file is missing.
        """
        config_path = self.config_dir / filename
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Missing configuration file: {config_path}")
        
        with config_path.open('r', encoding='utf-8') as f:
            content = f.read()
            try:
                # Normalize Windows paths in paths_config.yaml
                if filename == 'paths_config.yaml':
                    content = re.sub(
                        r'"([^"]*)"',
                        lambda m: '"' + m.group(1).replace('\\', '/') + '"',
                        content
                    )
                return yaml.safe_load(content)
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {filename}: {e}")
                raise

    def _validate_config(self, config: Dict[str, Any], required_keys: list, config_name: str) -> None:
        """
        Validate configuration for required keys.

        :param config: The configuration dictionary.
        :param required_keys: List of required keys.
        :param config_name: Name of the configuration file.
        :raises KeyError: If a required key is missing.
        """
        for key in required_keys:
            if key not in config:
                error_msg = f"Missing '{key}' in {config_name}"
                logger.error(error_msg)
                raise KeyError(error_msg)

    def _resolve_paths(self, config: Dict[str, Any]) -> None:
        """
        Resolve relative paths in configuration if enabled.

        :param config: The paths configuration dictionary.
        """
        general = config.get("general", {})
        allow_relative_paths = general.get("allow_relative_paths", False)

        if not allow_relative_paths:
            return  # Skip resolution if relative paths not allowed

        # Determine base directory for relative paths
        base_directory = general.get("base_directory", ".")
        base_path = Path(base_directory).resolve()

        if not base_path.exists():
            logger.warning(
                f"Base directory '{base_directory}' does not exist. Using current directory.")
            base_path = Path.cwd()

        # Resolve logs_dir if it's a relative path
        if "logs_dir" in general and not Path(
                general["logs_dir"]).is_absolute():
            general["logs_dir"] = str(
                (base_path / general["logs_dir"]).resolve())
            logger.info(f"Resolved logs_dir to: {general['logs_dir']}")

        # Resolve schema paths
        schemas_paths = config.get("schemas_paths", {})
        for schema, schema_config in schemas_paths.items():
            for path_key in ["input", "output"]:
                if path_key in schema_config and not Path(
                        schema_config[path_key]).is_absolute():
                    schema_config[path_key] = str(
                        (base_path / schema_config[path_key]).resolve())
                    logger.info(
                        f"Resolved {schema}.{path_key} to: {schema_config[path_key]}")

    def _ensure_image_support(self, model_name: str, expects_images: bool) -> None:
        """
        Validate that the model supports image inputs if expects_image_inputs is True.
        
        :param model_name: The model identifier.
        :param expects_images: Whether image inputs are expected.
        :raises ValueError: If model doesn't support images but expects_image_inputs is True.
        """
        if not expects_images:
            return
        
        try:
            from modules.llm.model_capabilities import detect_capabilities
            caps = detect_capabilities(model_name)
            
            if not caps.supports_image_input:
                error_msg = (
                    f"Model '{model_name}' does not support image inputs, "
                    f"but expects_image_inputs is set to True in model_config.yaml. "
                    f"Either change the model or set expects_image_inputs: false."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.debug(f"Model '{model_name}' supports image inputs (validated)")
            
        except ImportError:
            # If model_capabilities is not available, skip validation
            logger.warning("Could not validate image support - model_capabilities not available")

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

    def get_chunking_and_context_config(self) -> Dict[str, Any]:
        """
        Get the loaded chunking and context configuration.

        :return: The chunking and context configuration dictionary.
        """
        return self.chunking_and_context_config  # type: ignore

    def get_image_processing_config(self) -> Dict[str, Any]:
        """
        Get the image processing configuration.

        Loaded lazily from image_processing_config.yaml. Returns an empty
        dict if the file does not exist (image pipeline is optional).

        :return: The image processing configuration dictionary.
        """
        if self._image_processing_config is None:
            path = self.config_dir / "image_processing_config.yaml"
            if path.exists():
                self._image_processing_config = self._load_yaml("image_processing_config.yaml")
            else:
                self._image_processing_config = {}
        return self._image_processing_config

    def get_schemas_paths(self) -> Dict[str, Any]:
        """
        Get the schema-specific paths from the configuration.

        :return: A dictionary mapping schema names to their paths.
        """
        return self.paths_config.get("schemas_paths", {})  # type: ignore


def get_config_loader(force_reload: bool = False) -> ConfigLoader:
    """
    Get a cached ConfigLoader instance.
    
    Uses module-level caching to avoid redundant config file reads.
    This is the recommended way to access configuration throughout
    the application.
    
    :param force_reload: If True, reload configs even if cached
    :return: Configured ConfigLoader instance
    
    Example::
    
        from modules.config.loader import get_config_loader
        
        config = get_config_loader()
        model_config = config.get_model_config()
    """
    global _config_cache
    
    if _config_cache is None or force_reload:
        _config_cache = ConfigLoader()
        _config_cache.load_configs()
        logger.debug("Configuration loaded and cached")
    
    return _config_cache


def clear_config_cache() -> None:
    """
    Clear the configuration cache.
    
    Use this when configuration files have been modified and need
    to be reloaded.
    """
    global _config_cache
    _config_cache = None
    logger.debug("Configuration cache cleared")
