# modules/config/manager.py

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import sys

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigManager:
    """
    Handles configuration validation, loading, and preparation.
    Provides unified error messaging and validation logic.
    """

    def __init__(self, config_loader: Any) -> None:
        """Initialize with a ConfigLoader instance"""
        self.config_loader = config_loader
        self._validation_errors: list[str] = []

    def validate_paths(self, paths_config: Dict[str, Any], raise_on_error: bool = True) -> bool:
        """
        Validate path configurations based on the allow_relative_paths setting.

        If allow_relative_paths is enabled, paths should already be resolved.
        Otherwise, verify that all paths are absolute.

        :param paths_config: The loaded paths configuration
        :param raise_on_error: If True, raises ConfigValidationError; if False, returns False
        :return: True if validation passes, False otherwise
        :raises ConfigValidationError: If validation fails and raise_on_error is True
        """
        self._validation_errors.clear()
        general = paths_config.get("general", {})
        allow_relative_paths = general.get("allow_relative_paths", False)

        # Skip validation if using relative paths - they should have been resolved by ConfigLoader
        if allow_relative_paths:
            return True

        # Validate general logs_dir
        logs_dir = general.get("logs_dir")
        if logs_dir and not Path(logs_dir).is_absolute():
            self._add_error(
                f"The 'logs_dir' path '{logs_dir}' is not absolute. "
                f"Please use an absolute path or enable allow_relative_paths in paths_config.yaml."
            )

        # Validate each schema's input and output paths
        schemas_paths = paths_config.get("schemas_paths", {})
        for schema, schema_config in schemas_paths.items():
            input_path = schema_config.get("input")
            output_path = schema_config.get("output")
            if input_path and not Path(input_path).is_absolute():
                self._add_error(
                    f"The input path for schema '{schema}' ('{input_path}') is not absolute. "
                    f"Please use absolute paths or enable allow_relative_paths in paths_config.yaml."
                )
            if output_path and not Path(output_path).is_absolute():
                self._add_error(
                    f"The output path for schema '{schema}' ('{output_path}') is not absolute. "
                    f"Please use absolute paths or enable allow_relative_paths in paths_config.yaml."
                )

        if self._validation_errors:
            if raise_on_error:
                error_msg = "\n".join(self._validation_errors)
                raise ConfigValidationError(f"Path validation failed:\n{error_msg}")
            return False
        return True

    def _add_error(self, message: str) -> None:
        """Add a validation error message."""
        self._validation_errors.append(message)
        logger.error(message)

    def get_validation_errors(self) -> list[str]:
        """Return list of validation errors."""
        return self._validation_errors.copy()

    def load_developer_message(self, schema_name: str, raise_on_error: bool = True) -> Optional[str]:
        """
        Load the developer message corresponding to the given schema.

        :param schema_name: The name of the extraction schema
        :param raise_on_error: If True, raises FileNotFoundError; if False, returns None
        :return: The contents of the corresponding developer message file, or None if not found
        :raises FileNotFoundError: If the file cannot be read and raise_on_error is True
        """
        developer_messages_dir = Path("developer_messages")
        file_name = f"{schema_name}.txt"
        file_path = developer_messages_dir / file_name

        try:
            if file_path.exists():
                with file_path.open("r", encoding="utf-8") as f:
                    return f.read()
            else:
                error_msg = f"Developer message file '{file_name}' not found in {developer_messages_dir}."
                logger.error(error_msg)
                if raise_on_error:
                    raise FileNotFoundError(error_msg)
                return None
        except FileNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to read developer message file: {e}"
            logger.error(error_msg)
            if raise_on_error:
                raise IOError(error_msg) from e
            return None

    def get_schemas_paths(self) -> Dict[str, Any]:
        """
        Get the schema-specific paths from the configuration.

        :return: Dictionary of schema paths
        """
        return self.config_loader.get_schemas_paths()