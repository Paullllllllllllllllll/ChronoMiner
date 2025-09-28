# modules/config_manager.py

from pathlib import Path
from typing import Dict, Any, List
import logging
import sys

logger = logging.getLogger(__name__)


class ConfigManager:
	"""
	Handles configuration validation, loading, and preparation.
	"""

	def __init__(self, config_loader):
		"""Initialize with a ConfigLoader instance"""
		self.config_loader = config_loader

	def validate_paths(self, paths_config: Dict[str, Any]) -> None:
		"""
		Validate path configurations based on the allow_relative_paths setting.

		If allow_relative_paths is enabled, paths should already be resolved.
		Otherwise, verify that all paths are absolute.

		:param paths_config: The loaded paths configuration
		:raises: SystemExit if validation fails
		"""
		general = paths_config.get("general", {})
		allow_relative_paths = general.get("allow_relative_paths", False)

		# Skip validation if using relative paths - they should have been resolved by ConfigLoader
		if allow_relative_paths:
			return

		error_found = False

		# Validate general logs_dir
		logs_dir = general.get("logs_dir")
		if logs_dir and not Path(logs_dir).is_absolute():
			print(f"[ERROR] The 'logs_dir' path '{logs_dir}' is not absolute. "
			      f"Please use an absolute path or enable allow_relative_paths in paths_config.yaml.")
			error_found = True

		# Validate each schema's input and output paths
		schemas_paths = paths_config.get("schemas_paths", {})
		for schema, schema_config in schemas_paths.items():
			input_path = schema_config.get("input")
			output_path = schema_config.get("output")
			if input_path and not Path(input_path).is_absolute():
				print(
					f"[ERROR] The input path for schema '{schema}' ('{input_path}') is not absolute. "
					f"Please use absolute paths or enable allow_relative_paths in paths_config.yaml.")
				error_found = True
			if output_path and not Path(output_path).is_absolute():
				print(
					f"[ERROR] The output path for schema '{schema}' ('{output_path}') is not absolute. "
					f"Please use absolute paths or enable allow_relative_paths in paths_config.yaml.")
				error_found = True

		if error_found:
			sys.exit(1)

	def load_developer_message(self, schema_name: str) -> str:
		"""
		Load the developer message corresponding to the given schema.

		:param schema_name: The name of the extraction schema
		:return: The contents of the corresponding developer message file
		:raises: SystemExit if the file cannot be read
		"""
		developer_messages_dir = Path("developer_messages")
		file_name = f"{schema_name}.txt"
		file_path = developer_messages_dir / file_name

		try:
			if file_path.exists():
				with file_path.open("r", encoding="utf-8") as f:
					return f.read()
			else:
				print(
					f"[ERROR] Developer message file '{file_name}' not found in {developer_messages_dir}.")
				logger.error(f"Developer message file '{file_name}' not found.")
				sys.exit(1)
		except Exception as e:
			print(f"[ERROR] Failed to read developer message file: {e}")
			logger.error(f"Failed to read developer message file: {e}")
			sys.exit(1)

	def get_schemas_paths(self) -> Dict[str, Any]:
		"""
		Get the schema-specific paths from the configuration.

		:return: Dictionary of schema paths
		"""
		return self.config_loader.get_schemas_paths()


# Backward compatibility functions
def validate_paths(paths_config: Dict[str, Any]) -> None:
	"""
	Legacy function maintained for backward compatibility.
	"""
	config_manager = ConfigManager(None)
	return config_manager.validate_paths(paths_config)


def load_developer_message(schema_name: str) -> str:
	"""
	Legacy function maintained for backward compatibility.
	"""
	config_manager = ConfigManager(None)
	return config_manager.load_developer_message(schema_name)