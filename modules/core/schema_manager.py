# modules/schema_manager.py

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Manages loading and retrieval of JSON schemas and developer messages.
    """
    def __init__(self, schemas_dir: Optional[Path] = None, dev_messages_dir: Optional[Path] = None) -> None:
        if schemas_dir is None:
            self.schemas_dir: Path = Path(__file__).resolve().parent.parent / "schemas"
        else:
            self.schemas_dir = schemas_dir
        if dev_messages_dir is None:
            self.dev_messages_dir: Path = Path(__file__).resolve().parent.parent / "developer_messages"
        else:
            self.dev_messages_dir = dev_messages_dir
        self.schemas: Dict[str, dict] = {}
        self.schema_paths: Dict[str, Path] = {}
        self.dev_messages: Dict[str, str] = {}

    def load_schemas(self) -> None:
        """
        Load all JSON schemas from the schemas directory.
        """
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                with schema_file.open("r", encoding="utf-8") as f:
                    schema_content: dict = json.load(f)
                schema_name: Optional[str] = schema_content.get("name")
                if schema_name:
                    self.schemas[schema_name] = schema_content
                    self.schema_paths[schema_name] = schema_file.resolve()
                    logger.info(f"Loaded schema '{schema_name}' from {schema_file.name}")
                else:
                    logger.warning(f"Schema file {schema_file.name} does not contain a 'name' field.")
            except Exception as e:
                logger.error(f"Error loading schema from {schema_file}: {e}")

    def load_dev_messages(self) -> None:
        """
        Load developer messages from text files in the developer_messages directory.
        """
        # Load top-level message files.
        for message_file in self.dev_messages_dir.glob("*.txt"):
            schema_name = message_file.stem
            try:
                with message_file.open("r", encoding="utf-8") as f:
                    content: str = f.read().strip()
                self.dev_messages[schema_name] = content
                logger.info(f"Loaded developer message for schema '{schema_name}' from {message_file.name}")
            except Exception as e:
                logger.error(f"Error loading developer message from {message_file}: {e}")
        # Load aggregated messages from subdirectories.
        for subdir in self.dev_messages_dir.iterdir():
            if subdir.is_dir():
                schema_name: str = subdir.name
                parts: list[str] = []
                for file in sorted(subdir.glob("*.txt")):
                    try:
                        with file.open("r", encoding="utf-8") as f:
                            parts.append(f.read().strip())
                    except Exception as e:
                        logger.error(f"Error reading {file}: {e}")
                if parts:
                    self.dev_messages[schema_name] = "\n".join(parts)
                    logger.info(f"Loaded aggregated developer message for schema '{schema_name}' from folder {subdir.name}")

    def get_available_schemas(self) -> Dict[str, dict]:
        """
        Retrieve the loaded schemas.

        :return: Dictionary mapping schema names to schema content.
        """
        return self.schemas

    def list_schema_options(self) -> List[Tuple[str, Path]]:
        """Return schema names paired with their source file paths."""
        return sorted(
            [(name, self.schema_paths[name]) for name in self.schemas if name in self.schema_paths],
            key=lambda item: item[0].lower(),
        )

    def get_dev_message(self, schema_name: str) -> Optional[str]:
        """
        Get the developer message for a specific schema.

        :param schema_name: The name of the schema.
        :return: The developer message if available, else None.
        """
        return self.dev_messages.get(schema_name)
