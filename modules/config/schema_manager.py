"""Schema and developer-message loader for ChronoMiner.

Loads JSON schemas from ``schemas/`` and developer messages from
``developer_messages/`` (both anchored at the project root). Consumers
should retrieve instances via :func:`modules.config.get_config_loader`-
style patterns or construct :class:`SchemaManager` directly for tests.
"""

import json
import logging
from pathlib import Path

from modules.infra.paths import ensure_path_safe

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Manages loading and retrieval of JSON schemas and developer messages.
    """

    def __init__(
        self,
        schemas_dir: Path | None = None,
        dev_messages_dir: Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        if schemas_dir is None:
            self.schemas_dir: Path = project_root / "schemas"
        else:
            self.schemas_dir = Path(schemas_dir).resolve()
        if dev_messages_dir is None:
            self.dev_messages_dir: Path = project_root / "developer_messages"
        else:
            self.dev_messages_dir = Path(dev_messages_dir).resolve()
        self.schemas: dict[str, dict] = {}
        self.schema_paths: dict[str, Path] = {}
        self.dev_messages: dict[str, str] = {}

    def load_schemas(self) -> None:
        """
        Load all JSON schemas from the schemas directory.
        """
        for schema_file in self.schemas_dir.glob("*.json"):
            try:
                safe_schema_file = ensure_path_safe(schema_file)
                with safe_schema_file.open("r", encoding="utf-8") as f:
                    schema_content: dict = json.load(f)
                schema_name: str | None = schema_content.get("name")
                if schema_name:
                    self.schemas[schema_name] = schema_content
                    self.schema_paths[schema_name] = schema_file.resolve()
                    logger.info(
                        f"Loaded schema '{schema_name}' from {schema_file.name}"
                    )
                else:
                    logger.warning(
                        f"Schema file {schema_file.name} has no 'name' field."
                    )
            except json.JSONDecodeError as e:
                logger.error(
                    f"Invalid JSON in schema {schema_file.name} "
                    f"at line {e.lineno}, column {e.colno}: {e.msg}"
                )
            except (OSError, UnicodeDecodeError) as e:
                logger.error(f"Error loading schema from {schema_file}: {e}")

    def load_dev_messages(self) -> None:
        """
        Load developer messages from text files in the developer_messages directory.
        """
        # The directory is optional; absent it there is nothing to load. Guard
        # before iterdir(), which (unlike glob) raises FileNotFoundError.
        if not self.dev_messages_dir.is_dir():
            return
        # Load top-level message files.
        for message_file in self.dev_messages_dir.glob("*.txt"):
            schema_name = message_file.stem
            try:
                safe_message_file = ensure_path_safe(message_file)
                with safe_message_file.open("r", encoding="utf-8") as f:
                    content: str = f.read().strip()
                self.dev_messages[schema_name] = content
                logger.info(
                    f"Loaded developer message for schema '{schema_name}' "
                    f"from {message_file.name}"
                )
            except (OSError, UnicodeDecodeError) as e:
                logger.error(
                    f"Error loading developer message from {message_file}: {e}"
                )
        # Load aggregated messages from subdirectories.
        for subdir in self.dev_messages_dir.iterdir():
            if subdir.is_dir():
                schema_name = subdir.name
                parts: list[str] = []
                for file in sorted(subdir.glob("*.txt")):
                    try:
                        safe_file = ensure_path_safe(file)
                        with safe_file.open("r", encoding="utf-8") as f:
                            parts.append(f.read().strip())
                    except (OSError, UnicodeDecodeError) as e:
                        logger.error(f"Error reading {file}: {e}")
                if parts:
                    self.dev_messages[schema_name] = "\n".join(parts)
                    logger.info(
                        f"Loaded aggregated developer message for schema "
                        f"'{schema_name}' from folder {subdir.name}"
                    )

    def get_available_schemas(self) -> dict[str, dict]:
        """
        Retrieve the loaded schemas.

        :return: Dictionary mapping schema names to schema content.
        """
        return self.schemas

    def list_schema_options(self) -> list[tuple[str, Path]]:
        """Return schema names paired with their source file paths."""
        return sorted(
            [
                (name, self.schema_paths[name])
                for name in self.schemas
                if name in self.schema_paths
            ],
            key=lambda item: item[0].lower(),
        )

    def get_dev_message(self, schema_name: str) -> str | None:
        """
        Get the developer message for a specific schema.

        :param schema_name: The name of the schema.
        :return: The developer message if available, else None.
        """
        return self.dev_messages.get(schema_name)
