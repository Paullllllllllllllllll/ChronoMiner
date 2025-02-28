# modules/context_manager.py

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages loading and retrieval of additional context for schema-based processing.
    """
    def __init__(self, additional_context_dir: Optional[Path] = None) -> None:
        if additional_context_dir is None:
            self.additional_context_dir: Path = Path(__file__).resolve().parent.parent / "additional_context"
        else:
            self.additional_context_dir = additional_context_dir
        self.additional_context: Dict[str, str] = {}

    def load_additional_context(self) -> None:
        """
        Load all additional context files from the additional_context directory.
        """
        if not self.additional_context_dir.exists():
            logger.warning(f"Additional context directory not found: {self.additional_context_dir}. Creating directory.")
            self.additional_context_dir.mkdir(parents=True, exist_ok=True)
            return

        for context_file in self.additional_context_dir.glob("*.txt"):
            try:
                with context_file.open("r", encoding="utf-8") as f:
                    content: str = f.read().strip()
                schema_name: str = context_file.stem
                self.additional_context[schema_name] = content
                logger.info(f"Loaded additional context for schema '{schema_name}' from {context_file.name}")
            except Exception as e:
                logger.error(f"Error loading additional context from {context_file}: {e}")

    def get_additional_context(self, schema_name: str) -> Optional[str]:
        """
        Get the additional context for a specific schema.

        :param schema_name: The name of the schema.
        :return: The additional context if available, else None.
        """
        return self.additional_context.get(schema_name)