# modules/context_manager.py

import logging
from pathlib import Path
from typing import Dict, Optional

from modules.core.path_utils import ensure_path_safe

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages loading and retrieval of additional context for schema-based
    processing.

    The ContextManager loads context files from a specified directory and
    provides access to schema-specific additional context that can be
    injected into prompts during data extraction.
    """

    def __init__(
        self, additional_context_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize the ContextManager.

        Parameters
        ----------
        additional_context_dir : Optional[Path]
            Directory containing additional context files. If None, defaults
            to the 'additional_context' directory relative to this module.
        """
        if additional_context_dir is None:
            self.additional_context_dir: Path = (
                Path(__file__).resolve().parent.parent.parent
                / "additional_context"
            )
        else:
            self.additional_context_dir = additional_context_dir
        self.additional_context: Dict[str, str] = {}

    def load_additional_context(self) -> None:
        """
        Load all additional context files from the additional_context
        directory.

        Each .txt file in the directory is loaded, with the filename
        (without extension) serving as the schema name key.
        """
        safe_context_dir = ensure_path_safe(self.additional_context_dir)
        if not safe_context_dir.exists():
            logger.warning(
                "Additional context directory not found: %s. "
                "Creating directory.",
                self.additional_context_dir
            )
            safe_context_dir.mkdir(parents=True, exist_ok=True)
            return

        for context_file in safe_context_dir.glob("*.txt"):
            try:
                safe_context_file = ensure_path_safe(context_file)
                with safe_context_file.open("r", encoding="utf-8") as f:
                    content: str = f.read().strip()
                schema_name: str = context_file.stem
                self.additional_context[schema_name] = content
                logger.info(
                    "Loaded additional context for schema '%s' from %s",
                    schema_name,
                    context_file.name
                )
            except (OSError, UnicodeDecodeError) as e:
                logger.error(
                    "Error loading additional context from %s: %s",
                    context_file,
                    e
                )

    def get_additional_context(self, schema_name: str) -> Optional[str]:
        """
        Get the additional context for a specific schema.

        Parameters
        ----------
        schema_name : str
            The name of the schema.

        Returns
        -------
        Optional[str]
            The additional context if available, else None.
        """
        context = self.additional_context.get(schema_name)
        if context is None:
            logger.debug(
                "No additional context available for schema '%s'. "
                "Available schemas: %s",
                schema_name,
                list(self.additional_context.keys())
            )
        return context