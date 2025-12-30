from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from modules.core.context_resolver import resolve_context_for_extraction
from modules.llm.prompt_utils import load_prompt_template, render_prompt_with_schema


def build_system_prompt(
    *,
    schema_name: str,
    schema_obj: Dict[str, Any],
    prompt_path: Path,
    inject_schema: bool = True,
    text_file: Optional[Path] = None,
) -> str:
    """Build a complete system prompt with schema and context.
    
    Parameters
    ----------
    schema_name : str
        Name of the schema for context resolution
    schema_obj : Dict[str, Any]
        The JSON schema to inject
    prompt_path : Path
        Path to the prompt template file
    inject_schema : bool
        Whether to inject the schema into the prompt
    text_file : Optional[Path]
        Path to input file for file-specific context resolution
        
    Returns
    -------
    str
        The rendered system prompt with schema and context
    """
    prompt_template = load_prompt_template(prompt_path)
    context, _ = resolve_context_for_extraction(schema_name=schema_name, text_file=text_file)
    return render_prompt_with_schema(
        prompt_template,
        schema_obj,
        schema_name=schema_name,
        inject_schema=inject_schema,
        context=context,
    )
