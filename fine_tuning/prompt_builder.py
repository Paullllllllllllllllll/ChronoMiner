from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from modules.core.prompt_context import load_basic_context
from modules.llm.prompt_utils import load_prompt_template, render_prompt_with_schema


def build_system_prompt(
    *,
    schema_name: str,
    schema_obj: Dict[str, Any],
    prompt_path: Path,
    inject_schema: bool = True,
    additional_context: Optional[str] = None,
) -> str:
    prompt_template = load_prompt_template(prompt_path)
    basic_context = load_basic_context(schema_name=schema_name)
    return render_prompt_with_schema(
        prompt_template,
        schema_obj,
        schema_name=schema_name,
        inject_schema=inject_schema,
        additional_context=additional_context,
        basic_context=basic_context,
    )
