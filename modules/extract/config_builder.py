"""Build per-run effective configs from base YAML config plus CLI overrides.

Extracted from ``main/process_text_files.py``. These helpers compose the
runtime configuration for a single extraction run: they deep-copy the loaded
YAML and overlay any CLI-argument overrides (``--model``,
``--max-output-tokens``, ``--reasoning-effort``, ``--verbosity``,
``--temperature``, ``--top-p``, ``--chunk-size``, ``--concurrency-limit``,
``--delay``, ``--output``).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def build_effective_model_config(
    model_config: dict[str, Any], args: Any
) -> dict[str, Any]:
    """Build a per-run model config with CLI overrides applied."""
    effective_model_config = deepcopy(model_config or {})
    extraction_model = effective_model_config.setdefault("extraction_model", {})

    if getattr(args, "model", None):
        extraction_model["name"] = args.model

    if getattr(args, "max_output_tokens", None) is not None:
        extraction_model["max_output_tokens"] = int(args.max_output_tokens)

    if getattr(args, "reasoning_effort", None):
        reasoning = dict(extraction_model.get("reasoning", {}) or {})
        reasoning["effort"] = args.reasoning_effort
        extraction_model["reasoning"] = reasoning

    if getattr(args, "verbosity", None):
        text_config = dict(extraction_model.get("text", {}) or {})
        text_config["verbosity"] = args.verbosity
        extraction_model["text"] = text_config

    if getattr(args, "temperature", None) is not None:
        extraction_model["temperature"] = float(args.temperature)

    if getattr(args, "top_p", None) is not None:
        extraction_model["top_p"] = float(args.top_p)

    return effective_model_config


def build_effective_paths_config(
    paths_config: dict[str, Any], args: Any
) -> dict[str, Any]:
    """Build a per-run paths config, honoring CLI output overrides."""
    effective_paths_config = deepcopy(paths_config or {})
    if getattr(args, "output", None):
        general = dict(effective_paths_config.get("general", {}) or {})
        general["input_paths_is_output_path"] = False
        effective_paths_config["general"] = general
    return effective_paths_config


def build_effective_chunking_config(
    chunking_and_context_config: dict[str, Any], args: Any
) -> dict[str, Any]:
    """Build a per-run chunking config with optional CLI chunk-size override."""
    effective_chunking_config = {
        "chunking": dict(
            (chunking_and_context_config or {}).get("chunking", {}) or {}
        )
    }
    if getattr(args, "chunk_size", None) is not None:
        effective_chunking_config["chunking"]["default_tokens_per_chunk"] = int(
            args.chunk_size
        )
    return effective_chunking_config


def build_effective_concurrency_config(
    concurrency_config: dict[str, Any], args: Any
) -> dict[str, Any]:
    """Build a per-run concurrency config with CLI overrides applied."""
    effective = deepcopy(concurrency_config or {})
    extraction = effective.setdefault("concurrency", {}).setdefault(
        "extraction", {}
    )

    if getattr(args, "concurrency_limit", None) is not None:
        extraction["concurrency_limit"] = int(args.concurrency_limit)

    if getattr(args, "delay", None) is not None:
        extraction["delay_between_tasks"] = float(args.delay)

    return effective


# Backwards-compatibility aliases: existing callers in main/process_text_files.py
# use the underscore-prefixed names. Once that script is updated to use the
# public names, these can be removed.
_build_effective_model_config = build_effective_model_config
_build_effective_paths_config = build_effective_paths_config
_build_effective_chunking_config = build_effective_chunking_config
_build_effective_concurrency_config = build_effective_concurrency_config
