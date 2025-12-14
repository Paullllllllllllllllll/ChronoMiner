from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.config.loader import get_config_loader
from modules.llm.langchain_provider import ProviderConfig
from modules.llm.openai_utils import open_extractor, process_text_chunk


async def extract_chunks(
    *,
    chunk_texts: List[str],
    system_prompt: str,
    schema_definition: Optional[Dict[str, Any]],
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    config = get_config_loader()
    model_cfg = config.get_model_config() or {}

    effective_model = model_name or (model_cfg.get("transcription_model", {}) or {}).get("name")
    if not isinstance(effective_model, str) or not effective_model.strip():
        raise ValueError("Model name not found (set config/model_config.yaml transcription_model.name)")
    effective_model = effective_model.strip()

    provider = ProviderConfig._detect_provider(effective_model)
    api_key = ProviderConfig._get_api_key(provider)
    if not api_key:
        raise ValueError(f"API key not found for provider {provider}")

    results: List[Dict[str, Any]] = []

    async with open_extractor(
        api_key=api_key,
        prompt_path=Path("prompts/structured_output_prompt.txt"),
        model=effective_model,
        provider=provider,
    ) as extractor:
        for text in chunk_texts:
            raw = await process_text_chunk(
                text_chunk=text,
                extractor=extractor,
                system_message=system_prompt,
                json_schema=schema_definition,
            )
            output_text = raw.get("output_text") or ""
            parsed: Dict[str, Any]
            try:
                obj = json.loads(output_text) if isinstance(output_text, str) else output_text
                parsed = obj if isinstance(obj, dict) else {"raw_model_output": output_text}
            except Exception:
                parsed = {"raw_model_output": output_text}

            results.append(
                {
                    "output": parsed,
                    "raw": raw,
                }
            )

    return results


def extract_chunks_sync(
    *,
    chunk_texts: List[str],
    system_prompt: str,
    schema_definition: Optional[Dict[str, Any]],
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return asyncio.run(
        extract_chunks(
            chunk_texts=chunk_texts,
            system_prompt=system_prompt,
            schema_definition=schema_definition,
            model_name=model_name,
        )
    )
