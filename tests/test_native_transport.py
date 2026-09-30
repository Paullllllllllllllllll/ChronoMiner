"""PNG survives serialized synchronous retries and batch requests offline."""

from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

import modules.extract.processing_strategy as ps
from modules.batch.backends.anthropic_backend import AnthropicBatchBackend
from modules.batch.backends.base import BatchRequest
from modules.batch.backends.google_backend import GoogleBatchBackend
from modules.batch.backends.openai_backend import OpenAIBatchBackend
from modules.config.capabilities import detect_capabilities
from modules.images.page_stream import (
    PagePayload,
    build_image_provenance,
    stream_page_payloads,
)
from modules.llm.langchain_provider import LangChainLLM


async def png_payload(tmp_path: Path, provider: str, model: str) -> PagePayload:
    path = tmp_path / "image.png"
    Image.new("L", (32, 48), 128).save(path)
    section = "api" if provider == "openai" else provider
    payloads = [
        p
        async for p in stream_page_payloads(
            path,
            [1],
            {
                f"{section}_image_processing": {
                    "payload_format": "png",
                    "llm_detail": "original",
                }
            },
            provider,
            model,
            None,
        )
    ]
    assert isinstance(payloads[0], PagePayload)
    return payloads[0]


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "gpt-6-astra"),
        ("anthropic", "claude-opus-5"),
        ("google", "gemini-3-flash-preview"),
    ],
)
@pytest.mark.asyncio
async def test_png_sync_retry_serialization(
    tmp_path: Path, provider: str, model: str
) -> None:
    payload = await png_payload(tmp_path, provider, model)
    section = "api" if provider == "openai" else provider
    provenance = build_image_provenance(
        tmp_path / "image.png",
        {
            f"{section}_image_processing": {
                "payload_format": "png",
                "llm_detail": "original",
            }
        },
        provider,
        model,
        None,
    )
    captured = []

    async def invoke(messages: Any) -> AIMessage:
        captured.append(json.dumps([message.model_dump() for message in messages]))
        if len(captured) == 1:
            raise RuntimeError("Synthetic HTTP 503 Service unavailable")
        return AIMessage(content="extracted")

    llm = object.__new__(LangChainLLM)
    llm.config = SimpleNamespace(provider=provider, model=model)
    llm._initialized = True
    llm._chat_model = SimpleNamespace(ainvoke=invoke)
    extractor = SimpleNamespace(
        provider=provider,
        caps=detect_capabilities(model),
        llm=llm,
    )

    @asynccontextmanager
    async def open_extractor(**kwargs: Any) -> Any:
        yield extractor

    async def source() -> Any:
        yield payload

    strategy = ps.SynchronousProcessingStrategy(
        {
            "concurrency": {
                "extraction": {
                    "retry": {
                        "attempts": 2,
                        "wait_min_seconds": 0.001,
                        "wait_max_seconds": 0.001,
                    }
                }
            },
        }
    )
    with (
        patch.object(ps, "open_extractor", open_extractor),
        patch.object(ps.ProviderConfig, "_get_api_key", return_value="fixture"),
        patch.object(ps, "await_capacity", new=AsyncMock()),
        patch.object(ps, "get_shared_rate_limiter", return_value=MagicMock()),
    ):
        results = await strategy.process_chunks(
            chunks=[""],
            handler=None,
            dev_message="Extract.",
            model_config={"extraction_model": {"provider": provider, "name": model}},
            schema={},
            file_path=tmp_path / "image.png",
            temp_jsonl_path=tmp_path / "run.jsonl",
            console_print=lambda *_: None,
            image_source=source(),
            image_provenance=provenance,
        )
    assert len(results) == 1 and "error" not in results[0]
    assert len(captured) == 2 and captured[0] == captured[1]
    assert "image/png" in captured[0] and payload.base64 in captured[0]
    assert base64.b64decode(payload.base64).startswith(b"\x89PNG")
    header = json.loads(
        (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert header["image_provenance"] == provenance
    assert (
        header["image_settings_fingerprint"] == provenance["image_settings_fingerprint"]
    )


@pytest.mark.parametrize(
    "backend_cls,provider,model",
    [
        (OpenAIBatchBackend, "openai", "gpt-6-astra"),
        (AnthropicBatchBackend, "anthropic", "claude-opus-5"),
        (GoogleBatchBackend, "google", "gemini-3-flash-preview"),
    ],
)
@pytest.mark.asyncio
async def test_png_batch_serialization(
    tmp_path: Path, backend_cls: Any, provider: str, model: str
) -> None:
    payload = await png_payload(tmp_path, provider, model)
    backend = backend_cls()
    client = MagicMock()
    backend._client = client
    captured = []

    def upload(**kwargs: Any) -> Any:
        captured.append(kwargs["file"].read().decode("utf-8"))
        return MagicMock(id="file")

    client.files.create.side_effect = upload
    backend.submit_batch(
        [
            BatchRequest(
                "page",
                image_base64=payload.base64,
                mime_type=payload.mime_type,
                image_detail=payload.detail,
            )
        ],
        {"extraction_model": {"name": model}},
        system_prompt="Extract.",
    )
    if provider == "anthropic":
        captured.append(json.dumps(client.messages.batches.create.call_args.kwargs))
    elif provider == "google":
        captured.append(json.dumps(client.batches.create.call_args.kwargs))
    assert len(captured) == 1
    assert "image/png" in captured[0] and payload.base64 in captured[0]
