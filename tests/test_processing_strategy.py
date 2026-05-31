from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import modules.extract.processing_strategy as ps


class _AsyncExtractorCM:
    def __init__(self, extractor: object):
        self._extractor = extractor

    async def __aenter__(self) -> object:
        return self._extractor

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummyHandler:
    schema_name = "TestSchema"

    def prepare_payload(
        self,
        chunk: str,
        dev_message: str,
        model_config: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        return {"input": chunk, "system": dev_message}


@pytest.mark.asyncio
async def test_create_processing_strategy_factory() -> None:
    batch = ps.create_processing_strategy(use_batch=True)
    assert isinstance(batch, ps.BatchProcessingStrategy)

    sync = ps.create_processing_strategy(use_batch=False, concurrency_config={"x": 1})
    assert isinstance(sync, ps.SynchronousProcessingStrategy)
    assert sync.concurrency_config == {"x": 1}


@pytest.mark.asyncio
async def test_batch_processing_strategy_raises_for_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ps, "supports_batch", lambda provider: False)

    strat = ps.BatchProcessingStrategy()
    model_config = {
        "extraction_model": {"provider": "openrouter", "name": "openrouter/some"}
    }

    with pytest.raises(ValueError):
        await strat.process_chunks(
            chunks=["a"],
            handler=_DummyHandler(),
            dev_message="dev",
            model_config=model_config,
            schema={"type": "object"},
            file_path=tmp_path / "file.txt",
            temp_jsonl_path=tmp_path / "temp.jsonl",
            console_print=lambda *_args, **_kwargs: None,
        )


@pytest.mark.asyncio
async def test_batch_processing_strategy_writes_request_and_tracking_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ps, "supports_batch", lambda provider: True)

    class _Backend:
        def submit_batch(
            self,
            requests,
            model_config,
            *,
            system_prompt: str,
            schema: dict[str, Any] | None = None,
            schema_name: str | None = None,
        ):
            assert system_prompt == "dev"
            assert schema_name == "TestSchema"
            return ps.BatchHandle(
                provider="openai", batch_id="batch_123", metadata={"ok": True}
            )

    monkeypatch.setattr(ps, "get_batch_backend", lambda provider: _Backend())

    strat = ps.BatchProcessingStrategy()
    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "input.txt"

    model_config = {"extraction_model": {"provider": "openai", "name": "gpt-4o"}}

    res = await strat.process_chunks(
        chunks=["c1", "c2"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config=model_config,
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
    )

    assert res == []
    assert temp_jsonl.exists()

    lines = [
        json.loads(line)
        for line in temp_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 3

    assert "batch_request" in lines[0]
    assert "batch_request" in lines[1]
    assert lines[0]["batch_request"]["custom_id"] == f"{file_path.stem}-chunk-1"
    assert lines[1]["batch_request"]["custom_id"] == f"{file_path.stem}-chunk-2"

    assert "batch_tracking" in lines[2]
    assert lines[2]["batch_tracking"]["batch_id"] == "batch_123"


@pytest.mark.asyncio
async def test_batch_processing_strategy_builds_visual_batch_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When image_chunks are provided, BatchProcessingStrategy builds visual
    BatchRequests."""
    monkeypatch.setattr(ps, "supports_batch", lambda provider: True)

    captured_requests: list = []

    class _VisualBackend:
        def submit_batch(
            self,
            requests,
            model_config,
            *,
            system_prompt: str,
            schema: dict[str, Any] | None = None,
            schema_name: str | None = None,
        ):
            captured_requests.extend(requests)
            return ps.BatchHandle(
                provider="openai", batch_id="vis_batch_1", metadata={}
            )

    monkeypatch.setattr(ps, "get_batch_backend", lambda provider: _VisualBackend())

    strat = ps.BatchProcessingStrategy()
    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "doc.pdf"

    image_chunks = [
        {"base64": "AAA=", "mime_type": "image/jpeg", "detail": "low"},
        {"base64": "BBB=", "mime_type": "image/jpeg", "detail": "low"},
    ]

    model_config = {"extraction_model": {"provider": "openai", "name": "gpt-4o"}}

    res = await strat.process_chunks(
        chunks=[],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config=model_config,
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
        image_chunks=image_chunks,
    )

    assert res == []
    # Exactly two visual BatchRequests were submitted
    assert len(captured_requests) == 2
    for req in captured_requests:
        assert req.is_visual is True
        assert req.image_base64 is not None
        assert req.mime_type == "image/jpeg"
    # Custom IDs use -page- pattern
    assert captured_requests[0].custom_id == f"{file_path.stem}-page-1"
    assert captured_requests[1].custom_id == f"{file_path.stem}-page-2"

    # Tracking JSONL written with correct page metadata
    lines = [
        json.loads(line)
        for line in temp_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    page_records = [ln for ln in lines if "batch_request" in ln]
    assert len(page_records) == 2
    assert page_records[0]["batch_request"]["custom_id"] == f"{file_path.stem}-page-1"
    assert page_records[0]["batch_request"]["metadata"]["page_index"] == 1
    assert page_records[0]["batch_request"]["metadata"]["total_pages"] == 2

    tracking = [ln for ln in lines if "batch_tracking" in ln]
    assert len(tracking) == 1
    assert tracking[0]["batch_tracking"]["batch_id"] == "vis_batch_1"


@pytest.mark.asyncio
async def test_synchronous_processing_strategy_raises_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: None)
    )

    strat = ps.SynchronousProcessingStrategy(concurrency_config={})

    with pytest.raises(ValueError):
        await strat.process_chunks(
            chunks=["c1"],
            handler=_DummyHandler(),
            dev_message="dev",
            model_config={"extraction_model": {"name": "gpt-4o"}},
            schema={"type": "object"},
            file_path=tmp_path / "file.txt",
            temp_jsonl_path=tmp_path / "temp.jsonl",
            console_print=lambda *_args, **_kwargs: None,
        )


@pytest.mark.asyncio
async def test_synchronous_processing_strategy_writes_temp_jsonl_and_tracks_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )

    extractor = object()
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(extractor)
    )

    async def _process_text_chunk(
        *,
        text_chunk: str,
        extractor: object,
        system_message: str,
        json_schema: dict[str, Any],
        **kwargs,
    ):
        assert extractor is extractor
        assert system_message == "dev"
        return {
            "ok": True,
            "usage": {"input_tokens": 2, "output_tokens": 3},
            "text": text_chunk,
        }

    monkeypatch.setattr(ps, "process_text_chunk", _process_text_chunk)

    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "input.txt"

    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={"concurrency": {"extraction": {"concurrency_limit": 2}}}
    )

    results = await strat.process_chunks(
        chunks=["c1", "c2"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
    )

    assert len(results) == 2

    lines = [
        json.loads(line)
        for line in temp_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2
    assert lines[0]["custom_id"] == f"{file_path.stem}-chunk-1"
    assert lines[1]["custom_id"] == f"{file_path.stem}-chunk-2"


@pytest.mark.asyncio
async def test_synchronous_strategy_writes_chunk_index_for_ordering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression (C4): sync temp records must carry ``chunk_index`` so the
    final output can be ordered. Without it, ``_generate_output_files`` sorts
    by ``None or 0`` (all equal) and records land in completion order."""
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )

    async def _process_text_chunk(
        *,
        text_chunk: str,
        extractor: object,
        system_message: str,
        json_schema: dict[str, Any],
        **kwargs,
    ):
        return {"ok": True, "usage": {"input_tokens": 0, "output_tokens": 0}}

    monkeypatch.setattr(ps, "process_text_chunk", _process_text_chunk)

    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "input.txt"

    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={"concurrency": {"extraction": {"concurrency_limit": 3}}}
    )

    await strat.process_chunks(
        chunks=["c1", "c2", "c3"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
    )

    lines = [
        json.loads(line)
        for line in temp_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 3
    for rec in lines:
        assert "chunk_index" in rec
        idx_from_id = int(rec["custom_id"].rsplit("-chunk-", 1)[1])
        assert rec["chunk_index"] == idx_from_id
    # Sorting by chunk_index recovers ascending order regardless of write order.
    assert sorted(rec["chunk_index"] for rec in lines) == [1, 2, 3]


@pytest.mark.asyncio
async def test_synchronous_strategy_concurrent_writes_are_not_interleaved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression (A1): concurrent coroutines share one temp-file handle.
    Without the write lock, ``write``+``flush`` from different coroutines can
    interleave and corrupt lines. Every persisted line must parse as JSON and
    the line count must equal the chunk count."""
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )

    async def _process_text_chunk(
        *,
        text_chunk: str,
        extractor: object,
        system_message: str,
        json_schema: dict[str, Any],
        **kwargs,
    ):
        # Yield control so coroutines reach the write section concurrently,
        # maximizing the chance of an interleave without the lock.
        await ps.asyncio.sleep(0)
        return {
            "ok": True,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "payload": text_chunk * 50,
        }

    monkeypatch.setattr(ps, "process_text_chunk", _process_text_chunk)

    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "input.txt"
    chunks = [f"c{i}" for i in range(40)]

    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={
            "concurrency": {"extraction": {"concurrency_limit": 16}}
        }
    )

    await strat.process_chunks(
        chunks=chunks,
        handler=_DummyHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
    )

    raw_lines = [
        line
        for line in temp_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(raw_lines) == len(chunks)
    parsed = [json.loads(line) for line in raw_lines]
    assert sorted(rec["chunk_index"] for rec in parsed) == list(
        range(1, len(chunks) + 1)
    )


@pytest.mark.asyncio
async def test_synchronous_strategy_error_dict_carries_chunk_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression (A2): a failed chunk writes no temp record, so its result
    dict must carry ``chunk_index`` for the caller to identify which chunk
    failed (gather order cannot be trusted for that)."""
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )

    async def _process_text_chunk(
        *,
        text_chunk: str,
        extractor: object,
        system_message: str,
        json_schema: dict[str, Any],
        **kwargs,
    ):
        if text_chunk == "c2":
            raise RuntimeError("permanent failure: bad request 400")
        return {"ok": True, "usage": {"input_tokens": 0, "output_tokens": 0}}

    monkeypatch.setattr(ps, "process_text_chunk", _process_text_chunk)

    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "input.txt"

    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={"concurrency": {"extraction": {"concurrency_limit": 3}}}
    )

    results = await strat.process_chunks(
        chunks=["c1", "c2", "c3"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
    )

    errors = [
        r for r in results if isinstance(r, dict) and "error" in r
    ]
    assert len(errors) == 1
    # c2 is the second chunk -> 1-based index 2.
    assert errors[0]["chunk_index"] == 2
    # The failed chunk wrote no temp record.
    lines = [
        json.loads(line)
        for line in temp_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sorted(rec["chunk_index"] for rec in lines) == [1, 3]


@pytest.mark.asyncio
async def test_synchronous_processing_strategy_forwards_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )

    captured_kwargs: dict[str, Any] = {}

    def _open_extractor_stub(**kwargs):
        captured_kwargs.update(kwargs)
        return _AsyncExtractorCM(object())

    monkeypatch.setattr(ps, "open_extractor", _open_extractor_stub)

    async def _process_text_chunk(
        *,
        text_chunk: str,
        extractor: object,
        system_message: str,
        json_schema: dict[str, Any],
        **kwargs,
    ):
        return {
            "ok": True,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "text": text_chunk,
        }

    monkeypatch.setattr(ps, "process_text_chunk", _process_text_chunk)

    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={"concurrency": {"extraction": {"concurrency_limit": 1}}}
    )
    model_config = {
        "extraction_model": {
            "name": "gpt-5-mini",
            "max_output_tokens": 12000,
            "reasoning": {"effort": "high"},
            "text": {"verbosity": "low"},
        }
    }

    await strat.process_chunks(
        chunks=["c1"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config=model_config,
        schema={"type": "object"},
        file_path=tmp_path / "input.txt",
        temp_jsonl_path=tmp_path / "temp.jsonl",
        console_print=lambda *_args, **_kwargs: None,
    )

    assert captured_kwargs["model"] == "gpt-5-mini"
    assert captured_kwargs["model_config_override"] == model_config
    assert captured_kwargs["concurrency_config_override"] == strat.concurrency_config


@pytest.mark.asyncio
async def test_synchronous_processing_strategy_anthropic_rate_limit_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "anthropic")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )

    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )
    monkeypatch.setattr(ps.random, "uniform", lambda a, b: 0.0)

    async def _noop_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(ps.asyncio, "sleep", _noop_sleep)

    calls = {"n": 0}

    async def _process_text_chunk(
        *,
        text_chunk: str,
        extractor: object,
        system_message: str,
        json_schema: dict[str, Any],
        **kwargs,
    ):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 rate_limit")
        return {"ok": True, "usage": {"input_tokens": 0, "output_tokens": 0}}

    monkeypatch.setattr(ps, "process_text_chunk", _process_text_chunk)

    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "input.txt"

    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={
            "concurrency": {
                "extraction": {
                    "concurrency_limit": 5,
                    "retry": {
                        "attempts": 2,
                        "wait_min_seconds": 0.0,
                        "wait_max_seconds": 0.0,
                        "jitter_max_seconds": 0.0,
                    },
                }
            }
        }
    )

    results = await strat.process_chunks(
        chunks=["c1"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"name": "claude"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
    )

    assert results[0].get("ok") is True
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_anthropic_respects_configured_concurrency_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anthropic should use the configured concurrency_limit,
    not be hard-capped to 1."""
    monkeypatch.setattr(
        ps.ProviderConfig,
        "_detect_provider",
        staticmethod(lambda model: "anthropic"),
    )
    monkeypatch.setattr(
        ps.ProviderConfig,
        "_get_api_key",
        staticmethod(lambda provider: "key"),
    )
    monkeypatch.setattr(
        ps,
        "open_extractor",
        lambda **_kwargs: _AsyncExtractorCM(object()),
    )

    async def _process_text_chunk(**_kw: Any) -> dict[str, Any]:
        return {"ok": True, "usage": {"input_tokens": 0, "output_tokens": 0}}

    monkeypatch.setattr(ps, "process_text_chunk", _process_text_chunk)

    captured_semaphore_values: list[int] = []
    _original_semaphore = ps.asyncio.Semaphore

    class _CaptureSemaphore:
        def __init__(self, value: int = 1):
            captured_semaphore_values.append(value)
            self._sem = _original_semaphore(value)

        async def __aenter__(self):
            return await self._sem.__aenter__()

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object,
        ) -> None:
            await self._sem.__aexit__(exc_type, exc, tb)  # type: ignore[arg-type]

    monkeypatch.setattr(ps.asyncio, "Semaphore", _CaptureSemaphore)

    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={
            "concurrency": {
                "extraction": {
                    "concurrency_limit": 10,
                    "delay_between_tasks": 0,
                }
            }
        }
    )

    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "input.txt"

    await strat.process_chunks(
        chunks=["c1", "c2", "c3"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"name": "claude-sonnet"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
    )

    assert len(captured_semaphore_values) == 1
    assert captured_semaphore_values[0] == 3
