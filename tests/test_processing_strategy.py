from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import modules.core.processing_strategy as ps


class _DummyTokenTracker:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.added: List[int] = []

    def add_tokens(self, n: int) -> None:
        self.added.append(n)


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
        model_config: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {"input": chunk, "system": dev_message}


@pytest.mark.asyncio
async def test_create_processing_strategy_factory() -> None:
    batch = ps.create_processing_strategy(use_batch=True)
    assert isinstance(batch, ps.BatchProcessingStrategy)

    sync = ps.create_processing_strategy(use_batch=False, concurrency_config={"x": 1})
    assert isinstance(sync, ps.SynchronousProcessingStrategy)
    assert sync.concurrency_config == {"x": 1}


@pytest.mark.asyncio
async def test_batch_processing_strategy_raises_for_unsupported_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ps, "supports_batch", lambda provider: False)

    strat = ps.BatchProcessingStrategy()
    model_config = {"transcription_model": {"provider": "openrouter", "name": "openrouter/some"}}

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
async def test_batch_processing_strategy_writes_request_and_tracking_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ps, "supports_batch", lambda provider: True)

    class _Backend:
        def submit_batch(
            self,
            requests,
            model_config,
            *,
            system_prompt: str,
            schema: Optional[Dict[str, Any]] = None,
            schema_name: Optional[str] = None,
        ):
            assert system_prompt == "dev"
            assert schema_name == "TestSchema"
            return ps.BatchHandle(provider="openai", batch_id="batch_123", metadata={"ok": True})

    monkeypatch.setattr(ps, "get_batch_backend", lambda provider: _Backend())

    strat = ps.BatchProcessingStrategy()
    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "input.txt"

    model_config = {"transcription_model": {"provider": "openai", "name": "gpt-4o"}}

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

    lines = [json.loads(line) for line in temp_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3

    assert "batch_request" in lines[0]
    assert "batch_request" in lines[1]
    assert lines[0]["batch_request"]["custom_id"] == f"{file_path.stem}-chunk-1"
    assert lines[1]["batch_request"]["custom_id"] == f"{file_path.stem}-chunk-2"

    assert "batch_tracking" in lines[2]
    assert lines[2]["batch_tracking"]["batch_id"] == "batch_123"


@pytest.mark.asyncio
async def test_synchronous_processing_strategy_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai"))
    monkeypatch.setattr(ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: None))

    strat = ps.SynchronousProcessingStrategy(concurrency_config={})

    with pytest.raises(ValueError):
        await strat.process_chunks(
            chunks=["c1"],
            handler=_DummyHandler(),
            dev_message="dev",
            model_config={"transcription_model": {"name": "gpt-4o"}},
            schema={"type": "object"},
            file_path=tmp_path / "file.txt",
            temp_jsonl_path=tmp_path / "temp.jsonl",
            console_print=lambda *_args, **_kwargs: None,
        )


@pytest.mark.asyncio
async def test_synchronous_processing_strategy_writes_temp_jsonl_and_tracks_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai"))
    monkeypatch.setattr(ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key"))

    tracker = _DummyTokenTracker(enabled=True)
    monkeypatch.setattr(ps, "get_token_tracker", lambda: tracker)

    extractor = object()
    monkeypatch.setattr(ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(extractor))

    async def _process_text_chunk(*, text_chunk: str, extractor: object, system_message: str, json_schema: Dict[str, Any]):
        assert extractor is extractor
        assert system_message == "dev"
        return {"ok": True, "usage": {"input_tokens": 2, "output_tokens": 3}, "text": text_chunk}

    monkeypatch.setattr(ps, "process_text_chunk", _process_text_chunk)

    temp_jsonl = tmp_path / "temp.jsonl"
    file_path = tmp_path / "input.txt"

    strat = ps.SynchronousProcessingStrategy(concurrency_config={"concurrency": {"extraction": {"concurrency_limit": 2}}})

    results = await strat.process_chunks(
        chunks=["c1", "c2"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config={"transcription_model": {"name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
    )

    assert len(results) == 2
    assert tracker.added == [5, 5]

    lines = [json.loads(line) for line in temp_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["custom_id"] == f"{file_path.stem}-chunk-1"
    assert lines[1]["custom_id"] == f"{file_path.stem}-chunk-2"


@pytest.mark.asyncio
async def test_synchronous_processing_strategy_anthropic_rate_limit_retries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "anthropic"))
    monkeypatch.setattr(ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key"))

    tracker = _DummyTokenTracker(enabled=False)
    monkeypatch.setattr(ps, "get_token_tracker", lambda: tracker)

    monkeypatch.setattr(ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object()))
    monkeypatch.setattr(ps.random, "uniform", lambda a, b: 0.0)

    async def _noop_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(ps.asyncio, "sleep", _noop_sleep)

    calls = {"n": 0}

    async def _process_text_chunk(*, text_chunk: str, extractor: object, system_message: str, json_schema: Dict[str, Any]):
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
                    "retry": {"attempts": 2, "wait_min_seconds": 0.0, "wait_max_seconds": 0.0, "jitter_max_seconds": 0.0},
                }
            }
        }
    )

    results = await strat.process_chunks(
        chunks=["c1"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config={"transcription_model": {"name": "claude"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_args, **_kwargs: None,
    )

    assert results[0].get("ok") is True
    assert calls["n"] == 2
