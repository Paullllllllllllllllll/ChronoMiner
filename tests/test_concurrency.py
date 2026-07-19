"""Fan-out primitive contract (item 8).

The generic ``run_concurrent_tasks`` helper (which swallowed task exceptions and
returned ``None``) has been removed. The inline semaphore fan-out inside
``SynchronousProcessingStrategy`` is now the single fan-out primitive; these
tests pin its error-propagation contract: a failing unit yields a *structured*
error dict (never a silent ``None``), and every unit is accounted for exactly
once.
"""

from __future__ import annotations

import importlib
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


@pytest.mark.unit
def test_dead_concurrency_helper_removed():
    """The dead generic helper and its re-export are gone."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("modules.infra.concurrency")

    import modules.infra as infra

    assert not hasattr(infra, "run_concurrent_tasks")


@pytest.mark.asyncio
async def test_inline_fanout_propagates_structured_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing unit surfaces as a structured error dict carrying its
    chunk_index — never a silent None — and every unit is accounted for."""
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )

    async def _process_text_chunk(*, text_chunk: str, **_kwargs) -> dict[str, Any]:
        if text_chunk == "boom":
            # Non-retryable (400) so the loop returns immediately.
            raise RuntimeError("permanent failure: bad request 400")
        return {
            "ok": True,
            "output_text": "extracted",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(ps, "process_text_chunk", _process_text_chunk)

    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={
            "concurrency": {
                "extraction": {
                    "concurrency_limit": 4,
                    "retry": {"attempts": 1},
                }
            }
        }
    )

    results = await strat.process_chunks(
        chunks=["a", "boom", "c"],
        handler=_DummyHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=tmp_path / "in.txt",
        temp_jsonl_path=tmp_path / "t.jsonl",
        console_print=lambda *_a, **_k: None,
    )

    assert len(results) == 3
    # No silent None dropped by the fan-out.
    assert all(r is not None for r in results)

    errors = [r for r in results if isinstance(r, dict) and "error" in r]
    assert len(errors) == 1
    assert errors[0]["chunk_index"] == 2  # 1-based index of "boom"
