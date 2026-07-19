"""Regression tests for the Tier 1 hardening fixes.

Each test exercises the actual production composition path (not an isolated
unit), so it fails against the pre-fix code and passes against the fix:

* Bug 1  — normalize (rstrip) -> chunk join must preserve line breaks.
* Bug 2  — prompt templates resolve via an absolute PROMPTS_DIR, not CWD.
* Bug 3  — a sliced text run writes ABSOLUTE custom_ids / chunk_index and
           populates chunk_range; a legacy (unversioned) temp JSONL is refused
           on resume rather than silently corrupting the merge.
* Bug 4  — Anthropic batch results serialize the SDK Message to a plain dict.
* Bug 5  — batch request lists split by provider count/byte limits.
* Bug 6  — the batch-ID recovery artifact is written after submission.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import modules.extract.processing_strategy as ps


class _AsyncExtractorCM:
    def __init__(self, extractor: object) -> None:
        self._extractor = extractor

    async def __aenter__(self) -> object:
        return self._extractor

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _StubHandler:
    schema_name = "TestSchema"


# ---------------------------------------------------------------------------
# Bug 1: newline preservation across the normalize -> chunk composition
# ---------------------------------------------------------------------------


def test_normalize_then_chunk_preserves_line_boundaries() -> None:
    """Compose the production path (rstrip each line, then chunk) and assert
    words at line boundaries are not run together. On the old code (join with
    "") this produced 'Zuckerund'."""
    from modules.infra.chunking import ChunkingService

    raw_lines = ["Zucker\n", "und\n", "Mehl\n", "sowie\n"]
    # Exactly what FileProcessor.process_file does at read time.
    normalized = [line.rstrip("\n\r") for line in raw_lines]

    svc = ChunkingService(model_name="gpt-4o", default_tokens_per_chunk=100000)
    chunks, _ranges = svc.chunk_text(normalized, strategy="auto")

    joined = "\n".join(chunks)
    assert "Zuckerund" not in joined
    assert "Mehlsowie" not in joined
    assert "Zucker\nund\nMehl\nsowie" in joined


# ---------------------------------------------------------------------------
# Bug 2: prompt templates resolve absolutely, independent of CWD
# ---------------------------------------------------------------------------


def test_prompts_dir_is_absolute_and_prompts_exist() -> None:
    from modules.llm.prompt_utils import PROMPTS_DIR, load_prompt_template, prompt_path

    assert PROMPTS_DIR.is_absolute()
    for name in (
        "text_extraction_prompt.txt",
        "image_extraction_prompt.txt",
    ):
        p = prompt_path(name)
        assert p.is_absolute()
        assert p.exists(), f"bundled prompt missing: {p}"
        # Loadable regardless of the current working directory.
        assert load_prompt_template(p)


# ---------------------------------------------------------------------------
# Bug 3: absolute chunk indices for a sliced text run + chunk_range
# ---------------------------------------------------------------------------


def _make_file_processor() -> object:
    from modules.extract.file_processor import FileProcessor

    return FileProcessor(
        paths_config={
            "general": {
                "input_paths_is_output_path": True,
                "retain_temporary_jsonl": True,
            }
        },
        model_config={"extraction_model": {"name": "gpt-4o"}},
        chunking_config={"chunking": {"default_tokens_per_chunk": 1}},
        concurrency_config={"concurrency": {"extraction": {"concurrency_limit": 1}}},
    )


@pytest.mark.asyncio
async def test_sliced_text_run_writes_absolute_custom_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import modules.extract.file_processor as fp_mod

    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda m: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda p: "key")
    )
    monkeypatch.setattr(ps, "open_extractor", lambda **_k: _AsyncExtractorCM(object()))
    monkeypatch.setattr(fp_mod, "get_schema_handler", lambda _name: _StubHandler())

    async def _fake_chunk(*, text_chunk: str, **_kw: object) -> dict[str, object]:
        return {
            "ok": True,
            "output_text": "extracted",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    monkeypatch.setattr(ps, "process_text_chunk", _fake_chunk)

    doc = tmp_path / "doc.txt"
    doc.write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")

    from modules.infra.chunking import ChunkSlice

    fp = _make_file_processor()
    await fp.process_file(
        file_path=doc,
        use_batch=False,
        selected_schema={"schema": {"type": "object"}},
        prompt_template="PROMPT",
        schema_name="TestSchema",
        inject_schema=False,
        schema_paths={},
        global_chunking_method="auto",
        resume=False,
        chunk_slice=ChunkSlice(page_range=(2, 3)),
    )

    out = json.loads((tmp_path / "doc_output.json").read_text(encoding="utf-8"))
    records = out["records"]
    cids = {r["custom_id"] for r in records}
    # Document-space indices 2 and 3 — NOT slice-relative 1 and 2.
    assert cids == {"doc-chunk-2", "doc-chunk-3"}
    by_id = {r["custom_id"]: r for r in records}
    assert by_id["doc-chunk-2"]["chunk_index"] == 2
    assert by_id["doc-chunk-3"]["chunk_index"] == 3
    # chunk_range is populated (Bug 14).
    assert by_id["doc-chunk-2"]["chunk_range"] == [2, 2]
    assert by_id["doc-chunk-3"]["chunk_range"] == [3, 3]
    # Text output stamps the chunking behaviour version.
    assert out["_chronominer_metadata"]["chunking_text_version"] == 2


@pytest.mark.asyncio
async def test_resume_refused_for_unversioned_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A legacy temp JSONL (no version header) must be refused on resume rather
    than merged, and no API call is made."""
    import modules.extract.file_processor as fp_mod

    called = {"n": 0}

    async def _fake_chunk(*, text_chunk: str, **_kw: object) -> dict[str, object]:
        called["n"] += 1
        return {"ok": True, "output_text": "extracted", "usage": {}}

    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda m: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda p: "key")
    )
    monkeypatch.setattr(ps, "open_extractor", lambda **_k: _AsyncExtractorCM(object()))
    monkeypatch.setattr(ps, "process_text_chunk", _fake_chunk)
    monkeypatch.setattr(fp_mod, "get_schema_handler", lambda _name: _StubHandler())

    doc = tmp_path / "doc.txt"
    doc.write_text("l1\nl2\nl3\n", encoding="utf-8")

    # Partial prior output (chunk 1 of 3 done).
    out_path = tmp_path / "doc_output.json"
    out_path.write_text(
        json.dumps(
            {
                "_chronominer_metadata": {"total_chunks": 3},
                "records": [{"custom_id": "doc-chunk-1", "chunk_index": 1}],
            }
        ),
        encoding="utf-8",
    )
    # Legacy temp JSONL: no version header line.
    temp_path = tmp_path / "doc_temp.jsonl"
    temp_path.write_text(
        json.dumps(
            {"custom_id": "doc-chunk-1", "chunk_index": 1, "response": {"body": {}}}
        )
        + "\n",
        encoding="utf-8",
    )

    fp = _make_file_processor()
    await fp.process_file(
        file_path=doc,
        use_batch=False,
        selected_schema={"schema": {"type": "object"}},
        prompt_template="PROMPT",
        schema_name="TestSchema",
        inject_schema=False,
        schema_paths={},
        global_chunking_method="auto",
        resume=True,
    )

    # Refused before any processing: no API calls, prior output untouched.
    assert called["n"] == 0
    after = json.loads(out_path.read_text(encoding="utf-8"))
    assert [r["custom_id"] for r in after["records"]] == ["doc-chunk-1"]


def test_versioned_temp_is_resumable(tmp_path: Path) -> None:
    from modules.extract.resume import (
        build_temp_header,
        is_resumable_temp_jsonl,
    )

    temp = tmp_path / "doc_temp.jsonl"
    # Missing file is resumable.
    assert is_resumable_temp_jsonl(temp) is True
    # Current-version header is resumable.
    temp.write_text(json.dumps(build_temp_header()) + "\n", encoding="utf-8")
    assert is_resumable_temp_jsonl(temp) is True
    # Unversioned (legacy) file is refused.
    temp.write_text(json.dumps({"custom_id": "x"}) + "\n", encoding="utf-8")
    assert is_resumable_temp_jsonl(temp) is False


# ---------------------------------------------------------------------------
# Bug 4: Anthropic SDK Message serialization
# ---------------------------------------------------------------------------


def test_anthropic_message_to_dict_is_json_serializable() -> None:
    from modules.batch.backends.anthropic_backend import _message_to_dict

    class _FakeMessage:
        def model_dump(self, mode: str | None = None) -> dict[str, object]:
            return {"id": "msg_1", "content": [{"type": "text", "text": "hi"}]}

    out = _message_to_dict(_FakeMessage())
    # Must round-trip through json.dumps (the finalization step that used to
    # crash on the raw SDK object).
    assert json.loads(json.dumps({"message": out}))["message"]["id"] == "msg_1"


# ---------------------------------------------------------------------------
# Bug 5: batch request partitioning by count and byte limits
# ---------------------------------------------------------------------------


def test_partition_batch_requests_by_count_and_bytes() -> None:
    from modules.batch.backends.base import BatchRequest
    from modules.extract.processing_strategy import _partition_batch_requests

    reqs = [BatchRequest(custom_id=f"c-{i}", text="x") for i in range(10)]

    # Count limit: 4 per part -> 3 parts (4, 4, 2).
    parts = _partition_batch_requests(reqs, max_count=4, max_bytes=10**9)
    assert [len(p) for p in parts] == [4, 4, 2]
    # Every request preserved, order intact.
    assert [r.custom_id for part in parts for r in part] == [r.custom_id for r in reqs]

    # Byte limit dominates: a tiny cap forces one request per part.
    big = [BatchRequest(custom_id=f"b-{i}", text="y" * 5000) for i in range(3)]
    parts_b = _partition_batch_requests(big, max_count=50000, max_bytes=6000)
    assert [len(p) for p in parts_b] == [1, 1, 1]


@pytest.mark.asyncio
async def test_batch_split_writes_part_files_and_debug_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch that exceeds the provider count limit is split into _part{n}
    temp files (one tracking record each) and the recovery artifact lists every
    batch id."""
    monkeypatch.setattr(ps, "supports_batch", lambda provider: True)

    submitted: list[int] = []

    class _Backend:
        max_batch_size = 2
        max_batch_bytes = 10**9

        def submit_batch(
            self,
            requests: list[object],
            model_config: dict[str, object],
            *,
            system_prompt: str,
            schema: dict[str, object] | None = None,
            schema_name: str | None = None,
        ) -> object:
            submitted.append(len(requests))
            return ps.BatchHandle(
                provider="openai",
                batch_id=f"batch_{len(submitted)}",
                metadata={},
            )

    monkeypatch.setattr(ps, "get_batch_backend", lambda provider: _Backend())

    strat = ps.BatchProcessingStrategy()
    temp_jsonl = tmp_path / "doc_temp.jsonl"
    file_path = tmp_path / "doc.txt"

    res = await strat.process_chunks(
        chunks=["c1", "c2", "c3", "c4", "c5"],
        handler=_StubHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"provider": "openai", "name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_a, **_k: None,
    )
    assert res == []

    # 5 requests, max 2 per batch -> 3 parts.
    assert submitted == [2, 2, 1]
    part_files = sorted(tmp_path.glob("doc_temp_part*.jsonl"))
    assert [p.name for p in part_files] == [
        "doc_temp_part1.jsonl",
        "doc_temp_part2.jsonl",
        "doc_temp_part3.jsonl",
    ]
    # Each part carries exactly one tracking record.
    for pf in part_files:
        tracking = [
            json.loads(line)
            for line in pf.read_text(encoding="utf-8").splitlines()
            if line.strip() and "batch_tracking" in json.loads(line)
        ]
        assert len(tracking) == 1

    # Bug 6: recovery artifact lists every submitted batch id.
    debug = tmp_path / "doc_batch_submission_debug.json"
    assert debug.exists()
    payload = json.loads(debug.read_text(encoding="utf-8"))
    assert payload["provider"] == "openai"
    assert payload["batch_ids"] == ["batch_1", "batch_2", "batch_3"]


@pytest.mark.asyncio
async def test_single_part_batch_uses_plain_temp_and_writes_debug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch under the limits keeps the plain ``_temp.jsonl`` name (no _part
    suffix) and still writes the recovery artifact."""
    monkeypatch.setattr(ps, "supports_batch", lambda provider: True)

    class _Backend:
        max_batch_size = 50000
        max_batch_bytes = 10**9

        def submit_batch(self, requests, model_config, **_kw):  # type: ignore[no-untyped-def]
            return ps.BatchHandle(provider="openai", batch_id="only", metadata={})

    monkeypatch.setattr(ps, "get_batch_backend", lambda provider: _Backend())

    strat = ps.BatchProcessingStrategy()
    temp_jsonl = tmp_path / "doc_temp.jsonl"
    file_path = tmp_path / "doc.txt"

    await strat.process_chunks(
        chunks=["c1", "c2"],
        handler=_StubHandler(),
        dev_message="dev",
        model_config={"extraction_model": {"provider": "openai", "name": "gpt-4o"}},
        schema={"type": "object"},
        file_path=file_path,
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_a, **_k: None,
    )

    assert temp_jsonl.exists()
    assert not list(tmp_path.glob("doc_temp_part*.jsonl"))
    debug = tmp_path / "doc_batch_submission_debug.json"
    assert json.loads(debug.read_text(encoding="utf-8"))["batch_ids"] == ["only"]
