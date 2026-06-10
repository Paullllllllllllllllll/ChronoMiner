"""Tests for the streaming visual pipeline.

Covers the in-memory image helpers, the page-stream producer, temp-record
slimming, the streaming execution path of the synchronous strategy, the
file-level concurrency cap, and the slim_temp_jsonl maintenance utility.
All offline (no API calls).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import fitz
import pytest
from PIL import Image

import modules.extract.processing_strategy as ps
from modules.conversion.json_utils import strip_image_payloads
from modules.images.encoding import encode_bytes_to_base64
from modules.images.llm_preprocess import ImageProcessor
from modules.images.page_stream import (
    PageError,
    PagePayload,
    build_image_provenance,
    stream_page_payloads,
)

_IMAGE_CONFIG: dict[str, Any] = {
    "target_dpi": 72,
    "max_pixels_per_page": 0,
    "api_image_processing": {
        "grayscale_conversion": False,
        "handle_transparency": True,
        "jpeg_quality": 90,
        "resize_profile": "original",
        "llm_detail": "high",
        "original_max_side_px": 6000,
        "original_max_pixels": 10240000,
    },
}


def _make_pdf(path: Path, pages: int) -> None:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=300)
        page.insert_text((50, 100), f"Page {i + 1}")
    doc.save(path)
    doc.close()


def _make_image(path: Path, size: tuple[int, int] = (120, 80)) -> None:
    Image.new("RGB", size, (200, 10, 10)).save(path, format="PNG")


# ---------------------------------------------------------------------------
# In-memory helpers
# ---------------------------------------------------------------------------


def test_process_pil_matches_process_image(tmp_path: Path) -> None:
    src = tmp_path / "img.png"
    _make_image(src)

    processor = ImageProcessor(
        image_path=src,
        provider="openai",
        model_name="gpt-5-mini",
        image_config=_IMAGE_CONFIG,
    )
    on_disk = processor.process_image(tmp_path / "out")
    with Image.open(src) as img:
        in_memory = processor.process_pil(img)

    assert on_disk.read_bytes() == in_memory


def test_process_image_without_path_raises() -> None:
    processor = ImageProcessor(provider="openai", image_config=_IMAGE_CONFIG)
    with pytest.raises(ValueError, match="requires an image_path"):
        processor.process_image(Path("out"))


def test_encode_bytes_to_base64_roundtrip() -> None:
    data = b"\xff\xd8jpegdata"
    encoded = encode_bytes_to_base64(data, "image/jpeg")
    assert base64.b64decode(encoded) == data

    with pytest.raises(ValueError, match="Unsupported image MIME type"):
        encode_bytes_to_base64(data, "application/pdf")


# ---------------------------------------------------------------------------
# strip_image_payloads
# ---------------------------------------------------------------------------


def _fat_result() -> dict[str, Any]:
    big = "data:image/jpeg;base64," + ("A" * 5000)
    return {
        "output_text": "ok",
        "response_data": {"usage": {"total_tokens": 10}},
        "request_metadata": {
            "model": "gpt-5-mini",
            "provider": "openai",
            "messages": [
                {"role": "system", "content": [{"type": "input_text", "text": "sys"}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract."},
                        {"type": "image_url", "image_url": {"url": big}},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "B" * 4000,
                            },
                        },
                        {"type": "image_url", "image_url": big},
                    ],
                },
            ],
        },
    }


def test_strip_image_payloads_removes_images_keeps_rest() -> None:
    lean = strip_image_payloads(_fat_result())

    serialized = json.dumps(lean)
    assert "A" * 100 not in serialized
    assert "B" * 100 not in serialized

    user_content = lean["request_metadata"]["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "Extract."}
    assert user_content[1]["type"] == "image_omitted"
    assert user_content[1]["byte_size"] > 5000
    assert user_content[2] == {"type": "image_omitted", "byte_size": 4000}
    assert user_content[3]["type"] == "image_omitted"

    assert lean["output_text"] == "ok"
    assert lean["request_metadata"]["model"] == "gpt-5-mini"
    # System message untouched
    assert lean["request_metadata"]["messages"][0]["content"][0]["text"] == "sys"


def test_strip_image_payloads_idempotent_and_passthrough() -> None:
    lean = strip_image_payloads(_fat_result())
    assert strip_image_payloads(lean) == lean

    assert strip_image_payloads("not a dict") == "not a dict"
    assert strip_image_payloads({"output_text": "x"}) == {"output_text": "x"}
    # Non-data URLs (regular links) are preserved
    keep = {
        "request_metadata": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x/y.jpg"}}
                    ],
                }
            ]
        }
    }
    assert strip_image_payloads(keep) == keep


# ---------------------------------------------------------------------------
# Page-stream producer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_page_payloads_pdf_selected_indices(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, pages=5)

    payloads = [
        p
        async for p in stream_page_payloads(
            file_path=pdf_path,
            page_indices=[2, 4],
            image_config=_IMAGE_CONFIG,
            provider="openai",
            model_name="gpt-5-mini",
            image_detail="high",
        )
    ]

    assert [p.index for p in payloads] == [2, 4]
    for payload in payloads:
        assert isinstance(payload, PagePayload)
        jpeg_bytes = base64.b64decode(payload.base64)
        assert payload.sha256 == hashlib.sha256(jpeg_bytes).hexdigest()
        assert payload.byte_size == len(jpeg_bytes)
        assert payload.mime_type == "image/jpeg"
        assert payload.width > 0 and payload.height > 0
        assert payload.effective_dpi == 72
        chunk = payload.as_chunk()
        assert chunk["image_provenance"]["image_sha256"] == payload.sha256


@pytest.mark.asyncio
async def test_stream_page_payloads_single_image(tmp_path: Path) -> None:
    img_path = tmp_path / "scan.png"
    _make_image(img_path)

    payloads = [
        p
        async for p in stream_page_payloads(
            file_path=img_path,
            page_indices=[1],
            image_config=_IMAGE_CONFIG,
            provider="openai",
            model_name="gpt-5-mini",
            image_detail="high",
        )
    ]

    assert len(payloads) == 1
    assert isinstance(payloads[0], PagePayload)
    assert payloads[0].index == 1
    assert payloads[0].effective_dpi is None


@pytest.mark.asyncio
async def test_stream_page_payloads_emits_page_error(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, pages=2)

    # Page index 99 is out of range -> render error -> PageError yielded
    items = [
        p
        async for p in stream_page_payloads(
            file_path=pdf_path,
            page_indices=[1, 99],
            image_config=_IMAGE_CONFIG,
            provider="openai",
            model_name="gpt-5-mini",
            image_detail="high",
        )
    ]

    assert isinstance(items[0], PagePayload)
    assert isinstance(items[1], PageError)
    assert items[1].index == 99


def test_build_image_provenance(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, pages=1)

    prov = build_image_provenance(
        pdf_path, _IMAGE_CONFIG, "openai", "gpt-5-mini", "high"
    )

    assert prov["source_file"] == "doc.pdf"
    assert prov["source_sha256"] == hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    assert prov["pillow_version"]
    assert prov["pymupdf_version"]
    cfg = prov["image_config"]
    assert cfg["target_dpi"] == 72
    assert cfg["resize_profile"] == "original"
    assert cfg["jpeg_quality"] == 90
    assert cfg["detail"] == "high"


# ---------------------------------------------------------------------------
# Synchronous strategy: streaming execution path
# ---------------------------------------------------------------------------


class _AsyncExtractorCM:
    def __init__(self, extractor: object):
        self._extractor = extractor

    async def __aenter__(self) -> object:
        return self._extractor

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _dummy_payload(index: int) -> PagePayload:
    data = f"jpegbytes-{index}".encode()
    return PagePayload(
        index=index,
        base64=base64.b64encode(data).decode(),
        mime_type="image/jpeg",
        detail="high",
        sha256=hashlib.sha256(data).hexdigest(),
        width=100,
        height=150,
        byte_size=len(data),
        effective_dpi=300,
    )


@pytest.mark.asyncio
async def test_sync_strategy_streaming_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )

    seen_images: list[str] = []
    fat_image = "data:image/jpeg;base64," + ("Z" * 3000)

    async def _process_image_chunk(*, image_base64: str, **_kwargs: Any) -> dict:
        seen_images.append(image_base64)
        return {
            "output_text": "extracted",
            "response_data": {"usage": {}},
            "request_metadata": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": fat_image}}
                        ],
                    }
                ]
            },
        }

    monkeypatch.setattr(ps, "process_image_chunk", _process_image_chunk)

    async def _source() -> Any:
        yield _dummy_payload(1)
        yield PageError(index=2, error="render boom")
        yield _dummy_payload(3)

    temp_jsonl = tmp_path / "temp.jsonl"
    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={"concurrency": {"extraction": {"concurrency_limit": 2}}}
    )

    results = await strat.process_chunks(
        chunks=["", "", ""],
        handler=None,
        dev_message="dev",
        model_config={"extraction_model": {"name": "gpt-5-mini"}},
        schema={"type": "object"},
        file_path=tmp_path / "doc.pdf",
        temp_jsonl_path=temp_jsonl,
        console_print=lambda *_a, **_k: None,
        image_source=_source(),
    )

    assert len(results) == 3
    errors = [r for r in results if "error" in r]
    assert len(errors) == 1
    assert errors[0]["chunk_index"] == 2

    lines = [
        json.loads(line)
        for line in temp_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert sorted(rec["chunk_index"] for rec in lines) == [1, 3]
    for rec in lines:
        assert rec["image_provenance"]["image_sha256"]
        assert rec["custom_id"].endswith(f"-chunk-{rec['chunk_index']}")
    # The fat request image must not be persisted
    assert "Z" * 100 not in temp_jsonl.read_text(encoding="utf-8")
    # But the API itself received the real payloads
    assert len(seen_images) == 2


@pytest.mark.asyncio
async def test_sync_strategy_streaming_producer_error_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )

    async def _process_image_chunk(**_kwargs: Any) -> dict:
        return {"output_text": "ok"}

    monkeypatch.setattr(ps, "process_image_chunk", _process_image_chunk)

    async def _source() -> Any:
        yield _dummy_payload(1)
        raise RuntimeError("PDF went away")

    strat = ps.SynchronousProcessingStrategy(concurrency_config={})
    with pytest.raises(RuntimeError, match="PDF went away"):
        await strat.process_chunks(
            chunks=["", ""],
            handler=None,
            dev_message="dev",
            model_config={"extraction_model": {"name": "gpt-5-mini"}},
            schema={"type": "object"},
            file_path=tmp_path / "doc.pdf",
            temp_jsonl_path=tmp_path / "temp.jsonl",
            console_print=lambda *_a, **_k: None,
            image_source=_source(),
        )

    # The page completed before the failure was still persisted
    content = (tmp_path / "temp.jsonl").read_text(encoding="utf-8")
    assert '"chunk_index": 1' in content


@pytest.mark.asyncio
async def test_consume_image_source_bounded_queue() -> None:
    """The producer never runs more than the queue buffer ahead."""
    in_flight_high_water = 0
    produced = 0
    consumed = 0

    async def _source() -> Any:
        nonlocal produced
        for i in range(1, 41):
            produced += 1
            yield _dummy_payload(i)

    async def _call_and_record(idx: int, chunk: str, img_data: dict) -> dict:
        nonlocal consumed, in_flight_high_water
        await asyncio.sleep(0.001)
        consumed += 1
        in_flight_high_water = max(in_flight_high_water, produced - consumed)
        return {"ok": idx}

    results = await ps.SynchronousProcessingStrategy._consume_image_source(
        image_source=_source(),
        call_and_record=_call_and_record,
        console_print=lambda *_a, **_k: None,
        concurrency_limit=2,
        delay_between_tasks=0.0,
        unit_label="page",
    )

    assert len(results) == 40
    # Buffer is 2 * concurrency_limit = 4 (+ items held by workers)
    assert in_flight_high_water <= 4 + 2 + 1


# ---------------------------------------------------------------------------
# Transient-error classification and retry
# ---------------------------------------------------------------------------

# Abridged real Cloudflare 520 error from a production run (June 2026):
# pages failed without retry because 520 matched no enumerated code.
_CLOUDFLARE_520 = (
    "Error code: 520 - {'type': 'https://developers.cloudflare.com/"
    "support/troubleshooting/http-status-codes/cloudflare-5xx-errors/"
    "error-520/', 'title': 'Error 520: Web server is returning an "
    "unknown error', 'status': 520, 'cloudflare_error': True, "
    "'retryable': True, 'retry_after': 60}"
)


def test_classify_transient_error_5xx_codes() -> None:
    from modules.extract.processing_strategy import classify_transient_error

    # Cloudflare edge codes are server errors now
    for code in (500, 502, 503, 520, 521, 522, 524, 526):
        _, _, server = classify_transient_error(f"Error code: {code} - boom")
        assert server, f"{code} should classify as server error"

    # The real production message classifies as retryable
    is_429, is_timeout, server = classify_transient_error(_CLOUDFLARE_520)
    assert server and not is_429

    # Self-declared retryability is honored even without a code match
    _, _, server = classify_transient_error("weird failure 'retryable': True")
    assert server

    # Non-retryable errors stay non-retryable
    is_429, is_timeout, server = classify_transient_error(
        "Error code: 400 - invalid request body"
    )
    assert not (is_429 or is_timeout or server)
    is_429, is_timeout, server = classify_transient_error(
        "Error code: 401 - invalid API key"
    )
    assert not (is_429 or is_timeout or server)


@pytest.mark.asyncio
async def test_streaming_retries_cloudflare_520(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 520 on the first attempt is retried and the page completes."""
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )

    attempts = 0

    async def _process_image_chunk(**_kwargs: Any) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError(_CLOUDFLARE_520)
        return {"output_text": "ok after retry"}

    monkeypatch.setattr(ps, "process_image_chunk", _process_image_chunk)

    async def _source() -> Any:
        yield _dummy_payload(1)

    strat = ps.SynchronousProcessingStrategy(
        concurrency_config={
            "concurrency": {
                "extraction": {
                    "concurrency_limit": 1,
                    "retry": {"attempts": 3, "wait_min_seconds": 0.01,
                              "wait_max_seconds": 0.02},
                }
            }
        }
    )
    results = await strat.process_chunks(
        chunks=[""],
        handler=None,
        dev_message="dev",
        model_config={"extraction_model": {"name": "gpt-5-mini"}},
        schema={"type": "object"},
        file_path=tmp_path / "doc.pdf",
        temp_jsonl_path=tmp_path / "temp.jsonl",
        console_print=lambda *_a, **_k: None,
        image_source=_source(),
    )

    assert attempts == 2
    assert results == [{"output_text": "ok after retry"}]
    assert '"chunk_index": 1' in (tmp_path / "temp.jsonl").read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# File-level concurrency cap
# ---------------------------------------------------------------------------


def test_file_concurrency_limit(tmp_path: Path) -> None:
    from main.process_text_files import _file_concurrency_limit

    text_file = tmp_path / "a.txt"
    text_file.write_text("x", encoding="utf-8")
    pdf_file = tmp_path / "b.pdf"
    _make_pdf(pdf_file, pages=1)

    cfg = {"concurrency": {"extraction": {"max_concurrent_files": 6}}}
    assert _file_concurrency_limit(cfg, [text_file]) == 6
    assert _file_concurrency_limit(cfg, [text_file, pdf_file]) == 2
    assert _file_concurrency_limit({}, [text_file]) == 4
    bad = {"concurrency": {"extraction": {"max_concurrent_files": "junk"}}}
    assert _file_concurrency_limit(bad, [text_file]) == 4


# ---------------------------------------------------------------------------
# End-to-end visual pipeline (offline)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_visual_file_end_to_end_with_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Full chain: PDF -> streaming render -> mocked API -> output JSON.

    Verifies provenance stamping, lean temp records, and that a resumed
    second run skips the file before rendering anything.
    """
    from modules.extract.file_processor import FileProcessor

    pdf_path = tmp_path / "guide.pdf"
    _make_pdf(pdf_path, pages=3)

    class _FakeLoader:
        def get_image_processing_config(self) -> dict[str, Any]:
            return _IMAGE_CONFIG

    monkeypatch.setattr(
        "modules.config.loader.get_config_loader", lambda: _FakeLoader()
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_detect_provider", staticmethod(lambda model: "openai")
    )
    monkeypatch.setattr(
        ps.ProviderConfig, "_get_api_key", staticmethod(lambda provider: "key")
    )
    monkeypatch.setattr(
        ps, "open_extractor", lambda **_kwargs: _AsyncExtractorCM(object())
    )

    call_count = 0

    async def _process_image_chunk(**_kwargs: Any) -> dict:
        nonlocal call_count
        call_count += 1
        return {
            "output_text": json.dumps({"entries": [{"name": f"r{call_count}"}]}),
            "response_data": {"usage": {}},
            "request_metadata": {"messages": []},
        }

    monkeypatch.setattr(ps, "process_image_chunk", _process_image_chunk)

    processor = FileProcessor(
        paths_config={"general": {"input_paths_is_output_path": True}},
        model_config={"extraction_model": {"name": "gpt-5-mini"}},
        chunking_config={"chunking": {}},
        concurrency_config={"concurrency": {"extraction": {"concurrency_limit": 2}}},
    )

    await processor.process_file(
        file_path=pdf_path,
        use_batch=False,
        selected_schema={"schema": {"type": "object"}},
        prompt_template="Extract. {schema}",
        schema_name="TestSchema",
        inject_schema=False,
        schema_paths={},
        resume=True,
    )

    assert call_count == 3
    output_json = tmp_path / "guide_output.json"
    data = json.loads(output_json.read_text(encoding="utf-8"))

    meta = data["_chronominer_metadata"]
    assert meta["total_chunks"] == 3
    assert "partial" not in meta
    prov = meta["image_provenance"]
    assert prov["source_sha256"] == hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    assert prov["image_config"]["target_dpi"] == 72

    records = data["records"]
    assert [r["chunk_index"] for r in records] == [1, 2, 3]
    for record in records:
        assert record["image_provenance"]["image_sha256"]
        # Output records are leaned: no request-side payloads
        assert "request_metadata" not in record["response"]

    # Second run with resume: skipped before rendering, no extra API calls
    await processor.process_file(
        file_path=pdf_path,
        use_batch=False,
        selected_schema={"schema": {"type": "object"}},
        prompt_template="Extract. {schema}",
        schema_name="TestSchema",
        inject_schema=False,
        schema_paths={},
        resume=True,
    )
    assert call_count == 3


# ---------------------------------------------------------------------------
# slim_temp_jsonl utility
# ---------------------------------------------------------------------------


def test_slim_temp_jsonl_rewrites_fat_records(tmp_path: Path) -> None:
    from main.slim_temp_jsonl import slim_file

    fat_record = {
        "custom_id": "doc-chunk-1",
        "chunk_index": 1,
        "response": {"body": _fat_result()},
    }
    batch_record = {"batch_tracking": {"batch_id": "b1"}}
    target = tmp_path / "doc_temp.jsonl"
    with target.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(fat_record) + "\n")
        fh.write(json.dumps(batch_record) + "\n")
        fh.write("not-json-line\n")

    before, after_dry = slim_file(target, dry_run=True)
    assert target.stat().st_size == before  # dry run does not modify

    before2, after = slim_file(target, dry_run=False)
    assert before2 == before
    assert after == after_dry
    assert after < before

    lines = target.read_text(encoding="utf-8").splitlines()
    slimmed = json.loads(lines[0])
    assert "A" * 100 not in lines[0]
    assert slimmed["custom_id"] == "doc-chunk-1"
    assert slimmed["response"]["body"]["output_text"] == "ok"
    assert json.loads(lines[1]) == batch_record
    assert lines[2] == "not-json-line"
