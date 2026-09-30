"""Fingerprint guards and durable visual batch recovery, entirely offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import modules.extract.processing_strategy as ps
from main.check_batches import process_all_batches
from modules.batch.backends import BatchHandle, BatchStatus, BatchStatusInfo
from modules.batch.ops import _recover_missing_batch_ids, is_batch_temp_file
from modules.extract.batch_output import build_unified_batch_output
from modules.extract.file_processor import FileProcessor, _preprocess_context_image
from modules.extract.resume import (
    METADATA_KEY,
    build_temp_header,
    verify_image_settings,
)
from modules.images.native import image_settings_fingerprint
from modules.images.page_stream import build_image_provenance, resolve_target_dpi


def image_config(**overrides: Any) -> dict[str, Any]:
    return {
        "render_strategy": "direct",
        "max_pixels_per_page": 24000000,
        "api_image_processing": {
            "target_dpi": "native",
            "llm_detail": "original",
            **overrides,
        },
    }


def file_provenance(path: Path, **overrides: Any) -> dict[str, Any]:
    return build_image_provenance(
        path, image_config(**overrides), "openai", "gpt-6-astra", None
    )


@pytest.mark.asyncio
async def test_resubmission_keeps_artifact_only_batch_ids(tmp_path: Path) -> None:
    source = tmp_path / "scan.png"
    Image.new("L", (100, 150), 128).save(source)
    temp = tmp_path / "scan_temp.jsonl"
    # Crash state: the header was written and the recovery artifact names a
    # paid batch, but its tracking line never reached the temp file.
    temp.write_text(json.dumps({"batch_request": {}}) + "\n", encoding="utf-8")
    artifact = tmp_path / "scan_batch_submission_debug.json"
    artifact.write_text(
        json.dumps(
            {
                "batch_ids": ["batch-paid-1"],
                "provider": "openai",
                "batch_metadata": {"batch-paid-1": {"model": "gpt-6-astra"}},
            }
        ),
        encoding="utf-8",
    )
    backend = MagicMock(max_batch_size=50000, max_batch_bytes=10000000)
    backend.submit_batch.return_value = BatchHandle(
        provider="openai", batch_id="batch-paid-2"
    )
    with patch.object(ps, "get_batch_backend", return_value=backend):
        await ps.BatchProcessingStrategy().process_chunks(
            chunks=[""],
            handler=MagicMock(schema_name="TestSchema"),
            dev_message="Extract.",
            model_config={
                "extraction_model": {"provider": "openai", "name": "gpt-6-astra"}
            },
            schema={},
            file_path=source,
            temp_jsonl_path=temp,
            console_print=lambda *_: None,
            image_provenance=file_provenance(source),
            image_chunks=[
                {
                    "base64": "fixture",
                    "mime_type": "image/jpeg",
                    "detail": "original",
                    "image_provenance": {"image_sha256": "fixture"},
                }
            ],
        )
    recorded = json.loads(artifact.read_text(encoding="utf-8"))
    assert recorded["batch_ids"] == ["batch-paid-1", "batch-paid-2"]
    assert recorded["batch_metadata"]["batch-paid-1"] == {"model": "gpt-6-astra"}


def test_forced_batch_finalization_replaces_changed_output(tmp_path: Path) -> None:
    from modules.extract.batch_output import merge_existing_batch_output

    source = tmp_path / "source.png"
    Image.new("L", (100, 150), 128).save(source)
    old = file_provenance(source)
    new = file_provenance(source, payload_format="png")
    existing = tmp_path / "source_output.json"
    existing.write_text(
        json.dumps(
            {
                METADATA_KEY: {"image_provenance": old},
                "records": [{"custom_id": "old-page-1", "chunk_index": 1}],
            }
        ),
        encoding="utf-8",
    )
    built = {
        METADATA_KEY: {"image_provenance": new},
        "records": [{"custom_id": "new-page-1", "chunk_index": 1}],
    }
    assert merge_existing_batch_output(built, existing) is built


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_fingerprint_rejects_changed_settings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, suffix: str
) -> None:
    source = tmp_path / "source.png"
    Image.new("L", (100, 150), 128).save(source)
    provenance = file_provenance(source)
    settings = {**provenance["image_config"], "model_type": provenance["model_type"]}
    assert provenance["image_settings_fingerprint"] == image_settings_fingerprint(
        settings
    )
    assert image_settings_fingerprint(settings) == image_settings_fingerprint(
        dict(reversed(list(settings.items())))
    )
    path = tmp_path / f"run{suffix}"
    record = (
        build_temp_header(provenance)
        if suffix == ".jsonl"
        else {METADATA_KEY: {"image_provenance": provenance}}
    )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    if suffix == ".jsonl":
        assert (
            record["image_settings_fingerprint"]
            == provenance["image_settings_fingerprint"]
        )
    verify_image_settings(path, provenance)
    with pytest.raises(ValueError, match="payload_format.*--force"):
        verify_image_settings(path, file_provenance(source, payload_format="png"))
    assert "ERROR" in caplog.text
    path.write_text("{}\n", encoding="utf-8")
    verify_image_settings(path, provenance)
    assert "may mix settings" in caplog.text


@pytest.mark.parametrize("top", [None, 180])
def test_legacy_top_level_dpi(top: int | None) -> None:
    assert resolve_target_dpi({"target_dpi": top}, "openai", "gpt-6-astra") == (
        300 if top is None else top
    )
    assert (
        resolve_target_dpi(
            {"target_dpi": top, "api_image_processing": {"target_dpi": "native"}},
            "openai",
            "gpt-6-astra",
        )
        == "native"
    )


@pytest.mark.parametrize("saved_in", ["output", "temp", "part"])
@pytest.mark.asyncio
async def test_visual_resume_checks_before_render_or_recovery(
    tmp_path: Path, config_loader: Any, saved_in: str
) -> None:
    source = tmp_path / "scan.png"
    Image.new("L", (100, 150), 128).save(source)
    old = file_provenance(source)
    output = tmp_path / "scan_output.json"
    temp = tmp_path / "scan_temp.jsonl"
    saved_temp = temp.with_name("scan_temp_part1.jsonl") if saved_in == "part" else temp
    if saved_in == "output":
        data = {
            METADATA_KEY: {"image_provenance": old},
            "records": [{"custom_id": "scan-page-1"}],
        }
        output.write_text(json.dumps(data), encoding="utf-8")
    else:
        saved_temp.write_text(
            json.dumps(build_temp_header(old))
            + "\n"
            + json.dumps({"custom_id": "scan-page-1", "response": {}})
            + "\n",
            encoding="utf-8",
        )
    processor = FileProcessor(
        paths_config=config_loader.get_paths_config(),
        model_config={"extraction_model": {"name": "gpt-6-astra"}},
        chunking_config={},
        concurrency_config={},
    )
    with (
        patch.object(
            processor, "_setup_output_paths", return_value=(tmp_path, output, temp)
        ),
        patch.object(
            config_loader,
            "get_image_processing_config",
            return_value=image_config(payload_format="png"),
        ),
        patch(
            "modules.images.stream_page_payloads",
            side_effect=AssertionError("Must reject before rendering"),
        ),
    ):
        status = await processor.process_file(
            file_path=source,
            use_batch=False,
            selected_schema={"schema": {"type": "object"}},
            prompt_template="Extract.",
            schema_name="TestSchema",
            inject_schema=True,
            schema_paths={},
            global_chunking_method="auto",
            resume=True,
        )
    assert status == "failed"
    if saved_in in ("temp", "part"):
        assert json.loads(saved_temp.read_text(encoding="utf-8").splitlines()[0]) == (
            build_temp_header(old)
        )


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
@pytest.mark.asyncio
async def test_batch_manifest_survives_lost_temp_records(
    tmp_path: Path, provider: str
) -> None:
    source = tmp_path / "scan.png"
    Image.new("L", (100, 150), 128).save(source)
    provenance = file_provenance(source, payload_format="png")
    page_provenance = {"image_sha256": "fixture", "mime_type": "image/png"}
    model = {
        "openai": "gpt-6-astra",
        "anthropic": "claude-opus-5",
        "google": "gemini-3-flash-preview",
    }[provider]
    backend = MagicMock(max_batch_size=50000, max_batch_bytes=10000000)
    flushed = []
    real_fsync = ps.os.fsync

    def fsync(fd: int) -> None:
        real_fsync(fd)
        flushed.append(fd)

    def submit(*args: Any, **kwargs: Any) -> BatchHandle:
        manifests = list(tmp_path.glob("*_batch_request_manifest_*.json"))
        assert len(manifests) == 1 and flushed
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        assert manifest["image_provenance"] == provenance
        assert (
            manifest["image_settings_fingerprint"]
            == provenance["image_settings_fingerprint"]
        )
        assert manifest["requests"][0]["metadata"]["image_provenance"] == (
            page_provenance
        )
        assert "base64" not in json.dumps(manifest)
        return BatchHandle(provider=provider, batch_id="batch-fixture")

    backend.submit_batch.side_effect = submit
    temp = tmp_path / "scan_temp.jsonl"
    original_open = Path.open

    def crash_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == temp and args and args[0] == "a":
            raise RuntimeError("Synthetic crash before temp write")
        return original_open(path, *args, **kwargs)

    with (
        patch.object(ps, "get_batch_backend", return_value=backend),
        patch.object(ps.os, "fsync", side_effect=fsync),
        patch.object(Path, "open", crash_open),
        pytest.raises(RuntimeError, match="Synthetic crash"),
    ):
        await ps.BatchProcessingStrategy().process_chunks(
            chunks=[""],
            handler=MagicMock(schema_name="TestSchema"),
            dev_message="Extract.",
            model_config={"extraction_model": {"provider": provider, "name": model}},
            schema={},
            file_path=source,
            temp_jsonl_path=temp,
            console_print=lambda *_: None,
            image_provenance=provenance,
            image_chunks=[
                {
                    "base64": "fixture",
                    "mime_type": "image/png",
                    "detail": "original",
                    "image_provenance": page_provenance,
                }
            ],
        )
    recovered, recovered_provider, metadata = _recover_missing_batch_ids(
        temp, "scan", False
    )
    assert recovered == {"batch-fixture"} and recovered_provider == provider
    assert is_batch_temp_file(temp)
    verify_image_settings(temp, provenance)
    built = build_unified_batch_output(
        [{"custom_id": "scan-page-1", "response": "extracted"}],
        [
            {
                "batch_id": "batch-fixture",
                "provider": provider,
                "metadata": metadata["batch-fixture"],
            }
        ],
        schema_name="TestSchema",
    )
    assert built[METADATA_KEY]["image_provenance"] == provenance
    assert built[METADATA_KEY]["image_provenance"]["image_settings_fingerprint"]
    assert built["records"][0]["image_provenance"] == page_provenance
    # A lost manifest drops provenance but never the downloaded results.
    lost = build_unified_batch_output(
        [{"custom_id": "scan-page-1", "response": "extracted"}],
        [
            {
                "batch_id": "batch-fixture",
                "provider": provider,
                "metadata": {"request_manifest": str(tmp_path / "missing.json")},
            }
        ],
        schema_name="TestSchema",
    )
    assert len(lost["records"]) == 1
    assert not lost[METADATA_KEY].get("image_provenance")
    backend.get_status.return_value = BatchStatusInfo(status=BatchStatus.COMPLETED)
    responses = [{"custom_id": "scan-page-1", "response": "extracted"}]
    with (
        patch("main.check_batches.get_batch_backend", return_value=backend),
        patch(
            "main.check_batches.retrieve_responses_from_batch", return_value=responses
        ),
        patch("main.check_batches.get_schema_handler", return_value=MagicMock()),
    ):
        process_all_batches(
            root_folder=tmp_path,
            processing_settings={"retain_temporary_jsonl": True},
            schema_name="TestSchema",
            schema_config={},
            ui=None,
        )
    output = json.loads((tmp_path / "scan_output.json").read_text(encoding="utf-8"))
    assert output[METADATA_KEY]["image_provenance"] == provenance
    assert output["records"][0]["image_provenance"] == page_provenance


def test_context_image_remains_jpeg(tmp_path: Path, config_loader: Any) -> None:
    import base64

    source = tmp_path / "context.png"
    Image.new("L", (100, 150), 128).save(source)
    with patch.object(
        config_loader,
        "get_image_processing_config",
        return_value=image_config(payload_format="png"),
    ):
        payload = _preprocess_context_image(source, "openai", "gpt-6-astra", "original")
    assert payload["mime_type"] == "image/jpeg"
    assert base64.b64decode(payload["base64"]).startswith(b"\xff\xd8")
