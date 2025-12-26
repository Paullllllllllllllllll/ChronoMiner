from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


class DummyStrategy:
    async def process_chunks(
        self,
        *,
        chunks,
        handler,
        dev_message,
        model_config,
        schema,
        file_path,
        temp_jsonl_path,
        console_print,
    ):
        with temp_jsonl_path.open("w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks, 1):
                f.write(
                    json.dumps(
                        {
                            "custom_id": f"{file_path.stem}-chunk-{idx}",
                            "chunk_index": idx,
                            "chunk_range": [idx, idx],
                            "response": {"body": {"output_text": "ok"}},
                        }
                    )
                    + "\n"
                )
        return [{"ok": True} for _ in chunks]


class DummyHandler:
    schema_name = "TestSchema"


@pytest.mark.integration
def test_file_processor_writes_output_json_offline(tmp_path: Path, config_loader, monkeypatch):
    from modules.operations.extraction.file_processor import FileProcessorRefactored

    monkeypatch.setattr(
        "modules.operations.extraction.file_processor.create_processing_strategy",
        lambda use_batch, concurrency_config=None: DummyStrategy(),
    )
    monkeypatch.setattr(
        "modules.operations.extraction.file_processor.get_schema_handler",
        lambda schema_name: DummyHandler(),
    )

    input_file = tmp_path / "input.txt"
    input_file.write_text("hello\nworld\n", encoding="utf-8")

    paths_config = config_loader.get_paths_config()
    schemas_paths = config_loader.get_schemas_paths()
    schema_paths = schemas_paths["TestSchema"]

    fp = FileProcessorRefactored(
        paths_config=paths_config,
        model_config=config_loader.get_model_config(),
        chunking_config={"chunking": {"default_tokens_per_chunk": 10}},
        concurrency_config=config_loader.get_concurrency_config(),
    )

    async def _run():
        await fp.process_file(
            file_path=input_file,
            use_batch=False,
            selected_schema={"schema": {"type": "object"}},
            prompt_template="Schema={{TRANSCRIPTION_SCHEMA}}",
            schema_name="TestSchema",
            inject_schema=True,
            schema_paths=schema_paths,
            global_chunking_method="auto",
            context_settings={"use_additional_context": False},
            context_manager=None,
            ui=None,
        )

    asyncio.run(_run())

    output_dir = Path(schema_paths["output"])
    out_json = output_dir / f"{input_file.stem}_output.json"
    assert out_json.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data
    assert data[0]["custom_id"].startswith(input_file.stem)
