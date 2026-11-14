"""The ``_output.json`` renderer against the characterization goldens.

Run on Windows with:

    uv --directory <repo> run pytest tests/output/test_output_json.py -q

Oracle 1: a store built as an extraction writes it (units with content,
usage and custom ids, the run metadata) renders the golden
``outputs/output_json_shape`` file, compared after the goldens'
normalization (JSON text parsed, the timestamp a placeholder), and the keys
come in the old writers' order. Oracle 2: every golden ``_output.json`` is
loaded into a store (records as units with their response fields verbatim,
failed chunks as failed units, the remaining chunks pending, the metadata as
run metadata) and must render back equal. Each distinct golden content is
checked once.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chronominer.common.store import PlannedUnit, Store, UnitResult
from chronominer.output.output_json import METADATA_KEY
from chronominer.output.render import render_item
from chronominer.output.store import KIND_CHUNK, metadata_writer, open_store

CHARACTERIZATION = Path(__file__).resolve().parents[1] / "characterization"
GOLDEN = CHARACTERIZATION / "golden"
SHAPE = (
    GOLDEN / "outputs" / "output_json_shape" / "files" / "out" / "ledger_output.json"
)
FLAT_SCHEMA = CHARACTERIZATION / "schemas" / "flat_entries.json"
COMPUTED = ("total_chunks", "partial", "failed_chunks", "version")
IDENTITY = ("schema_version", "schema_sha256")
"""Keys read from the recorded schema hash and its artifact, never stored."""


def _schema_texts() -> dict[str, str]:
    """Return the characterization schema texts by their SHA-256, as CM hashes them."""
    texts: dict[str, str] = {}
    for path in (CHARACTERIZATION / "schemas").glob("*.json"):
        text = path.read_text(encoding="utf-8")
        texts[hashlib.sha256(text.encode("utf-8")).hexdigest()] = text
    return texts


def _parsed(value: Any) -> Any:
    """Return *value* with JSON text parsed, as the golden snapshots show it."""
    if isinstance(value, dict):
        return {key: _parsed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_parsed(item) for item in value]
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            return _parsed(json.loads(value))
        except json.JSONDecodeError:
            return value
    return value


def _golden(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


async def _render(store: Store, name: str, root: Path, lease: Any) -> dict[str, Any]:
    output = root / f"{name}_output.json"
    report = await render_item(store, name, (), output, lease)
    assert report.written == [output]
    return json.loads(output.read_text(encoding="utf-8"))


def _render_shape(
    root: Path, schema_text: str | None, *, artifact: bool = True
) -> dict[str, Any]:
    """Render the shape golden's document from a store built as extract builds it.

    *schema_text* None records no schema hash; with *artifact* False the
    hash is recorded but its text is not stored.
    """
    records = _golden(SHAPE)["records"]

    async def main() -> dict[str, Any]:
        async with open_store(root) as store:
            await store.begin_run("extract")
            doc_id = await store.ensure_document("ledger", kind="text")
            lease = await store.acquire(doc_id)
            assert lease is not None
            planned = [
                PlannedUnit(index, source_index="{}-{}".format(*record["chunk_range"]))
                for index, record in enumerate(records)
            ]
            await store.plan_units(
                lease, KIND_CHUNK, planned, model="gpt-4o", provider="openai"
            )
            sha: str | None = None
            if schema_text is not None:
                sha = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
                if artifact:
                    assert await store.store_artifact("schema", schema_text) == sha
            await store.write(
                lease,
                metadata_writer(
                    store,
                    doc_id,
                    {
                        "schema_name": "FlatEntries",
                        "model_name": "gpt-4o",
                        "chunking_method": "auto",
                        "chunking_text_version": 2,
                        "document_total_chunks": 4,
                    },
                    schema_sha256=sha,
                ),
            )
            for index, record in enumerate(records):
                await store.write_unit(
                    lease,
                    UnitResult(
                        kind=KIND_CHUNK,
                        idx=index,
                        status="done",
                        model="gpt-4o",
                        provider="openai",
                        content=_text(record["response"]["output_text"]),
                        usage=record["response"]["usage"],
                        provenance={"custom_id": record["custom_id"]},
                    ),
                )
            return await _render(store, "ledger", root, lease)

    rendered = asyncio.run(main())
    meta = rendered[METADATA_KEY]
    assert meta["timestamp"]
    meta["timestamp"] = "<timestamp>"
    return rendered


ORDER_AFTER_IDENTITY = [
    "model_name",
    "chunking_method",
    "total_chunks",
    "timestamp",
    "version",
    "chunking_text_version",
    "document_total_chunks",
]


def test_an_extraction_store_renders_the_golden_shape(tmp_path: Path) -> None:
    schema_text = FLAT_SCHEMA.read_text(encoding="utf-8")
    rendered = _render_shape(tmp_path, schema_text)
    meta = rendered[METADATA_KEY]

    assert _parsed(rendered) == _golden(SHAPE)
    assert list(rendered) == [METADATA_KEY, "records"]
    assert list(meta) == ["schema_name", *IDENTITY, *ORDER_AFTER_IDENTITY]
    assert meta["schema_version"] == json.loads(schema_text)["schema_version"]
    assert meta["schema_sha256"] == hashlib.sha256(schema_text.encode()).hexdigest()
    record = rendered["records"][0]
    assert list(record) == ["custom_id", "chunk_index", "chunk_range", "response"]
    assert list(record["response"]) == ["output_text", "response_data", "usage"]
    assert list(record["response"]["response_data"]) == ["usage", "model", "provider"]
    assert isinstance(record["response"]["output_text"], str)


def _without_identity(document: dict[str, Any]) -> dict[str, Any]:
    meta = {
        key: value
        for key, value in document[METADATA_KEY].items()
        if key not in IDENTITY
    }
    return {**document, METADATA_KEY: meta}


@pytest.mark.parametrize("recorded", [False, True], ids=["no-hash", "no-artifact"])
def test_a_document_without_its_schema_identity_gets_neither_key(
    tmp_path: Path, recorded: bool
) -> None:
    schema_text = FLAT_SCHEMA.read_text(encoding="utf-8") if recorded else None
    rendered = _render_shape(tmp_path, schema_text, artifact=False)

    assert _parsed(rendered) == _without_identity(_golden(SHAPE))
    assert list(rendered[METADATA_KEY]) == ["schema_name", *ORDER_AFTER_IDENTITY]


def test_a_schema_without_a_version_gets_its_hash_alone(tmp_path: Path) -> None:
    schema = json.loads(FLAT_SCHEMA.read_text(encoding="utf-8"))
    del schema["schema_version"]
    text = json.dumps(schema)
    meta = _render_shape(tmp_path, text)[METADATA_KEY]

    assert list(meta) == ["schema_name", "schema_sha256", *ORDER_AFTER_IDENTITY]
    assert meta["schema_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_stored_identity_keys_are_never_reported(tmp_path: Path) -> None:
    async def main() -> dict[str, Any]:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")
            doc_id = await store.ensure_document("ledger", kind="text")
            lease = await store.acquire(doc_id)
            assert lease is not None
            await store.plan_units(
                lease, KIND_CHUNK, [PlannedUnit(0)], model="m", provider="openai"
            )
            values = {"schema_name": "X", "schema_version": "9", "schema_sha256": "0"}
            await store.write(lease, metadata_writer(store, doc_id, values))
            await store.write_unit(
                lease, UnitResult(KIND_CHUNK, 0, "done", "m", "openai", content="{}")
            )
            return await _render(store, "ledger", tmp_path, lease)

    meta = asyncio.run(main())[METADATA_KEY]

    assert not set(IDENTITY) & set(meta)


def _golden_files() -> list[Path]:
    return sorted(GOLDEN.rglob("*_output.json"))


def _content(path: Path) -> str:
    return json.dumps(_golden(path), sort_keys=True)


def _distinct_golden_files() -> list[Path]:
    """Return one golden per content; many runs leave the same ``_output.json``."""
    seen: set[str] = set()
    distinct: list[Path] = []
    for path in _golden_files():
        content = _content(path)
        if content not in seen:
            seen.add(content)
            distinct.append(path)
    return distinct


def _unit_indices(golden: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    """Return the done, failed and pending unit indices a golden implies."""
    meta = golden[METADATA_KEY]
    done = [int(record["chunk_index"]) - 1 for record in golden["records"]]
    failed = [int(index) - 1 for index in meta.get("failed_chunks", [])]
    used = set(done) | set(failed)
    pending: list[int] = []
    candidate = 0
    while len(used) + len(pending) < int(meta["total_chunks"]):
        if candidate not in used:
            pending.append(candidate)
        candidate += 1
    return done, failed, pending


async def _seed_golden(store: Store, golden: dict[str, Any]) -> Any:
    meta = golden[METADATA_KEY]
    doc_id = await store.ensure_document("doc")
    lease = await store.acquire(doc_id)
    assert lease is not None
    done, failed, pending = _unit_indices(golden)
    planned = [PlannedUnit(index) for index in sorted({*done, *failed, *pending})]
    model = str(meta.get("model_name"))
    await store.plan_units(lease, KIND_CHUNK, planned, model=model, provider="p")
    values = {
        key: value
        for key, value in meta.items()
        if key not in COMPUTED and key not in IDENTITY
    }
    sha: str | None = None
    if "schema_sha256" in meta:
        text = _schema_texts()[meta["schema_sha256"]]
        sha = await store.store_artifact("schema", text)
    await store.write(lease, metadata_writer(store, doc_id, values, schema_sha256=sha))
    for index, record in zip(done, golden["records"], strict=True):
        response = dict(record["response"])
        output_text = response.pop("output_text")
        provenance: dict[str, Any] = {
            "custom_id": record["custom_id"],
            "chunk_range": record["chunk_range"],
            "response": response,
        }
        if "image_provenance" in record:
            provenance["image_provenance"] = record["image_provenance"]
        await store.write_unit(
            lease,
            UnitResult(
                kind=KIND_CHUNK,
                idx=index,
                status="done",
                model=model,
                provider="p",
                content=_text(output_text),
                provenance=provenance,
            ),
        )
    for index in failed:
        await store.write_unit(
            lease, UnitResult(KIND_CHUNK, index, "failed", model, "p", error="failed")
        )
    return lease


@pytest.mark.parametrize(
    "path",
    _distinct_golden_files(),
    ids=lambda path: path.relative_to(GOLDEN).as_posix(),
)
def test_every_golden_output_renders_back_from_the_store(
    tmp_path: Path, path: Path
) -> None:
    golden = _golden(path)

    async def main() -> dict[str, Any]:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")
            lease = await _seed_golden(store, golden)
            return await _render(store, "doc", tmp_path, lease)

    rendered = asyncio.run(main())

    assert _parsed(rendered) == golden


def test_the_goldens_hold_cases() -> None:
    files = _golden_files()
    assert len(files) > 100
    assert any(
        "batch_tracking" in _golden(path)[METADATA_KEY] for path in files
    ) and any("image_provenance" in _golden(path)["records"][0] for path in files)


def test_batch_tracking_is_rendered_from_the_submissions(tmp_path: Path) -> None:
    async def main() -> dict[str, Any]:
        async with open_store(tmp_path) as store:
            await store.begin_run("extract")
            doc_id = await store.ensure_document("ledger", kind="text")
            lease = await store.acquire(doc_id)
            assert lease is not None
            await store.plan_units(
                lease,
                KIND_CHUNK,
                [PlannedUnit(0), PlannedUnit(1)],
                model="gpt-4o",
                provider="openai",
            )
            await store.write(
                lease, metadata_writer(store, doc_id, {"schema_name": "FlatEntries"})
            )
            submission = await store.open_submission(
                provider="openai",
                model="gpt-4o",
                units=[store.units(doc_id)[0].key, store.units(doc_id)[1].key],
            )
            await store.record_batch(submission.submission_id, "batch-1")
            await store.mark_finalized(submission.submission_id)
            await store.write_unit(
                lease,
                UnitResult(KIND_CHUNK, 0, "done", "gpt-4o", "openai", content="{}"),
            )
            await store.write_unit(
                lease,
                UnitResult(KIND_CHUNK, 1, "failed", "gpt-4o", "openai", error="x"),
            )
            return await _render(store, "ledger", tmp_path, lease)

    meta = asyncio.run(main())[METADATA_KEY]
    tracking = meta["batch_tracking"]

    assert meta["failed_chunks"] == [2] and meta["partial"] is True
    assert tracking["batch_ids"] == ["batch-1"]
    assert tracking["provider"] == "openai"
    assert tracking["parts"] == 1 and tracking["completed_batches"] == 1
    assert tracking["fully_completed"] is True
    assert tracking["tracking"][0]["schema_name"] == "FlatEntries"
    assert tracking["tracking"][0]["request_count"] == 2
