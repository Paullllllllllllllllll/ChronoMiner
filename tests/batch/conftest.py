"""Fake batch APIs and store helpers of the batch tests.

:class:`FakeBackend` subclasses :class:`~chronominer.common.batch.
BatchBackend`, so the shared splitting, missing-result and cancel logic runs;
:class:`World` holds the batches of every provider. The ``world`` fixture
installs it in place of the backends :mod:`chronominer.batch.clients`
builds and sets the provider key variables. :func:`save_world` and
:func:`load_batches` keep a world's batches in a JSON file, which a child
process shares with its test.

:class:`FakeOpenAI`, :class:`FakeAnthropic` and :class:`FakeGenai` stand in
for the SDK clients under the shipped backends (``sdk`` fixture): they keep
their batches in memory, list them newest first in pages, and can lose the
response of a create after recording its batch.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import chronominer.batch.clients as clients_module
from chronominer.common.batch import (
    BatchBackend,
    BatchHandle,
    BatchListing,
    BatchPart,
    BatchRequest,
    BatchResultItem,
    BatchStatus,
    BatchStatusInfo,
    BatchSummary,
    PartUpload,
    create_backend,
)
from chronominer.common.store import Store
from chronominer.common.usage import Usage
from chronominer.extract.images import PageImage
from chronominer.extract.units import DocumentHandle
from chronominer.output.store import metadata_writer
from chronominer.settings import Settings
from chronominer.spec import ModelSpec, RunSpec, resolve_schema, resolve_spec
from tests.conftest import DEFAULT_TEST_SCHEMA

KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
}
USAGE = Usage(input_tokens=100, output_tokens=20, total_tokens=120)


def answer(text: str, *, empty: bool = False) -> str:
    """Return a schema answer with one entry naming *text*; none if *empty*."""
    return json.dumps(
        {
            "contains_no_content_of_requested_type": empty,
            "entries": [] if empty else [{"person": text, "wage": 2.5}],
        },
        ensure_ascii=False,
    )


@dataclass
class FakeBatch:
    """One submitted batch: its requests' bodies and its state.

    A batch created through the part API keeps its *submission_id* and
    *part*; the listing reports them as each provider does.
    """

    provider: str
    batch_id: str
    bodies: dict[str, dict[str, Any]]
    status: BatchStatus = BatchStatus.COMPLETED
    cancelled: bool = False
    submission_id: str | None = None
    part: int = 0


@dataclass
class World:
    """Batches and scripted results of every provider.

    *results* maps a custom id to its result item, else the answer is
    :func:`answer` of ``"text of <custom id>"``; *dropped* custom ids get no
    result. *calls* records ``(provider, operation, batch id)``. *errors*
    maps ``(provider, operation)`` to errors raised one per call, in order,
    before the operation runs.
    """

    batches: dict[str, FakeBatch] = field(default_factory=dict)
    results: dict[str, BatchResultItem] = field(default_factory=dict)
    dropped: set[str] = field(default_factory=set)
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    errors: dict[tuple[str, str], list[Exception]] = field(default_factory=dict)
    max_requests: int | None = None
    fail_submit: bool = False
    fail_list: set[str] = field(default_factory=set)
    status: BatchStatus = BatchStatus.COMPLETED
    file_uploads: bool = False
    """Part uploads create provider files (else requests travel inline)."""
    uploads: dict[str, list[str]] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)
    cleaned: list[str] = field(default_factory=list)
    lose_create: bool = False
    """A create records its batch and then loses the response."""
    hooks: dict[str, Callable[[str], None]] = field(default_factory=dict)
    """Callbacks run at the start of ``upload``, ``create`` and ``cleanup``,
    and as ``accepted`` once a create has recorded its batch."""

    def add(
        self, provider: str, *, status: BatchStatus, batch_id: str | None = None
    ) -> str:
        """Add a batch with no requests; return its id."""
        batch_id = batch_id or f"{provider}-batch-{len(self.batches) + 1}"
        self.batches[batch_id] = FakeBatch(provider, batch_id, {}, status)
        return batch_id

    def bodies(self) -> list[dict[str, Any]]:
        """Return every submitted body, in submission order."""
        return [
            body for batch in self.batches.values() for body in batch.bodies.values()
        ]

    def count(self, provider: str, operation: str) -> int:
        """Return how often *provider* was asked for *operation*."""
        return sum(1 for call in self.calls if call[:2] == (provider, operation))


class NotFoundError(LookupError):
    """The provider's 404 for a batch id it does not know."""

    status_code = 404

    def __init__(self, batch_id: str) -> None:
        super().__init__(f"Error code: 404 - batch {batch_id} not found")


class FakeBackend(BatchBackend):
    """A provider's batch API over :class:`World`."""

    provider_name = "fake"
    default_max_requests = 1000
    default_max_bytes = 10**9

    def __init__(self, world: World, provider: str) -> None:
        super().__init__(None, max_requests=world.max_requests)
        self.world = world
        self.provider = provider

    def _record(self, operation: str, batch_id: str = "") -> None:
        self.world.calls.append((self.provider, operation, batch_id))
        queued = self.world.errors.get((self.provider, operation))
        if queued:
            raise queued.pop(0)

    def _wrap(self, request: BatchRequest, body: dict[str, Any]) -> dict[str, Any]:
        return {"custom_id": request.custom_id, "body": body}

    def _submit_part(self, entries: list[dict[str, Any]], *, label: str) -> BatchHandle:
        self._record("submit")
        if self.world.fail_submit:
            raise RuntimeError("submission refused")
        batch_id = f"{self.provider}-batch-{len(self.world.batches) + 1}"
        bodies = {str(entry["custom_id"]): entry["body"] for entry in entries}
        self.world.batches[batch_id] = FakeBatch(
            self.provider, batch_id, bodies, self.world.status
        )
        return BatchHandle(
            provider=self.provider,
            batch_id=batch_id,
            custom_ids=list(bodies),
            metadata={"request_count": len(entries)},
        )

    def status(self, handle: BatchHandle) -> BatchStatusInfo:
        """Raise like the shipped backends: queued errors, then 404 when gone."""
        self._record("status", handle.batch_id)
        batch = self.world.batches.get(handle.batch_id)
        if batch is None:
            raise NotFoundError(handle.batch_id)
        return BatchStatusInfo(status=batch.status, provider_status=batch.status.value)

    def _fetch_results(
        self, handle: BatchHandle
    ) -> tuple[list[BatchResultItem], str | None]:
        self._record("results", handle.batch_id)
        batch = self.world.batches[handle.batch_id]
        items = []
        for custom_id in batch.bodies:
            if custom_id in self.world.dropped:
                continue
            scripted = self.world.results.get(custom_id)
            items.append(
                scripted
                or BatchResultItem(
                    custom_id=custom_id,
                    success=True,
                    text=answer(f"text of {custom_id}"),
                    usage=USAGE,
                )
            )
        return items, None

    def _cancel(self, handle: BatchHandle) -> None:
        self._record("cancel", handle.batch_id)
        batch = self.world.batches.get(handle.batch_id)
        if batch is None:
            raise RuntimeError(f"batch {handle.batch_id} not found")
        batch.cancelled = True
        batch.status = BatchStatus.CANCELLED

    def list_active(self, *, page_size: int = 100) -> list[BatchSummary]:
        self._record("list")
        if self.provider in self.world.fail_list:
            raise RuntimeError("listing refused")
        return [
            BatchSummary(self.provider, batch.batch_id, batch.status)
            for batch in self.world.batches.values()
            if batch.provider == self.provider
            and batch.status in (BatchStatus.PENDING, BatchStatus.IN_PROGRESS)
        ]

    def _hook(self, operation: str, argument: str) -> None:
        hook = self.world.hooks.get(operation)
        if hook is not None:
            hook(argument)

    def _upload_part(
        self, part: BatchPart, *, submission_id: str, label: str
    ) -> PartUpload:
        self._hook("upload", submission_id)
        self._record("upload", submission_id)
        if not self.world.file_uploads:
            return PartUpload(self.provider)
        file_id = f"{self.provider}-file-{len(self.world.uploads) + 1}"
        self.world.uploads[file_id] = part.custom_ids
        return PartUpload(self.provider, file_id)

    def _create_part(
        self,
        part: BatchPart,
        upload: PartUpload,
        *,
        submission_id: str,
        label: str,
    ) -> BatchHandle:
        self._hook("create", submission_id)

        def create() -> BatchHandle:
            handle = self._submit_part(part.entries, label=label)
            batch = self.world.batches[handle.batch_id]
            batch.submission_id, batch.part = submission_id, part.index
            self._hook("accepted", handle.batch_id)
            if self.world.lose_create:
                raise TimeoutError("the response of the create was lost")
            return handle

        return self._guarded_create(part, upload, submission_id, create)

    def _delete_upload(self, file_id: str) -> None:
        self._record("delete", file_id)
        self.world.deleted.append(file_id)

    def cleanup(self, handle: BatchHandle) -> None:
        self._hook("cleanup", handle.batch_id)
        self._record("cleanup", handle.batch_id)
        self.world.cleaned.append(handle.batch_id)

    def _list_page(
        self, page_size: int, cursor: str | None
    ) -> tuple[list[BatchListing], str | None]:
        """List every batch of the provider on one page.

        OpenAI batches report their submission in the metadata, Google
        batches in the display name, Anthropic batches not at all.
        """
        self._record("list_batches")
        if self.provider in self.world.fail_list:
            raise RuntimeError("listing refused")
        listings = []
        for batch in self.world.batches.values():
            if batch.provider != self.provider:
                continue
            listing = BatchListing(
                self.provider,
                batch.batch_id,
                batch.status,
                batch.status.value,
                request_count=len(batch.bodies),
            )
            if batch.submission_id and self.provider == "openai":
                listing.metadata = {
                    "submission_id": batch.submission_id,
                    "part": str(batch.part),
                }
            elif batch.submission_id and self.provider == "google":
                listing.display_name = f"batch-{batch.submission_id}-p{batch.part}"
            listings.append(listing)
        return listings, None


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    """Install :class:`World` behind the batch clients; set every key variable."""
    state = World()
    for variable in KEYS.values():
        monkeypatch.setenv(variable, f"test-{variable.lower()}")
    monkeypatch.setattr(
        clients_module, "create_client", lambda provider, key, **_: (provider, key)
    )
    monkeypatch.setattr(
        clients_module,
        "create_backend",
        lambda provider, client: FakeBackend(state, provider),
    )
    return state


DEFAULT_MODELS: Mapping[str, str] = {
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-sonnet-4-6",
    "google": "gemini-3-flash-preview",
}


def model_spec(
    provider: str = "openai", model: str | None = None, **flags: Any
) -> ModelSpec:
    """Return the run's model spec for *provider* and *model* (or its default)."""
    values: dict[str, Any] = {
        "input": Path("."),
        "schema": DEFAULT_TEST_SCHEMA,
        "provider": provider,
        "model": model or DEFAULT_MODELS[provider],
        **flags,
    }
    return resolve_spec(values).spec.model


@pytest.fixture
def settings() -> Settings:
    """Code settings."""
    return Settings()


def install_world(world: World, monkeypatch: Any) -> None:
    """Build every batch backend over *world*; *monkeypatch* sets attributes."""
    monkeypatch.setattr(
        clients_module, "create_client", lambda provider, key, **_: (provider, key)
    )
    monkeypatch.setattr(
        clients_module,
        "create_backend",
        lambda provider, client: FakeBackend(world, provider),
    )


def save_world(world: World, path: Path) -> None:
    """Write the batches of *world* to *path*."""
    data = [
        {
            "provider": batch.provider,
            "batch_id": batch.batch_id,
            "bodies": batch.bodies,
            "status": batch.status.value,
            "cancelled": batch.cancelled,
            "submission_id": batch.submission_id,
            "part": batch.part,
        }
        for batch in world.batches.values()
    ]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_batches(path: Path) -> dict[str, FakeBatch]:
    """Read the batches :func:`save_world` wrote."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        entry["batch_id"]: FakeBatch(
            provider=entry["provider"],
            batch_id=entry["batch_id"],
            bodies=entry["bodies"],
            status=BatchStatus(entry["status"]),
            cancelled=entry["cancelled"],
            submission_id=entry["submission_id"],
            part=entry["part"],
        )
        for entry in data
    }


def run_spec(root: Path, provider: str = "openai", **flags: Any) -> RunSpec:
    """Return an extract spec for *provider*'s default model."""
    values = {
        "input": root,
        "schema": DEFAULT_TEST_SCHEMA,
        "provider": provider,
        "model": DEFAULT_MODELS[provider],
        **flags,
    }
    return resolve_spec(values).spec


def page_images(count: int, *, unrendered: frozenset[int] = frozenset()) -> list[Any]:
    """Return *count* rendered PDF page payloads; *unrendered* ones failed."""
    return [
        PageImage(
            index=index,
            image_name=f"page_{index + 1:04d}",
            page_index=index,
            base64="" if index in unrendered else "UEFHRQ==",
            render_error="broken page" if index in unrendered else None,
            image_provenance={"dpi": 300},
            custom_id=f"doc-page-{index + 1}",
        )
        for index in range(count)
    ]


@dataclass(frozen=True)
class TextChunk:
    """A text chunk payload as batch submission reads it."""

    index: int
    text: str
    chunk_range: tuple[int, int]

    @property
    def label(self) -> str:
        return f"lines {self.chunk_range[0]}-{self.chunk_range[1]}"

    @property
    def source_index(self) -> str:
        return f"{self.chunk_range[0]}-{self.chunk_range[1]}"

    def provenance(self) -> dict[str, Any]:
        return {
            "custom_id": f"doc-chunk-{self.index + 1}",
            "chunk_range": list(self.chunk_range),
        }


def text_chunks(count: int, *, lines: int = 3) -> list[TextChunk]:
    """Return *count* text chunks of *lines* lines each."""
    return [
        TextChunk(
            index,
            "\n".join(f"Paid to worker {index}-{line}" for line in range(lines)),
            (index * lines + 1, (index + 1) * lines),
        )
        for index in range(count)
    ]


async def leased_document(
    store: Store,
    name: str = "doc",
    provider: str = "openai",
    *,
    kind: str = "pdf",
    input_path: Path | None = None,
) -> DocumentHandle:
    """Create document *name* in the open *store*, lease it, record its schema.

    The document's run metadata names :data:`DEFAULT_TEST_SCHEMA`, whose
    text is stored as an artifact, as an extraction records them.
    """
    doc_id = await store.ensure_document(
        name, kind=kind, input_path=str(input_path) if input_path else None
    )
    lease = await store.acquire(doc_id)
    assert lease is not None
    schema = resolve_schema(DEFAULT_TEST_SCHEMA)
    sha = await store.store_artifact("schema", schema.path.read_text(encoding="utf-8"))
    await store.write(
        lease,
        metadata_writer(store, doc_id, {"schema_name": schema.name}, schema_sha256=sha),
    )
    return DocumentHandle(
        store, doc_id, name, lease, DEFAULT_MODELS[provider], provider
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _lost(operation: str) -> httpx.ReadTimeout:
    return httpx.ReadTimeout(f"{operation} timed out")


def openai_line(custom_id: str, text: str) -> str:
    """Return an output-file line with compact separators and *text*."""
    record = {
        "id": f"resp-{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        },
        "error": None,
    }
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def openai_error_line(custom_id: str) -> str:
    """Return an error-file line of a request that failed."""
    record = {
        "custom_id": custom_id,
        "response": None,
        "error": {"code": "invalid_image", "message": "Bild unlesbar"},
    }
    return json.dumps(record, ensure_ascii=False)


class FakeOpenAI:
    """OpenAI files and batches, listed newest first in pages of *page_size*."""

    def __init__(self, page_size: int = 2) -> None:
        self.page_size = page_size
        self.data: dict[str, bytes] = {}
        self.records: list[Any] = []
        self.deleted: list[str] = []
        self.lose_create = False
        self.creates = 0
        self.on_delete: Callable[[str], None] | None = None
        self.files = SimpleNamespace(
            create=self._upload, content=self._content, delete=self._delete
        )
        self.batches = SimpleNamespace(
            create=self._create,
            retrieve=self._retrieve,
            list=self._list,
            cancel=self._cancel,
        )

    def with_options(self, **kwargs: Any) -> FakeOpenAI:
        return self

    def _upload(self, *, file: tuple[str, bytes], purpose: str) -> Any:
        file_id = f"file-{len(self.data) + 1}"
        self.data[file_id] = file[1]
        return SimpleNamespace(id=file_id)

    def _content(self, file_id: str) -> Any:
        data = self.data[file_id]
        return SimpleNamespace(read=lambda: data)

    def _delete(self, file_id: str) -> None:
        if self.on_delete is not None:
            self.on_delete(file_id)
        self.deleted.append(file_id)

    def add(self, *, status: str = "in_progress", **metadata: str) -> Any:
        """Record a batch of another submission; return it."""
        batch = SimpleNamespace(
            id=f"batch_{len(self.records) + 1}",
            status=status,
            created_at=int(_now().timestamp()),
            metadata=dict(metadata) or None,
            request_counts=SimpleNamespace(total=1, completed=0, failed=0),
            input_file_id=None,
            output_file_id=None,
            error_file_id=None,
            errors=None,
        )
        self.records.append(batch)
        return batch

    def _create(self, **kwargs: Any) -> Any:
        self.creates += 1
        lines = self.data[kwargs["input_file_id"]].decode("utf-8").splitlines()
        batch = self.add(**kwargs["metadata"])
        batch.input_file_id = kwargs["input_file_id"]
        batch.request_counts.total = len(lines)
        if self.lose_create:
            raise _lost("batches.create")
        return batch

    def _retrieve(self, batch_id: str) -> Any:
        for batch in self.records:
            if batch.id == batch_id:
                return batch
        raise LookupError(batch_id)

    def _list(self, *, limit: int, after: str | None = None) -> Any:
        newest = list(reversed(self.records))
        start = 0
        if after is not None:
            start = next(i for i, b in enumerate(newest) if b.id == after) + 1
        page = newest[start : start + self.page_size]
        return SimpleNamespace(data=page, has_more=start + self.page_size < len(newest))

    def _cancel(self, batch_id: str) -> Any:
        batch = self._retrieve(batch_id)
        batch.status = "cancelling"
        return batch

    def custom_ids(self, batch_id: str) -> list[str]:
        """Return the custom ids of a batch's input file, in order."""
        batch = self._retrieve(batch_id)
        lines = self.data[batch.input_file_id].decode("utf-8").splitlines()
        return [json.loads(line)["custom_id"] for line in lines]

    def complete(
        self, batch_id: str, *, failed: frozenset[int] = frozenset()
    ) -> tuple[bytes, bytes]:
        """Finish a batch: output and error files; return their bytes.

        The requests at the positions in *failed* go to the error file.
        """
        batch = self._retrieve(batch_id)
        ids = self.custom_ids(batch_id)
        output = "".join(
            openai_line(cid, answer(f"Zürich «{position}»")) + "\n"
            for position, cid in enumerate(ids)
            if position not in failed
        ).encode("utf-8")
        errors = "".join(
            openai_error_line(cid) + "\r\n"
            for position, cid in enumerate(ids)
            if position in failed
        ).encode("utf-8")
        batch.output_file_id = f"file-{len(self.data) + 1}"
        self.data[batch.output_file_id] = output
        if errors:
            batch.error_file_id = f"file-{len(self.data) + 1}"
            self.data[batch.error_file_id] = errors
        batch.status = "completed"
        batch.request_counts.completed = len(ids) - len(failed)
        batch.request_counts.failed = len(failed)
        return output, errors


class FakeAnthropic:
    """Anthropic message batches, listed newest first in pages of *page_size*."""

    def __init__(self, page_size: int = 2) -> None:
        self.page_size = page_size
        self.records: list[Any] = []
        self.requests: dict[str, list[dict[str, Any]]] = {}
        self.lose_create = False
        self.creates = 0
        self.messages = SimpleNamespace(
            batches=SimpleNamespace(
                create=self._create,
                retrieve=self._retrieve,
                list=self._list,
                results=self._results,
                cancel=self._retrieve,
            )
        )

    def with_options(self, **kwargs: Any) -> FakeAnthropic:
        return self

    def add(self, count: int, created: datetime | None = None) -> Any:
        """Record a batch with *count* requests; return it."""
        batch = SimpleNamespace(
            id=f"msgbatch_{len(self.records) + 1}",
            processing_status="in_progress",
            created_at=created or _now(),
            request_counts=SimpleNamespace(
                processing=count, succeeded=0, errored=0, canceled=0, expired=0
            ),
            results_url=None,
        )
        self.records.append(batch)
        return batch

    def _create(self, *, requests: list[dict[str, Any]]) -> Any:
        self.creates += 1
        batch = self.add(len(requests))
        self.requests[batch.id] = requests
        if self.lose_create:
            raise _lost("messages.batches.create")
        return batch

    def _retrieve(self, batch_id: str) -> Any:
        for batch in self.records:
            if batch.id == batch_id:
                return batch
        raise LookupError(batch_id)

    def _list(self, *, limit: int, after_id: str | None = None) -> Any:
        newest = list(reversed(self.records))
        start = 0
        if after_id is not None:
            start = next(i for i, b in enumerate(newest) if b.id == after_id) + 1
        page = newest[start : start + self.page_size]
        return SimpleNamespace(
            data=page,
            has_more=start + self.page_size < len(newest),
            last_id=page[-1].id if page else None,
        )

    def complete(self, batch_id: str) -> None:
        """End a batch with every request succeeded."""
        batch = self._retrieve(batch_id)
        count = len(self.requests[batch_id])
        batch.processing_status = "ended"
        batch.request_counts.processing = 0
        batch.request_counts.succeeded = count
        batch.results_url = f"https://example.invalid/{batch_id}/results"

    def _results(self, batch_id: str) -> list[dict[str, Any]]:
        return [
            {
                "custom_id": request["custom_id"],
                "result": {
                    "type": "succeeded",
                    "message": {
                        "content": [
                            {"type": "text", "text": answer(request["custom_id"])}
                        ],
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                        "stop_reason": "end_turn",
                    },
                },
            }
            for request in self.requests[batch_id]
        ]


class FakeGenai:
    """Gemini batches with inline results, listed in token pages."""

    def __init__(self, page_size: int = 2) -> None:
        self.page_size = page_size
        self.records: list[Any] = []
        self.sources: dict[str, list[dict[str, Any]]] = {}
        self.deleted: list[str] = []
        self.uploads: dict[str, str] = {}
        self.lose_create = False
        self.creates = 0
        self.files = SimpleNamespace(
            upload=self._upload, download=self._download, delete=self._delete
        )
        self.batches = SimpleNamespace(
            create=self._create, list=self._list, get=self._get, cancel=self._get
        )

    def _upload(self, *, file: io.BytesIO, config: dict[str, Any]) -> Any:
        name = f"files/{len(self.uploads) + 1}"
        self.uploads[name] = file.read().decode("utf-8")
        return SimpleNamespace(name=name)

    def _download(self, *, file: str) -> bytes:
        raise AssertionError("the tests read Gemini results inline")

    def _delete(self, *, name: str) -> None:
        self.deleted.append(name)

    def add(self, display_name: str) -> Any:
        """Record a batch job with *display_name*; return it."""
        job = SimpleNamespace(
            name=f"batches/{len(self.records) + 1}",
            display_name=display_name,
            state="JOB_STATE_RUNNING",
            create_time=_now(),
            dest=None,
            error=None,
        )
        self.records.append(job)
        return job

    def _create(self, *, model: str, src: Any, config: dict[str, Any]) -> Any:
        self.creates += 1
        job = self.add(config["display_name"])
        if isinstance(src, str):
            lines = [json.loads(line) for line in self.uploads[src].splitlines()]
            src = [
                {**line["request"], "metadata": {"key": line["key"]}}
                for line in lines
                if line
            ]
        self.sources[job.name] = list(src)
        if self.lose_create:
            raise _lost("batches.create")
        return job

    def _get(self, *, name: str) -> Any:
        for job in self.records:
            if job.name == name:
                return job
        raise LookupError(name)

    def _list(self, *, config: dict[str, Any]) -> Any:
        token = config.get("page_token")
        start = int(token) if token else 0
        page = self.records[start : start + self.page_size]
        following = start + self.page_size
        return SimpleNamespace(
            page=page,
            config={
                **config,
                "page_token": str(following) if following < len(self.records) else None,
            },
        )

    def complete(self, name: str) -> None:
        """Finish a job with one inline answer per request."""
        job = self._get(name=name)
        inlined = [
            {
                "metadata": {"key": request["metadata"]["key"]},
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": answer(request["metadata"]["key"])}]
                            }
                        }
                    ],
                    "usage_metadata": {
                        "prompt_token_count": 100,
                        "candidates_token_count": 20,
                    },
                },
            }
            for request in self.sources[name]
        ]
        job.state = "JOB_STATE_SUCCEEDED"
        job.dest = SimpleNamespace(file_name=None, inlined_responses=inlined)


SDK_FAKES: Mapping[str, type[Any]] = {
    "openai": FakeOpenAI,
    "anthropic": FakeAnthropic,
    "google": FakeGenai,
}


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build the shipped backends over one fake SDK client per provider."""
    clients = {provider: fake() for provider, fake in SDK_FAKES.items()}
    for variable in KEYS.values():
        monkeypatch.setenv(variable, f"test-{variable.lower()}")
    monkeypatch.setattr(
        clients_module, "create_client", lambda provider, key, **_: clients[provider]
    )
    monkeypatch.setattr(clients_module, "create_backend", create_backend)
    return clients
