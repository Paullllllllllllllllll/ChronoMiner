"""Provider batch APIs behind one interface.

The caller builds every provider-native request body with an injected
builder and hands in a ready SDK client. A backend wraps the bodies in its
provider's batch envelope, splits them by the provider's count and byte
limits, and submits one batch per part. It then reports status, downloads
results, cancels, lists batches and deletes remote files.

A caller that tracks each part on its own (a store with one submission per
part) uses the part API instead of :meth:`BatchBackend.submit`: plan the
parts, upload one part, create its batch under the caller's submission id,
and on a failed create learn whether the provider refused it or the outcome
is unknown. :meth:`BatchBackend.list_batches` finds the batches of a
submission again.

Results carry the output text, the provider payload as a plain dict, the
provider record verbatim where the backend can read it, token usage and
error messages. Every submitted custom id without a result gets a fallback
item, so a finalizer sees each request exactly once; a record that names no
custom id, such as a malformed result line, is kept as an extra item.
"""

from __future__ import annotations

import importlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, ClassVar, Literal, TypeVar

import httpx

from .http_timeouts import build_httpx_timeout
from .responses import ResponseOutcome
from .usage import Usage, coerce_token_count

__all__ = [
    "BATCH_PROVIDERS",
    "BatchBackend",
    "BatchCreateError",
    "BatchError",
    "BatchHandle",
    "BatchListing",
    "BatchNotReadyError",
    "BatchPart",
    "BatchRequest",
    "BatchResultItem",
    "BatchStatus",
    "BatchStatusInfo",
    "BatchSubmitError",
    "BatchSummary",
    "BodyBuilder",
    "EventHooks",
    "MISSING_RESULT_ERROR",
    "PartUpload",
    "RawRecordKind",
    "ResultOrigin",
    "TERMINAL_STATUSES",
    "add_event_hooks",
    "anthropic_client",
    "as_datetime",
    "as_plain_dict",
    "count",
    "create_backend",
    "create_client",
    "get_field",
    "google_client",
    "is_definite_refusal",
    "jsonl_bytes",
    "jsonl_lines",
    "malformed_line",
    "openai_client",
    "outcome_counts",
    "sdk_http_client",
    "split_by_limits",
    "status_code_of",
    "supports_batch",
    "without_retries",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

MISSING_RESULT_ERROR = "no result returned by provider"


class BatchStatus(Enum):
    """Provider-neutral state of a batch job."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


TERMINAL_STATUSES = frozenset(
    {
        BatchStatus.COMPLETED,
        BatchStatus.FAILED,
        BatchStatus.CANCELLED,
        BatchStatus.EXPIRED,
    }
)


class BatchError(RuntimeError):
    """A batch operation failed."""


class BatchNotReadyError(BatchError):
    """Results were requested for a batch that has not finished."""


class BatchSubmitError(BatchError):
    """A part failed to submit; ``handles`` lists the parts submitted before it."""

    def __init__(self, message: str, handles: Sequence[BatchHandle]) -> None:
        super().__init__(message)
        self.handles = list(handles)


@dataclass(frozen=True)
class PartUpload:
    """Where a part's requests went before its create call.

    ``file_id`` names the provider file (an OpenAI file id, a Google file
    name); None means the requests travel inline in the create call.
    """

    provider: str
    file_id: str | None = None


class BatchCreateError(BatchError):
    """Creating the batch of one part failed; the upload is kept.

    ``definite`` is True when the provider refused the create with a 4xx
    status other than 408, so no batch exists. Otherwise (a timeout, a 408
    status, a connection error, a 5xx status or a lost response) a batch may
    exist and the caller has to reconcile before it submits the requests
    again.
    """

    def __init__(
        self, message: str, *, definite: bool, upload: PartUpload, part: int
    ) -> None:
        super().__init__(message)
        self.definite = definite
        self.upload = upload
        self.part = part

    @property
    def ambiguous(self) -> bool:
        """Whether the provider may hold a batch for the part."""
        return not self.definite


@dataclass
class BatchRequest:
    """One request of a batch; the body builder decides which fields it uses."""

    custom_id: str
    text: str = ""
    image_base64: str | None = None
    mime_type: str | None = None
    image_detail: str | None = None
    order_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_visual(self) -> bool:
        """Whether the request carries image data."""
        return self.image_base64 is not None


BodyBuilder = Callable[[BatchRequest], dict[str, Any]]


@dataclass
class BatchPart:
    """One part of a planned submission: its requests and their envelopes.

    ``index`` counts from 1 up to ``total``; ``entries`` holds the provider
    envelope of each request, in request order.
    """

    index: int
    total: int
    requests: list[BatchRequest]
    entries: list[dict[str, Any]]

    @property
    def custom_ids(self) -> list[str]:
        """The custom ids of the part's requests, in order."""
        return [request.custom_id for request in self.requests]


@dataclass
class BatchHandle:
    """A submitted batch: provider, batch id, its custom ids and provider data."""

    provider: str
    batch_id: str
    custom_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable form."""
        return {
            "provider": self.provider,
            "batch_id": self.batch_id,
            "custom_ids": list(self.custom_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BatchHandle:
        """Rebuild a handle from :meth:`to_dict` output."""
        custom_ids = data.get("custom_ids") or []
        metadata = data.get("metadata") or {}
        return cls(
            provider=str(data.get("provider") or ""),
            batch_id=str(data.get("batch_id") or ""),
            custom_ids=[str(cid) for cid in custom_ids],
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )


@dataclass
class BatchStatusInfo:
    """Status of a batch job with request counts where the provider has them."""

    status: BatchStatus
    provider_status: str = ""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    pending_requests: int = 0
    error_message: str | None = None
    results_available: bool = False
    output_file_id: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether the batch has finished (completed, failed, cancelled, expired)."""
        return self.status in TERMINAL_STATUSES

    def counts(self) -> dict[str, int]:
        """Return the request counts as stored with a submission check."""
        return {
            "total": self.total_requests,
            "completed": self.completed_requests,
            "failed": self.failed_requests,
            "pending": self.pending_requests,
        }


def outcome_counts(total: int, failed: int) -> dict[str, int]:
    """Return the counts of a finalized submission, from its stored outcomes.

    *total* is the number of requests and *failed* the units their results
    left failed (an error, a missing result, a stopped or invalid answer).
    The provider's own counts are not used here: Gemini reports none, and a
    request the provider counts as succeeded can still fail its unit.
    """
    failed = max(0, min(failed, total))
    return {"total": total, "completed": total - failed, "failed": failed, "pending": 0}


ResultOrigin = Literal["output", "error_file", "missing", "unmatched"]
RawRecordKind = Literal["line", "sdk"]
"""``'line'``: the record as downloaded (a line of a result file or body, or
an element of a captured response body); ``'sdk'``: the JSON dump of the SDK
object, where no response body was captured."""


@dataclass
class BatchResultItem:
    """Result of one request, or a provider record that names none.

    ``origin`` tells where the item came from: the provider's output, its
    error file, a fallback for a custom id the provider returned nothing
    for, or (``'unmatched'``) a record without a custom id, such as a
    malformed result line, kept with its ``raw`` and an error naming it.
    ``raw`` is the provider record as downloaded (``raw_kind`` ``'line'``),
    or else the JSON dump of the SDK object (``'sdk'``); a fallback item has
    none.

    ``outcome`` is how a returned response ended, read as
    :func:`~.responses.inspect_response` reads a synchronous one. A content
    filter, a refusal or a truncated answer fails the item (``error`` is the
    outcome's message and ``error_code`` its kind), as it fails a
    synchronous unit.
    """

    custom_id: str
    success: bool = False
    text: str = ""
    response: dict[str, Any] = field(default_factory=dict)
    usage: Usage = Usage()
    error: str | None = None
    error_code: str | None = None
    origin: ResultOrigin = "output"
    raw: str | None = None
    raw_kind: RawRecordKind | None = None
    outcome: ResponseOutcome = ResponseOutcome("ok")

    def settle(self, outcome: ResponseOutcome, empty_error: str) -> None:
        """Set ``outcome`` and the success of a returned response.

        A stopping outcome (:attr:`~.responses.ResponseOutcome.stops_page`)
        fails the item with the outcome's message; else
        the item succeeds when it has text and fails with *empty_error*
        when it has none.
        """
        self.outcome = outcome
        if outcome.stops_page:
            self.error, self.error_code = outcome.message, outcome.kind
        elif self.text:
            self.success = True
        else:
            self.error = empty_error


@dataclass(frozen=True)
class BatchSummary:
    """One batch from a provider listing."""

    provider: str
    batch_id: str
    status: BatchStatus
    provider_status: str = ""

    def handle(self) -> BatchHandle:
        """Return a handle for status, cancel and results calls."""
        return BatchHandle(provider=self.provider, batch_id=self.batch_id)


@dataclass
class BatchListing:
    """One batch from a reconciliation listing.

    ``metadata`` is the provider's batch metadata (OpenAI) and
    ``display_name`` its display name (Google); a field the provider does
    not report stays empty or None.
    """

    provider: str
    batch_id: str
    status: BatchStatus
    provider_status: str = ""
    created_at: datetime | None = None
    request_count: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    display_name: str | None = None

    def carries(self, submission_id: str) -> bool:
        """Whether the metadata or the display name names *submission_id*."""
        if self.metadata.get("submission_id") == submission_id:
            return True
        return bool(self.display_name) and submission_id in str(self.display_name)

    def handle(self) -> BatchHandle:
        """Return a handle for status, cancel and results calls."""
        return BatchHandle(provider=self.provider, batch_id=self.batch_id)


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Return ``obj[name]`` for a mapping, else ``getattr(obj, name)``."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def count(value: Any) -> int:
    """Return a reported count as a non-negative ``int`` (0 when unusable)."""
    return max(0, coerce_token_count(value) or 0)


def as_plain_dict(obj: Any) -> dict[str, Any]:
    """Convert an SDK object to a JSON-serializable dict; never raises."""
    if isinstance(obj, Mapping):
        return dict(obj)
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except Exception:  # noqa: BLE001 - fall through to the next shape
            logger.debug("model_dump failed on %s", type(obj).__name__)
        else:
            if isinstance(dumped, dict):
                return dumped
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:  # noqa: BLE001 - fall through to the repr
            logger.debug("to_dict failed on %s", type(obj).__name__)
        else:
            if isinstance(converted, dict):
                return converted
    return {"repr": repr(obj)}


def split_by_limits(
    items: Sequence[T],
    sizes: Sequence[int],
    *,
    max_count: int,
    max_bytes: int,
) -> list[list[T]]:
    """Split ``items`` into parts under a count limit and a byte limit.

    ``sizes`` gives each item's byte size. A part closes when the next item
    would exceed either limit; an item larger than ``max_bytes`` gets a part
    of its own, and the provider decides whether to accept it.
    """
    if len(items) != len(sizes):
        raise ValueError("items and sizes differ in length")
    max_count = max(1, max_count)
    max_bytes = max(1, max_bytes)
    parts: list[list[T]] = []
    current: list[T] = []
    current_bytes = 0
    for item, size in zip(items, sizes, strict=True):
        if current and (len(current) >= max_count or current_bytes + size > max_bytes):
            parts.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += size
    if current:
        parts.append(current)
    return parts


_ENVELOPE_BODY_KEYS = ("body", "params", "request")
"""Where an envelope holds its request: OpenAI, Anthropic, Google."""


def _entry_body(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the request body inside a batch envelope, or None."""
    for key in _ENVELOPE_BODY_KEYS:
        body = entry.get(key)
        if isinstance(body, Mapping):
            return dict(body)
    return None


def jsonl_bytes(entries: Sequence[Mapping[str, Any]]) -> bytes:
    """Serialize ``entries`` as UTF-8 JSONL with LF line ends."""
    return "".join(
        json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries
    ).encode("utf-8")


def jsonl_lines(text: str) -> Iterator[tuple[int, str]]:
    """Yield the numbered lines of a JSONL text without their line ends.

    Lines break at LF only (a CR before it is dropped), so a line separator
    inside a JSON string stays part of its line.
    """
    for number, line in enumerate(text.split("\n"), start=1):
        yield number, line.removesuffix("\r")


def malformed_line(number: int, line: str, source: str) -> BatchResultItem:
    """Return the item of a result line that is not a JSON object.

    It carries no custom id; *source* names the result in the error.
    """
    logger.warning("Malformed line %d of %s kept without a custom id", number, source)
    return BatchResultItem(
        custom_id="",
        error=f"Malformed line {number} of {source}: not a JSON object",
        error_code="malformed",
        origin="unmatched",
        raw=line,
        raw_kind="line",
    )


def as_datetime(value: Any) -> datetime | None:
    """Return a timezone-aware time from a datetime, Unix seconds or ISO text.

    A naive time counts as UTC; an unusable value gives None.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, int | float) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _in_window(
    created_at: datetime | None, since: datetime | None, until: datetime | None
) -> bool:
    if created_at is None:
        return True
    if since is not None and created_at < since:
        return False
    return until is None or created_at <= until


def _http_status(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value < 600:
        return value
    return None


def status_code_of(exc: BaseException) -> int | None:
    """Return the HTTP status an SDK or httpx error carries, if any."""
    for name in ("status_code", "code"):
        status = _http_status(getattr(exc, name, None))
        if status is not None:
            return status
    return _http_status(getattr(getattr(exc, "response", None), "status_code", None))


def is_definite_refusal(exc: BaseException) -> bool:
    """Whether *exc* is a 4xx refusal, after which the provider holds no batch.

    408 Request Timeout is not one: the provider may have created the batch.
    """
    status = status_code_of(exc)
    return status is not None and 400 <= status < 500 and status != 408


def without_retries(client: Any) -> Any:
    """Return a copy of an SDK client that does not retry, if it has one.

    The SDK's own retry of a create whose response was lost would create a
    second batch; clients without ``with_options`` are returned unchanged.
    """
    with_options = getattr(client, "with_options", None)
    if not callable(with_options):
        return client
    try:
        return with_options(max_retries=0)
    except TypeError:
        return client


class BatchBackend(ABC):
    """Batch operations of one provider over an injected SDK client.

    Subclasses wrap request bodies in the provider envelope, submit one part,
    map the provider status and parse result records; this class holds the
    splitting, the part bookkeeping and the fallback for missing results.
    """

    provider_name: ClassVar[str]
    default_max_requests: ClassVar[int]
    default_max_bytes: ClassVar[int]
    lists_newest_first: ClassVar[bool] = False

    def __init__(
        self,
        client: Any,
        *,
        max_requests: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self.client = client
        self.max_requests = max(1, max_requests or self.default_max_requests)
        self.max_bytes = max(1, max_bytes or self.default_max_bytes)

    def plan_parts(
        self, requests: Sequence[BatchRequest], build_body: BodyBuilder
    ) -> list[BatchPart]:
        """Wrap ``requests`` and split them into parts, as :meth:`submit` does.

        Custom ids must be unique. Ids of equal length give equal sizes, so
        requests renamed to such ids split the same way again.
        """
        if not requests:
            return []
        ids = [request.custom_id for request in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("custom ids must be unique within a submission")
        entries = [self._wrap(request, build_body(request)) for request in requests]
        sizes = [len(jsonl_bytes([entry])) for entry in entries]
        groups = split_by_limits(
            list(range(len(entries))),
            sizes,
            max_count=self.max_requests,
            max_bytes=self.max_bytes,
        )
        if len(groups) > 1:
            logger.info(
                "Splitting %d %s request(s) into %d batches",
                len(entries),
                self.provider_name,
                len(groups),
            )
        return [
            BatchPart(
                index=index,
                total=len(groups),
                requests=[requests[position] for position in group],
                entries=[entries[position] for position in group],
            )
            for index, group in enumerate(groups, start=1)
        ]

    def request_bodies(self, part: BatchPart) -> dict[str, dict[str, Any]]:
        """Return the request body of each custom id of *part*, as submitted.

        The default is the body inside each envelope: the ``body`` of an
        OpenAI line and the ``params`` of an Anthropic request. A backend
        that converts the envelopes when it submits them returns the
        converted form.
        """
        bodies: dict[str, dict[str, Any]] = {}
        for request, entry in zip(part.requests, part.entries, strict=True):
            body = _entry_body(entry)
            if body is not None:
                bodies[request.custom_id] = body
        return bodies

    def submit(
        self,
        requests: Sequence[BatchRequest],
        build_body: BodyBuilder,
        *,
        label: str = "batch",
        on_submitted: Callable[[BatchHandle], None] | None = None,
    ) -> list[BatchHandle]:
        """Submit ``requests`` as one or more batches; return one handle per batch.

        ``build_body`` turns each request into the provider-native body.
        ``on_submitted`` runs after each part, so the caller can persist the
        handle before the next upload. A failing part raises
        :class:`BatchSubmitError` carrying the handles submitted before it.
        """
        handles: list[BatchHandle] = []
        for part in self.plan_parts(requests, build_body):
            try:
                handle = self._submit_part(part.entries, label=label)
            except Exception as exc:
                raise BatchSubmitError(
                    f"{self.provider_name} batch part {part.index} of {part.total} "
                    f"failed to submit: {exc}",
                    handles,
                ) from exc
            logger.info(
                "Submitted %s batch %s (%d requests)",
                self.provider_name,
                handle.batch_id,
                len(part.entries),
            )
            handles.append(handle)
            if on_submitted is not None:
                on_submitted(handle)
        return handles

    def upload_part(
        self,
        part: BatchPart,
        *,
        submission_id: str,
        label: str = "batch",
        on_uploaded: Callable[[PartUpload], None] | None = None,
    ) -> PartUpload:
        """Upload the requests of *part* where the provider needs a file.

        ``on_uploaded`` receives the upload reference before this method
        returns, so the caller can record the file before the create call.
        An upload error propagates.
        """
        upload = self._upload_part(part, submission_id=submission_id, label=label)
        if on_uploaded is not None:
            on_uploaded(upload)
        return upload

    def create_part(
        self,
        part: BatchPart,
        upload: PartUpload,
        *,
        submission_id: str,
        label: str = "batch",
    ) -> BatchHandle:
        """Create the batch of *part* under *submission_id* and return its handle.

        A failed create raises :class:`BatchCreateError`, which tells a
        refusal from an unknown outcome; the upload is never deleted here.
        """
        handle = self._create_part(
            part, upload, submission_id=submission_id, label=label
        )
        handle.metadata.update({"submission_id": submission_id, "part": part.index})
        logger.info(
            "Created %s batch %s (submission %s, part %d of %d, %d requests)",
            self.provider_name,
            handle.batch_id,
            submission_id,
            part.index,
            part.total,
            len(part.entries),
        )
        return handle

    def submit_part(
        self,
        part: BatchPart,
        *,
        submission_id: str,
        label: str = "batch",
        on_uploaded: Callable[[PartUpload], None] | None = None,
    ) -> BatchHandle:
        """Upload *part*, report the upload, then create its batch."""
        upload = self.upload_part(
            part, submission_id=submission_id, label=label, on_uploaded=on_uploaded
        )
        return self.create_part(part, upload, submission_id=submission_id, label=label)

    def delete_upload(self, upload: PartUpload) -> None:
        """Delete the provider file of *upload*; a provider error propagates."""
        if upload.file_id:
            self._delete_upload(upload.file_id)

    def list_batches(
        self,
        *,
        submission_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        page_size: int = 100,
        max_pages: int | None = None,
    ) -> list[BatchListing]:
        """List the provider's batches across result pages, in provider order.

        *submission_id* keeps the batches whose metadata or display name
        names it. *since* and *until* bound the creation time (naive times
        count as UTC); a batch without a reported time passes the window.
        For a provider that lists newest first, paging stops at the first
        page created entirely before *since*.
        """
        since, until = as_datetime(since), as_datetime(until)
        found: list[BatchListing] = []
        cursor: str | None = None
        pages = 0
        while True:
            listings, cursor = self._list_page(page_size, cursor)
            pages += 1
            found.extend(
                listing
                for listing in listings
                if (submission_id is None or listing.carries(submission_id))
                and _in_window(listing.created_at, since, until)
            )
            if cursor is None or (max_pages is not None and pages >= max_pages):
                break
            if (
                since is not None
                and self.lists_newest_first
                and listings
                and all(
                    listing.created_at is not None and listing.created_at < since
                    for listing in listings
                )
            ):
                break
        return found

    def request_cancel(self, handle: BatchHandle) -> None:
        """Ask the provider to cancel the batch; a provider error propagates."""
        self._cancel(handle)

    def _create_error(
        self, exc: BaseException, part: BatchPart, upload: PartUpload, submission: str
    ) -> BatchCreateError:
        definite = is_definite_refusal(exc)
        outcome = "refused" if definite else "outcome unknown"
        return BatchCreateError(
            f"{self.provider_name} batch part {part.index} of {part.total} "
            f"(submission {submission}) failed to create, {outcome}: {exc}",
            definite=definite,
            upload=upload,
            part=part.index,
        )

    def _guarded_create(
        self,
        part: BatchPart,
        upload: PartUpload,
        submission_id: str,
        create: Callable[[], T],
    ) -> T:
        """Run the provider's create call; turn its failure into a create error."""
        try:
            return create()
        except Exception as exc:
            raise self._create_error(exc, part, upload, submission_id) from exc

    def _upload_part(
        self, part: BatchPart, *, submission_id: str, label: str
    ) -> PartUpload:
        """Upload one part; the default sends its requests inline."""
        return PartUpload(self.provider_name)

    def _create_part(
        self,
        part: BatchPart,
        upload: PartUpload,
        *,
        submission_id: str,
        label: str,
    ) -> BatchHandle:
        """Create one part's batch; the default submits the part's entries."""
        return self._guarded_create(
            part,
            upload,
            submission_id,
            lambda: self._submit_part(part.entries, label=label),
        )

    def _delete_upload(self, file_id: str) -> None:
        """Delete one provider file."""
        raise NotImplementedError(
            f"The {self.provider_name} batch backend keeps no uploaded files"
        )

    def _list_page(
        self, page_size: int, cursor: str | None
    ) -> tuple[list[BatchListing], str | None]:
        """Return one page of batches and the cursor of the next, if any."""
        raise NotImplementedError(
            f"The {self.provider_name} batch backend cannot list batches"
        )

    def results(self, handle: BatchHandle) -> list[BatchResultItem]:
        """Return one item per custom id of a finished batch.

        Output and error-file records come first, in provider order; the
        first record for a custom id wins. Every custom id of the handle that
        produced no record gets a failed item with origin ``missing``.
        Records without a custom id that carry ``raw`` (malformed lines)
        follow as failed items with origin ``unmatched``; such records
        without ``raw`` are dropped. Raises :class:`BatchNotReadyError` while
        the batch is still running; download errors propagate.
        """
        records, batch_error = self._fetch_results(handle)
        items: list[BatchResultItem] = []
        unmatched: list[BatchResultItem] = []
        seen: set[str] = set()
        for item in records:
            if not item.custom_id:
                if item.raw is None:
                    logger.warning(
                        "Dropping a %s result without a custom id from batch %s",
                        self.provider_name,
                        handle.batch_id,
                    )
                    continue
                item.custom_id = ""
                item.success = False
                item.origin = "unmatched"
                item.error = item.error or "Result record without a custom id"
                unmatched.append(item)
                continue
            if item.custom_id in seen:
                logger.debug("Duplicate result for %s ignored", item.custom_id)
                continue
            seen.add(item.custom_id)
            items.append(item)
        for custom_id in handle.custom_ids:
            if custom_id not in seen:
                seen.add(custom_id)
                items.append(
                    BatchResultItem(
                        custom_id=custom_id,
                        error=batch_error or MISSING_RESULT_ERROR,
                        error_code="missing",
                        origin="missing",
                    )
                )
        return items + unmatched

    def cancel(self, handle: BatchHandle) -> bool:
        """Request cancellation; return False when the provider refused."""
        try:
            self._cancel(handle)
        except Exception as exc:  # noqa: BLE001 - reported as a False result
            logger.error(
                "Failed to cancel %s batch %s: %s",
                self.provider_name,
                handle.batch_id,
                exc,
            )
            return False
        return True

    def cleanup(self, handle: BatchHandle) -> None:
        """Delete remote input and result files; never raises.

        Call it only after the results are written durably.
        """
        return None

    def diagnose_failure(self, handle: BatchHandle) -> str:
        """Return a readable reason for a batch that did not complete."""
        info = self.status(handle)
        if info.error_message:
            return info.error_message
        return f"Batch {handle.batch_id} has status {info.status.value}"

    @abstractmethod
    def status(self, handle: BatchHandle) -> BatchStatusInfo:
        """Return the batch status; a failed lookup raises the provider error.

        ``UNKNOWN`` marks a provider state the backend does not map.
        """

    @abstractmethod
    def list_active(self, *, page_size: int = 100) -> list[BatchSummary]:
        """Return the provider's batches that have not reached a terminal state."""

    @abstractmethod
    def _wrap(self, request: BatchRequest, body: dict[str, Any]) -> dict[str, Any]:
        """Return the provider envelope of one request body."""

    @abstractmethod
    def _submit_part(self, entries: list[dict[str, Any]], *, label: str) -> BatchHandle:
        """Submit one part and return its handle."""

    @abstractmethod
    def _fetch_results(
        self, handle: BatchHandle
    ) -> tuple[list[BatchResultItem], str | None]:
        """Return the parsed records and a batch-level error, if any."""

    @abstractmethod
    def _cancel(self, handle: BatchHandle) -> None:
        """Ask the provider to cancel the batch."""


BATCH_PROVIDERS: tuple[str, ...] = ("anthropic", "google", "openai")

_BACKENDS = {
    "anthropic": ("batch_anthropic", "AnthropicBatchBackend"),
    "google": ("batch_google", "GoogleBatchBackend"),
    "openai": ("batch_openai", "OpenAIBatchBackend"),
}


def supports_batch(provider: str) -> bool:
    """Whether ``provider`` has a batch backend."""
    return provider.strip().lower() in _BACKENDS


def _normalized(provider: str) -> str:
    name = provider.strip().lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Provider {provider!r} has no batch API; "
            f"batch providers: {', '.join(BATCH_PROVIDERS)}"
        )
    return name


def create_backend(
    provider: str,
    client: Any,
    *,
    max_requests: int | None = None,
    max_bytes: int | None = None,
) -> BatchBackend:
    """Return the backend of ``provider`` over ``client``."""
    module_name, class_name = _BACKENDS[_normalized(provider)]
    module = importlib.import_module(f".{module_name}", __package__)
    backend_class: type[BatchBackend] = getattr(module, class_name)
    return backend_class(client, max_requests=max_requests, max_bytes=max_bytes)


EventHooks = Mapping[str, Sequence[Callable[[Any], Any]]]
"""httpx ``event_hooks``: hook lists under ``"request"`` and ``"response"``."""


def create_client(
    provider: str,
    api_key: str,
    *,
    timeout: float | None = None,
    connect_timeout: float | None = None,
    http_client: httpx.Client | None = None,
    event_hooks: EventHooks | None = None,
) -> Any:
    """Return an SDK client for ``provider`` with an explicit key.

    ``timeout`` is the read timeout in seconds (None keeps the SDK default);
    ``connect_timeout`` applies to OpenAI and Anthropic, whose clients take
    per-phase timeouts. Google takes a single timeout. ``http_client``
    replaces the SDK's own httpx client; ``event_hooks`` are added to the
    httpx client the SDK client sends through (see :func:`add_event_hooks`).
    """
    name = _normalized(provider)
    client: Any
    if name == "openai":
        client = openai_client(
            api_key,
            timeout=timeout,
            connect_timeout=connect_timeout,
            http_client=http_client,
        )
    elif name == "anthropic":
        client = anthropic_client(
            api_key,
            timeout=timeout,
            connect_timeout=connect_timeout,
            http_client=http_client,
        )
    else:
        client = google_client(api_key, timeout=timeout, http_client=http_client)
    if event_hooks:
        add_event_hooks(http_client or client, event_hooks)
    return client


def sdk_http_client(client: Any) -> httpx.Client | None:
    """Return the sync httpx client an SDK client sends through, if found.

    OpenAI and Anthropic clients hold it as ``_client``, a google-genai
    client as ``_api_client._httpx_client``.
    """
    for path in (("_client",), ("_api_client", "_httpx_client")):
        target: Any = client
        for name in path:
            target = getattr(target, name, None)
        if isinstance(target, httpx.Client):
            return target
    return None


def add_event_hooks(client: Any, hooks: EventHooks) -> bool:
    """Add *hooks* once to an httpx client or to an SDK client's httpx client.

    Returns False when no httpx client is found (a stand-in client), in
    which case nothing changes.
    """
    target = client if isinstance(client, httpx.Client) else sdk_http_client(client)
    if target is None:
        logger.debug("No httpx client found on %s", type(client).__name__)
        return False
    merged = {event: list(found) for event, found in target.event_hooks.items()}
    for event, added in hooks.items():
        current = merged.setdefault(event, [])
        current.extend(hook for hook in added if hook not in current)
    target.event_hooks = merged
    return True


def _httpx_kwargs(
    timeout: float | None,
    connect_timeout: float | None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if timeout is not None:
        kwargs["timeout"] = (
            build_httpx_timeout(timeout)
            if connect_timeout is None
            else build_httpx_timeout(timeout, connect=connect_timeout)
        )
    if http_client is not None:
        kwargs["http_client"] = http_client
    return kwargs


def openai_client(
    api_key: str,
    *,
    timeout: float | None = None,
    connect_timeout: float | None = None,
    http_client: httpx.Client | None = None,
) -> Any:
    """Return an ``openai.OpenAI`` client."""
    from openai import OpenAI

    return OpenAI(
        api_key=api_key, **_httpx_kwargs(timeout, connect_timeout, http_client)
    )


def anthropic_client(
    api_key: str,
    *,
    timeout: float | None = None,
    connect_timeout: float | None = None,
    http_client: httpx.Client | None = None,
) -> Any:
    """Return an ``anthropic.Anthropic`` client."""
    from anthropic import Anthropic

    return Anthropic(
        api_key=api_key, **_httpx_kwargs(timeout, connect_timeout, http_client)
    )


def google_client(
    api_key: str,
    *,
    timeout: float | None = None,
    http_client: httpx.Client | None = None,
) -> Any:
    """Return a ``google.genai.Client``; the SDK takes the timeout in ms."""
    from google import genai

    http_options: Any = {}
    if timeout is not None:
        http_options["timeout"] = int(timeout * 1000)
    if http_client is not None:
        http_options["httpx_client"] = http_client
    if not http_options:
        return genai.Client(api_key=api_key)
    return genai.Client(api_key=api_key, http_options=http_options)
