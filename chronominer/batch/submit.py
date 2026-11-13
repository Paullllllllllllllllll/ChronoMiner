"""Batch submission of an item's chunks on the store.

:func:`submit_batch` is the entry point of batch extraction and
:func:`submit_units` the submission of given units, which repair reuses. A
chunk payload is a text chunk (its ``text``) or a page image (its
``base64`` data); see :func:`chunk_request`.
The payloads are taken in slices of at most :data:`SLICE_BYTES` of
requests; each slice is split into parts by the provider's count and byte
limits, and every part becomes one submission of the store: the submission and its
requests, whose custom ids are scoped to the submission id, are committed
before the upload, and each request's body as submitted (payloads redacted)
is recorded next, so finalization can store it as the request of the unit's
response row; the upload reference is recorded before the create; the
provider batch id is recorded after it. Each part is attempted once, since
a retried create could duplicate a batch.

A definite refusal marks the submission failed and deletes its upload; its
units stay pending. A create whose outcome is unknown marks the submission
unresolved and keeps its upload; its units stay blocked until ``batch
check`` adopts the provider's batch or the user abandons the submission.
Provider errors end in a :class:`SubmitOutcome` and never propagate.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from chronominer.batch.bodies import BodyPlan, build_plan
from chronominer.batch.clients import BatchClients
from chronominer.common.batch import (
    BatchBackend,
    BatchCreateError,
    BatchPart,
    BatchRequest,
    PartUpload,
    supports_batch,
)
from chronominer.common.provider_requests import RequestTooLargeError
from chronominer.common.store import (
    Lease,
    PlannedUnit,
    Store,
    Submission,
    SubmissionRejectedError,
    UnitKey,
    UnitResult,
)
from chronominer.extract.images import resolve_image_settings
from chronominer.settings import Settings
from chronominer.spec import (
    ModelSpec,
    ResolvedSchema,
    RunSpec,
    SpecError,
    resolve_schema,
)

__all__ = [
    "LABEL",
    "REQUEST_TOO_LARGE",
    "ROLE",
    "SLICE_BYTES",
    "BatchSubmissionError",
    "SubmitDocument",
    "SubmitOutcome",
    "UnitRequest",
    "chunk_request",
    "fail_oversized",
    "split_oversized",
    "submit_batch",
    "submit_units",
]

logger = logging.getLogger(__name__)

ROLE = "extract"
LABEL = "batch-extraction"
REQUEST_TOO_LARGE = "request_too_large"
"""The error type of a unit whose request exceeds a size limit of the provider."""

SubmitStatus = Literal["submitted", "refused", "unresolved"]

_PROBE_PREFIX = "0" * 8
"""Stands in for a submission id prefix while parts are planned."""

SLICE_BYTES: Final = 64 * 1024 * 1024
"""The most request bytes (image data or text) a batch submission takes at once.

A larger request is taken alone; the provider's limits on a part, all above
this figure, still apply within a slice."""


class BatchSubmissionError(RuntimeError):
    """The requests of a submission cannot be built."""


@dataclass(frozen=True)
class SubmitOutcome:
    """How a submission ended.

    ``submitted``: every part was accepted; the units wait for ``batch
    check``. ``refused``: a part was refused; its submission is failed, its
    upload deleted and its units pending, as are those of the parts after
    it. ``unresolved``: a create got no confirmed response; that part's
    units stay blocked. *submission_ids* lists every submission opened, in
    part order; *detail* says what went wrong.
    """

    status: SubmitStatus
    submission_ids: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True)
class UnitRequest:
    """One unit of a submission and the request that extracts it.

    A unit without a request (its page could not be rendered) joins the
    first part's submission without being sent, so its finalization records
    the missing result as a failed chunk.
    """

    unit: UnitKey
    request: BatchRequest | None = None


class SubmitDocument(Protocol):
    """The per-document handle :func:`submit_batch` works on.

    The store must be open with a run begun and hold *lease*.
    """

    @property
    def store(self) -> Store:
        """The output store the document's units are written to."""
        ...

    @property
    def lease(self) -> Lease:
        """The document lease held for the submission."""
        ...

    @property
    def doc_id(self) -> int:
        """The document's id in the store."""
        ...

    @property
    def name(self) -> str:
        """The document name used in errors and log messages."""
        ...

    @property
    def kind(self) -> str:
        """The kind of the units the submission creates."""
        ...


def _probe_id(position: int) -> str:
    """Return a stand-in custom id as long as the store's scoped ids."""
    return f"{_PROBE_PREFIX}-{position:05d}"


def _plan_parts(
    planner: BatchBackend, units: Sequence[UnitRequest], plan: BodyPlan
) -> list[tuple[BatchPart, list[UnitRequest]]]:
    """Split the units that carry a request into parts, in order.

    The requests are planned under stand-in ids of the store's id length,
    so renaming a part's requests to its scoped ids keeps the split. Units
    without a request join the first part. The planned envelopes are not
    kept: :func:`_scoped_part` wraps the requests again.
    """
    probes: dict[str, UnitRequest] = {}
    requests: list[BatchRequest] = []
    for unit in units:
        if unit.request is None:
            continue
        probe = _probe_id(len(requests))
        probes[probe] = unit
        requests.append(replace(unit.request, custom_id=probe))
    unsent = [unit for unit in units if unit.request is None]
    planned = []
    for part in planner.plan_parts(requests, plan.body):
        members = [probes[custom_id] for custom_id in part.custom_ids]
        if part.index == 1:
            members.extend(unsent)
        planned.append((replace(part, entries=[]), members))
    return planned


def _scoped_part(
    store: Store,
    submission_id: str,
    part: BatchPart,
    members: Sequence[UnitRequest],
    plan: BodyPlan,
    planner: BatchBackend,
) -> BatchPart:
    """Return *part* with its requests renamed to the submission's custom ids."""
    scoped = {
        unit: custom_id
        for custom_id, unit in store.submission_requests(submission_id).items()
    }
    requests = [
        replace(member.request, custom_id=scoped[member.unit])
        for member in members
        if member.request is not None
    ]
    replanned = planner.plan_parts(requests, plan.body)
    if len(replanned) != 1:
        raise BatchSubmissionError(
            f"Batch part {part.index} split again under its scoped custom ids"
        )
    return BatchPart(part.index, part.total, requests, replanned[0].entries)


async def _submit_part(
    store: Store,
    part: BatchPart,
    members: Sequence[UnitRequest],
    *,
    plan: BodyPlan,
    planner: BatchBackend,
    clients: BatchClients,
    label: str,
    opened: list[str],
) -> SubmitOutcome | None:
    """Submit one part; return the outcome that ends the submission, if any.

    The store commits the submission before :func:`upload` runs, records
    the upload reference before :func:`create` runs and the batch id after.
    """
    provider, model = plan.provider, plan.model
    stage = "open"
    scoped: list[BatchPart] = []
    uploads: list[PartUpload] = []

    async def upload(submission: Submission) -> str | None:
        nonlocal stage
        submission_id = submission.submission_id
        opened.append(submission_id)
        stage = "upload"
        final = _scoped_part(store, submission_id, part, members, plan, planner)
        scoped.append(final)
        await store.record_requests(submission_id, planner.request_bodies(final))
        uploaded = await clients.call(
            provider,
            lambda backend: backend.upload_part(
                final, submission_id=submission_id, label=label
            ),
            model=model,
            retry=False,
            label=f"upload of batch part {part.index}",
        )
        uploads.append(uploaded)
        stage = "uploaded"
        return uploaded.file_id

    async def create(submission: Submission) -> str:
        nonlocal stage
        stage = "create"
        final, uploaded = scoped[0], uploads[0]
        try:
            handle = await clients.call(
                provider,
                lambda backend: backend.create_part(
                    final, uploaded, submission_id=submission.submission_id, label=label
                ),
                model=model,
                retry=False,
                label=f"create of batch part {part.index}",
            )
        except BatchCreateError as exc:
            if exc.definite:
                raise SubmissionRejectedError(str(exc)) from exc
            raise
        except Exception as exc:
            raise SubmissionRejectedError(str(exc)) from exc
        stage = "created"
        return handle.batch_id

    try:
        submission = await store.submit(
            provider=provider,
            model=model,
            units=[member.unit for member in members],
            upload=upload,
            create=create,
            part=part.index,
        )
    except SubmissionRejectedError as exc:
        for uploaded in uploads:
            await clients.delete_upload(uploaded, model=model)
        return SubmitOutcome("refused", tuple(opened), str(exc))
    except Exception as exc:
        if stage == "upload":
            logger.warning("Upload of batch part %d failed: %s", part.index, exc)
            return SubmitOutcome("refused", tuple(opened), f"upload failed: {exc}")
        if stage != "create":
            raise
        submission_id = opened[-1]
        logger.warning(
            "Batch submission %s has no confirmed create: %s", submission_id, exc
        )
        return SubmitOutcome(
            "unresolved",
            tuple(opened),
            f"the create of batch part {part.index} got no confirmed response "
            f"({exc}); run 'chronominer batch check', which finds the "
            f"batch, or abandon submission {submission_id} with "
            f"'chronominer batch check --abandon {submission_id}'",
        )
    logger.info(
        "Submitted %s batch %s (submission %s, part %d of %d, %d unit(s))",
        provider,
        submission.batch_id,
        submission.submission_id,
        part.index,
        part.total,
        len(members),
    )
    return None


def split_oversized(
    units: Sequence[UnitRequest], plan: BodyPlan
) -> tuple[list[UnitRequest], dict[UnitKey, str]]:
    """Split off the units whose request exceeds a size limit of the provider.

    Returns the other units and the error of each unit split off
    (:class:`~chronominer.common.provider_requests.RequestTooLargeError`);
    those units are not submitted.
    """
    kept: list[UnitRequest] = []
    errors: dict[UnitKey, str] = {}
    for unit in units:
        if unit.request is not None:
            try:
                plan.body(unit.request)
            except RequestTooLargeError as exc:
                key = unit.unit
                logger.warning(
                    "%s unit %d of document %d: %s", key.kind, key.idx, key.doc_id, exc
                )
                errors[key] = str(exc)
                continue
        kept.append(unit)
    return kept, errors


async def fail_oversized(
    store: Store, errors: Mapping[UnitKey, str], *, model: str, provider: str
) -> None:
    """Store each unit of *errors* as failed with its error, keeping its provenance.

    The store must hold the lease of each unit's document.
    """
    held = store.held_leases()
    for key, error in errors.items():
        row = next(row for row in store.units(key.doc_id) if row.key == key)
        await store.write_unit(
            held[key.doc_id],
            UnitResult(
                kind=key.kind,
                idx=key.idx,
                status="failed",
                model=model,
                provider=provider,
                outcome="failed",
                error=error,
                error_type=REQUEST_TOO_LARGE,
                provenance=row.provenance,
                attempts=row.attempts,
            ),
        )


async def submit_units(
    store: Store,
    units: Sequence[UnitRequest],
    *,
    plan: BodyPlan,
    clients: BatchClients,
    label: str = LABEL,
    first_part: int = 1,
) -> SubmitOutcome:
    """Submit *units* to the batch API of *plan*'s model, one submission per part.

    The units must exist in the store, which must hold the lease of each
    unit's document. The parts are submitted in order and numbered from
    *first_part*; the first refused or unresolved part ends the submission,
    and the units of the parts after it stay pending.

    Raises:
        BatchSubmissionError: No unit has a request, or the requests cannot
            be built.
    """
    if not any(unit.request is not None for unit in units):
        raise BatchSubmissionError("No unit of the submission has a request to send")
    planner = clients.planner(plan.provider)
    try:
        parts = _plan_parts(planner, units, plan)
    except ValueError as exc:
        raise BatchSubmissionError(f"Cannot build the batch requests: {exc}") from exc
    opened: list[str] = []
    for planned, members in parts:
        part = replace(
            planned,
            index=planned.index + first_part - 1,
            total=planned.total + first_part - 1,
        )
        ended = await _submit_part(
            store,
            part,
            members,
            plan=plan,
            planner=planner,
            clients=clients,
            label=label,
            opened=opened,
        )
        if ended is not None:
            return ended
    return SubmitOutcome("submitted", tuple(opened))


def _label(payload: Any) -> str:
    name = getattr(payload, "image_name", None) or getattr(payload, "label", None)
    return str(name) if name else f"chunk_{int(payload.index) + 1:04d}"


def _source_index(payload: Any) -> int | str:
    page_index = getattr(payload, "page_index", None)
    if isinstance(page_index, int):
        return page_index
    source_index = getattr(payload, "source_index", None)
    if isinstance(source_index, int | str):
        return source_index
    return _label(payload)


def _provenance(payload: Any) -> Mapping[str, Any] | None:
    provenance = getattr(payload, "provenance", None)
    value = provenance() if callable(provenance) else provenance
    return value if isinstance(value, Mapping) and value else None


def _render_error(payload: Any) -> str | None:
    error = getattr(payload, "render_error", None)
    return str(error) if error else None


async def _record_chunks(
    document: SubmitDocument, payloads: Sequence[Any], model: str, provider: str
) -> None:
    """Create the payloads' units and record their provenance."""
    store, lease, kind = document.store, document.lease, document.kind
    await store.plan_units(
        lease,
        kind,
        [
            PlannedUnit(int(payload.index), _label(payload), _source_index(payload))
            for payload in payloads
        ],
        model=model,
        provider=provider,
    )
    for payload in payloads:
        provenance = _provenance(payload)
        if not provenance:
            continue
        await store.write_unit(
            lease,
            UnitResult(
                kind=kind,
                idx=int(payload.index),
                status="pending",
                model=model,
                provider=provider,
                label=_label(payload),
                source_index=_source_index(payload),
                provenance=provenance,
                attempts=0,
            ),
        )


def chunk_request(payload: Any, *, detail: str | None = None) -> BatchRequest | None:
    """Return the batch request of one chunk payload, or None when it has none.

    A payload with ``base64`` image data is an image chunk sent with
    *detail*; any other payload sends its ``text``. A page that could not
    be rendered, or a chunk without text, has no request.
    """
    if _render_error(payload) is not None:
        return None
    index = int(payload.index)
    image = getattr(payload, "base64", None)
    if image:
        return BatchRequest(
            custom_id=f"chunk-{index}",
            image_base64=str(image),
            mime_type=getattr(payload, "mime_type", None) or "image/jpeg",
            image_detail=detail,
            order_index=index,
        )
    text = getattr(payload, "text", None)
    if not text:
        return None
    return BatchRequest(custom_id=f"chunk-{index}", text=str(text), order_index=index)


def _schema(spec: RunSpec, schema_path: Path | None) -> ResolvedSchema:
    """Return the run's schema: the file *schema_path*, else ``--schema``.

    Raises:
        BatchSubmissionError: The schema cannot be resolved.
    """
    try:
        return resolve_schema(str(schema_path or spec.prompt.schema))
    except SpecError as exc:
        raise BatchSubmissionError(str(exc)) from exc


def _plan(
    spec: RunSpec,
    settings: Settings,
    schema: ResolvedSchema,
    context: str | None,
    context_image: Path | None,
) -> BodyPlan:
    model = spec.model
    try:
        detail = resolve_image_settings(model.provider, model.model, spec.images)[0][
            "resolved_detail"
        ]
        return build_plan(
            model,
            settings,
            schema=schema.schema,
            context=context,
            context_image=context_image,
            detail=detail,
            images=spec.images,
        )
    except (OSError, ValueError) as exc:
        raise BatchSubmissionError(f"Cannot build the batch requests: {exc}") from exc


@dataclass(frozen=True)
class _Prepared:
    """One payload with its unit request, and the size error that keeps it unsent."""

    payload: Any
    unit: UnitRequest
    oversized: str | None = None

    @property
    def sendable(self) -> bool:
        return self.unit.request is not None and self.oversized is None


def _payload_bytes(request: BatchRequest | None, plan: BodyPlan) -> int:
    """Return the bytes *request* adds to a slice.

    These are its image data or text and the context image, which every
    request of *plan* carries again.
    """
    if request is None:
        return 0
    shared = len(plan.context_image[0]) if plan.context_image is not None else 0
    return len(request.image_base64 or "") + len(request.text.encode("utf-8")) + shared


async def _each(payloads: Sequence[Any] | AsyncIterable[Any]) -> AsyncGenerator[Any]:
    """Yield *payloads*; an asynchronous source is closed when this is."""
    if not isinstance(payloads, AsyncIterable):
        for payload in payloads:
            yield payload
        return
    close = getattr(payloads, "aclose", None)
    try:
        async for payload in payloads:
            yield payload
    finally:
        if close is not None:
            await close()


async def _slices(
    payloads: AsyncIterator[Any], document: SubmitDocument, plan: BodyPlan
) -> AsyncGenerator[list[_Prepared]]:
    """Yield the payloads in slices of at most :data:`SLICE_BYTES` of requests.

    A slice ends before the request that would take it over the limit, so a
    request larger than the limit is a slice of its own. Payloads without a
    sendable request (an unrendered page, an oversized request) join the
    open slice, so every slice holds a sendable request unless no payload
    has one.
    """
    current: list[_Prepared] = []
    held = 0
    async for payload in payloads:
        key = UnitKey(document.doc_id, document.kind, int(payload.index))
        unit = UnitRequest(key, chunk_request(payload, detail=plan.detail))
        _, oversized = split_oversized([unit], plan)
        prepared = _Prepared(payload, unit, oversized.get(key))
        size = _payload_bytes(unit.request, plan)
        opened = any(member.sendable for member in current)
        if prepared.sendable and opened and held + size > SLICE_BYTES:
            yield current
            current, held = [], 0
        current.append(prepared)
        held += size
    if current:
        yield current


async def _submit_slice(
    prepared: Sequence[_Prepared],
    document: SubmitDocument,
    plan: BodyPlan,
    clients: BatchClients,
    *,
    spec: ModelSpec,
    first_part: int,
) -> SubmitOutcome:
    """Record one slice's chunks, fail its oversized ones and submit the others.

    Raises:
        BatchSubmissionError: The slice holds no request within the size
            limits; only a document's sole slice can.
    """
    if not any(member.unit.request is not None for member in prepared):
        raise BatchSubmissionError(
            f"No chunk of '{document.name}' could be prepared for batch submission"
        )
    model, provider = spec.model, spec.provider
    await _record_chunks(
        document, [member.payload for member in prepared], model, provider
    )
    oversized = {
        member.unit.unit: member.oversized
        for member in prepared
        if member.oversized is not None
    }
    await fail_oversized(document.store, oversized, model=model, provider=provider)
    if not any(member.sendable for member in prepared):
        raise BatchSubmissionError(
            f"No chunk of '{document.name}' is within the provider's size limits"
        )
    units = [member.unit for member in prepared if member.oversized is None]
    return await submit_units(
        document.store, units, plan=plan, clients=clients, first_part=first_part
    )


async def submit_batch(
    payloads: Sequence[Any] | AsyncIterable[Any],
    *,
    document: SubmitDocument,
    spec: RunSpec,
    settings: Settings,
    schema_path: Path | None = None,
    context: str | None = None,
    context_image: Path | None = None,
) -> SubmitOutcome:
    """Submit the chunks of one document to the batch API of *spec*.

    Each payload is the unit of its order index in *document*; its unit is
    created when missing and its provenance recorded. A page that could not
    be rendered is submitted without a request (see :class:`UnitRequest`);
    a chunk whose request exceeds a size limit of the provider is stored as
    failed with that error and not submitted. *schema_path* names the schema
    file; without it ``--schema`` is resolved.

    The payloads are taken, recorded and submitted in slices of at most
    :data:`SLICE_BYTES` of requests (see :func:`_slices`), each slice in
    one or more parts numbered on from the slices before; a slice is
    released before the next is taken. A refused or unresolved part ends
    the submission, and the payloads after it are not taken.

    Raises:
        BatchSubmissionError: The provider has no batch API, no chunk has a
            request within the size limits, or the requests cannot be built.
    """
    model = spec.model
    if not supports_batch(model.provider):
        raise BatchSubmissionError(
            f"Provider '{model.provider}' does not support batch processing"
        )
    plan = _plan(spec, settings, _schema(spec, schema_path), context, context_image)
    clients = BatchClients(settings, ROLE)
    opened: list[str] = []
    outcome: SubmitOutcome | None = None
    taken = _each(payloads)
    async with (
        contextlib.aclosing(taken),
        contextlib.aclosing(_slices(taken, document, plan)) as slices,
    ):
        async for prepared in slices:
            try:
                outcome = await _submit_slice(
                    prepared,
                    document,
                    plan,
                    clients,
                    spec=model,
                    first_part=len(opened) + 1,
                )
            except BatchSubmissionError as exc:
                if not opened:
                    raise
                outcome = SubmitOutcome("refused", (), str(exc))
            del prepared
            opened.extend(outcome.submission_ids)
            if outcome.status != "submitted":
                break
    if outcome is None:
        raise BatchSubmissionError(
            f"No chunk of '{document.name}' could be prepared for batch submission"
        )
    outcome = replace(outcome, submission_ids=tuple(opened))
    logger.info(
        "Batch submission of '%s' %s: %s",
        document.name,
        outcome.status,
        ", ".join(outcome.submission_ids) or "no submission",
    )
    return outcome
