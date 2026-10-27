"""Batch submission of an item's chunks on the store.

:func:`submit_batch` is the entry point of batch extraction and
:func:`submit_units` the submission of given units, which repair reuses. A
chunk payload is a text chunk (its ``text``) or a page image (its
``base64`` data); see :func:`chunk_request`.
The requests are split into parts by the provider's count and byte limits,
and every part becomes one submission of the store: the submission and its
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

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

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
from chronominer.spec import ResolvedSchema, RunSpec, SpecError, resolve_schema

__all__ = [
    "LABEL",
    "ROLE",
    "BatchSubmissionError",
    "SubmitDocument",
    "SubmitOutcome",
    "UnitRequest",
    "chunk_request",
    "submit_batch",
    "submit_units",
]

logger = logging.getLogger(__name__)

ROLE = "extract"
LABEL = "batch-extraction"

SubmitStatus = Literal["submitted", "refused", "unresolved"]

_PROBE_PREFIX = "0" * 8
"""Stands in for a submission id prefix while parts are planned."""


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
    without a request join the first part.
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
        planned.append((part, members))
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


async def submit_units(
    store: Store,
    units: Sequence[UnitRequest],
    *,
    plan: BodyPlan,
    clients: BatchClients,
    label: str = LABEL,
) -> SubmitOutcome:
    """Submit *units* to the batch API of *plan*'s model, one submission per part.

    The units must exist in the store, which must hold the lease of each
    unit's document. The parts are submitted in order; the first refused or
    unresolved part ends the submission, and the units of the parts after
    it stay pending.

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
    for part, members in parts:
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


async def submit_batch(
    payloads: Sequence[Any],
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
    be rendered is submitted without a request (see :class:`UnitRequest`).
    *schema_path* names the schema file; without it ``--schema`` is
    resolved.

    Raises:
        BatchSubmissionError: The provider has no batch API, no chunk has a
            request, or the requests cannot be built.
    """
    model = spec.model
    if not supports_batch(model.provider):
        raise BatchSubmissionError(
            f"Provider '{model.provider}' does not support batch processing"
        )
    plan = _plan(spec, settings, _schema(spec, schema_path), context, context_image)
    requests = [chunk_request(payload, detail=plan.detail) for payload in payloads]
    if not any(request is not None for request in requests):
        raise BatchSubmissionError(
            f"No chunk of '{document.name}' could be prepared for batch submission"
        )
    await _record_chunks(document, payloads, model.model, model.provider)
    units = [
        UnitRequest(
            UnitKey(document.doc_id, document.kind, int(payload.index)), request
        )
        for payload, request in zip(payloads, requests, strict=True)
    ]
    outcome = await submit_units(
        document.store, units, plan=plan, clients=BatchClients(settings, ROLE)
    )
    logger.info(
        "Batch submission of '%s' %s: %s",
        document.name,
        outcome.status,
        ", ".join(outcome.submission_ids) or "no submission",
    )
    return outcome
