"""Run a classify command: ask the decision model and record its answers.

Classification is shadow only: it records answers in the ``classifications``
component of each output root's store and never changes an extraction. Per
document, the units (:mod:`chronominer.classify.units`) are grouped in order
into requests by :func:`~chronominer.common.decision_runner.plan_requests`
(entries never across chunks) and asked through the run's
:class:`~chronominer.llm.decision_model.DecisionModel` under the role
``classify``. A request whose every unit has a stored answer to every
question, by the same fingerprint, input, content and request, is reused
(:func:`~chronominer.common.classify_store.reusable_answers`) and appended
again under this run (:func:`~chronominer.common.classify_store.republish_answers`);
``--force`` asks again. Answers are written with
:func:`~chronominer.common.classify_store.record_answers`: chunk and
document rows without a lease and without a document row, entry rows under
the document's lease. A stored document whose input changed refuses the item
(:class:`~chronominer.common.classify_store.ChangedInputError`).

A dry run builds the same units and requests, reads which would be reused and
opens no store for writing and no client.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx

from chronominer.classify.plan import (
    BLOCKED,
    FINISHED,
    READY,
    REFUSED,
    ROLE,
    SKIPPED,
    ClassifyPlan,
    ClassifyTarget,
    stored_images,
)
from chronominer.classify.units import (
    DOCUMENT_LOCATOR,
    SidecarMissingError,
    entry_units,
    page_count,
    page_units,
    text_chunk_unit,
    text_document_unit,
    text_item_chunks,
)
from chronominer.common.classify_store import (
    ChangedInputError,
    DocumentChangedError,
    RowContext,
    answer_rows,
    record_answers,
    republish_answers,
    reusable_answers,
)
from chronominer.common.contract import ExitCode, exit_code_for
from chronominer.common.decision_runner import Request, Unit, plan_requests
from chronominer.common.decisions import (
    MAX_QUESTIONS,
    Image,
    Json,
    MalformedAnswer,
    Part,
    dialect_for,
    request_fingerprint,
)
from chronominer.common.sqlite import read
from chronominer.common.store import (
    Lease,
    LeaseKeeper,
    LeaseLostError,
    Store,
    StoreError,
    file_sha256,
)
from chronominer.common.usage import Usage
from chronominer.events import ItemFinished, ItemStarted, NullEvents, RunEvents
from chronominer.extract.chunks import TextChunk
from chronominer.extract.images import resolve_image_settings
from chronominer.llm.caller import CallEnv
from chronominer.llm.decision_model import DecisionModel
from chronominer.output.store import (
    database_path,
    document_metadata,
    image_settings_record,
    lease_reason,
    open_store,
)
from chronominer.providers import infer_provider, split_provider_prefix
from chronominer.ranges.lines import read_lines
from chronominer.run import ConfigurationError, StoreRefusedError, writer_options
from chronominer.spec import ClassifyItemsSpec

__all__ = [
    "CLASSIFIED",
    "FAILED",
    "ClassifyResult",
    "TargetResult",
    "run_classify",
]

CLASSIFIED: Final = "classified"
FAILED: Final = "failed"

_ERRORS_SHOWN: Final = 3
"""The errors of a document its result names."""


@dataclass(frozen=True)
class TargetResult:
    """The outcome of one document.

    *state* is ``classified``, ``failed`` (a request or a page failed),
    ``refused``, ``blocked`` or ``skipped``; *asked* and *reused* count the
    units asked (in a dry run: that would be asked) and reused, *failed* the
    units of failed requests and pages. *reason* says why a document was not
    classified, or names its first errors.
    """

    name: str
    document: str
    state: str
    asked: int = 0
    reused: int = 0
    failed: int = 0
    reason: str = ""


@dataclass(frozen=True)
class ClassifyResult:
    """The outcome of a classify command.

    *usage* holds the tokens of the decision-model calls per role and
    *databases* the database files opened for writing.
    """

    targets: tuple[TargetResult, ...]
    usage: Mapping[str, Usage] = field(default_factory=dict)
    databases: tuple[str, ...] = ()
    seconds: float = 0.0

    def count(self, state: str) -> int:
        """Return the number of documents in *state*."""
        return sum(1 for target in self.targets if target.state == state)

    @property
    def total_usage(self) -> Usage:
        """The tokens of all roles together."""
        total = Usage()
        for usage in self.usage.values():
            total = total + usage
        return total

    @property
    def asked(self) -> int:
        """The units asked about."""
        return sum(target.asked for target in self.targets)

    @property
    def reused(self) -> int:
        """The units whose stored answers were reused."""
        return sum(target.reused for target in self.targets)

    @property
    def exit_code(self) -> ExitCode:
        """1 when a document failed, was refused or blocked by a lease, else 0."""
        held = self.count(FAILED) + self.count(REFUSED) + self.count(BLOCKED)
        return exit_code_for(failed=held)


class _NotClassified(Exception):
    """A document left out at run time, with its state and reason."""

    def __init__(self, state: str, reason: str) -> None:
        super().__init__(reason)
        self.state = state


@dataclass
class _Tally:
    asked: int = 0
    reused: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class _Stores:
    """The stores of a run, one per output root, opened on first use.

    Opening a store migrates it, records the run and starts the heartbeat
    that renews the run's leases on entry documents.
    """

    def __init__(self, plan: ClassifyPlan, events: RunEvents) -> None:
        self._plan = plan
        self._events = events
        self._stores: dict[Path, Store] = {}
        self._keepers: list[LeaseKeeper] = []

    @property
    def databases(self) -> tuple[str, ...]:
        """The database files of the opened stores, in opening order."""
        return tuple(str(store.path.resolve()) for store in self._stores.values())

    async def get(self, root: Path) -> Store:
        """Return the open store of output root *root*.

        Raises:
            StoreRefusedError: The root's database is refused.
        """
        key = root.resolve()
        store = self._stores.get(key)
        if store is not None:
            return store
        key.mkdir(parents=True, exist_ok=True)
        store = open_store(key, writer=writer_options(self._events))
        try:
            await store.open()
        except StoreError as exc:
            raise StoreRefusedError(str(exc)) from exc
        self._stores[key] = store
        await store.begin_run(self._plan.command, spec=self._plan.effective_spec())
        keeper = store.heartbeat(on_lost=self._lost)
        keeper.start()
        self._keepers.append(keeper)
        return store

    def _lost(self, lease: Lease) -> None:
        self._events.warning(
            f"The lease on document {lease.doc_id} expired; its remaining answers "
            "are not stored by this run."
        )

    async def close(self, exit_code: int) -> None:
        """Stop the heartbeats, record the end of the run and close the stores."""
        for keeper in self._keepers:
            await keeper.stop()
        for store in self._stores.values():
            with contextlib.suppress(sqlite3.Error):
                if store.run_id is not None:
                    await store.end_run(exit_code)
            with contextlib.suppress(sqlite3.Error):
                await store.close()


type _Seen = tuple[int, int] | None


def _bind_input(target: ClassifyTarget) -> _Seen:
    """Refuse an input whose content changed since planning; return its stat.

    The size and modification time are read before the hash, so a later
    :func:`_check_input` that sees them unchanged vouches for the bytes read
    in between.
    """
    assert target.path is not None
    try:
        stat = target.path.stat()
        seen: _Seen = (stat.st_size, stat.st_mtime_ns)
        sha = file_sha256(target.path)
    except OSError:
        seen, sha = None, None
    if sha is None or sha != target.input_sha256:
        raise _NotClassified(
            REFUSED,
            f"its input {target.path.name} changed since planning; classify it again",
        )
    return seen


def _check_input(target: ClassifyTarget, seen: _Seen) -> None:
    """Hash the input again when its size or modification time changed."""
    assert target.path is not None
    try:
        stat = target.path.stat()
        now: _Seen = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        now = None
    if now is None or now != seen:
        _bind_input(target)


def _model_of(name: str) -> tuple[str, str]:
    """Return the provider and model name of a ``--model`` value."""
    prefix, model = split_provider_prefix(name)
    return prefix or infer_provider(model), model


class _Classify:
    """Classify the targets of one plan; without a model, only count (dry run)."""

    def __init__(
        self,
        plan: ClassifyPlan,
        model: DecisionModel | None,
        stores: _Stores | None,
        events: RunEvents,
    ) -> None:
        self.plan = plan
        self.classifier = plan.classifier
        self.model = model
        self.stores = stores
        self.events = events
        decisions = plan.settings.decisions
        self.budget = decisions.pages_per_request
        self.max_px = decisions.image_max_px
        self.questions = frozenset(q.name for q in self.classifier.questions)
        self.size = min(
            self.classifier.per_request,
            MAX_QUESTIONS // len(self.classifier.questions),
        )

    def _fingerprint(self, upstream: Mapping[str, Any] | str) -> str:
        """Return the request fingerprint; *upstream* is the page record or a kind."""
        choice = self.plan.spec.classify
        entry = self.classifier.unit == "entry"
        pages = not isinstance(upstream, str)
        return request_fingerprint(
            tool="chronominer",
            dialect=dialect_for(choice.model, choice.provider),
            model=choice.model,
            endpoint=choice.endpoint,
            classifier=self.classifier.sha256,
            unit=self.classifier.unit,
            image_max_px=self.max_px if pages else None,
            context="document+schema" if entry else "document",
            upstream=upstream if isinstance(upstream, str) else dict(upstream),
        )

    def _context_parts(self, target: ClassifyTarget) -> tuple[Part, ...]:
        if self.classifier.unit == "entry":
            data = {"document": target.document, "schema": self.classifier.schema}
            return (Json(data),)
        return (Json({"document": target.document}),)

    async def _store(self, root: Path) -> Store | None:
        return await self.stores.get(root) if self.stores is not None else None

    def _row_context(
        self, target: ClassifyTarget, store: Store | None, upstream: Any
    ) -> RowContext:
        entry = self.classifier.unit == "entry"
        return RowContext(
            document=target.document,
            fingerprint=self._fingerprint(upstream),
            doc_id=target.doc_id if entry else None,
            input_sha256=target.input_sha256,
            run_id=store.run_id if store is not None else None,
            seen_doc_id=None if entry else target.seen_doc_id,
        )

    async def target(
        self, target: ClassifyTarget, index: int, total: int
    ) -> TargetResult:
        """Classify one target; a target not ready keeps its state and reason."""
        if target.state != READY:
            return TargetResult(
                target.name, target.document, target.state, reason=target.reason
            )
        self.events.item_started(ItemStarted(target.name, target.kind, index, total))
        try:
            if self.classifier.unit == "entry":
                tally = await self._entries(target)
            else:
                tally = await self._item(target)
        except _NotClassified as exc:
            result = TargetResult(
                target.name, target.document, exc.state, reason=str(exc)
            )
        except DocumentChangedError as exc:
            result = TargetResult(target.name, target.document, FAILED, reason=str(exc))
        except ChangedInputError as exc:
            result = TargetResult(
                target.name, target.document, REFUSED, reason=str(exc)
            )
        except (LeaseLostError, SidecarMissingError, ValueError) as exc:
            reason = str(exc) or type(exc).__name__
            result = TargetResult(target.name, target.document, FAILED, reason=reason)
        else:
            result = TargetResult(
                target.name,
                target.document,
                FAILED if tally.failed else CLASSIFIED,
                tally.asked,
                tally.reused,
                tally.failed,
                "; ".join(tally.errors[:_ERRORS_SHOWN]),
            )
        success = result.state in (CLASSIFIED, SKIPPED)
        if result.state in (BLOCKED, SKIPPED, REFUSED):
            self.events.warning(
                f"Not classified ({result.state}): '{target.name}': {result.reason}"
            )
        self.events.item_finished(
            ItemFinished(
                target.name, index, total, success, error=result.reason or None
            )
        )
        return result

    async def _item(self, target: ClassifyTarget) -> _Tally:
        spec = self.plan.spec
        assert isinstance(spec, ClassifyItemsSpec)
        assert target.path is not None
        store = await self._store(target.root)
        tally = _Tally()
        seen = _bind_input(target)
        if target.kind == "text":
            lines = read_lines(target.path)
            _check_input(target, seen)
            chunks = text_item_chunks(
                target.path,
                lines,
                stored=target.stored,
                mode=spec.chunking.mode,
                chunk_tokens=spec.chunking.chunk_tokens,
                model=spec.model,
            )
            units = self._text_units(chunks, spec.model)
            context = self._row_context(target, store, "text")
            await self._ask_units(
                target, units, context, store, None, tally, images=False
            )
            return tally
        return await self._pages(target, spec, store, tally, seen)

    async def _pages(  # noqa: PLR0913 - the item's context
        self,
        target: ClassifyTarget,
        spec: ClassifyItemsSpec,
        store: Store | None,
        tally: _Tally,
        seen: _Seen,
    ) -> _Tally:
        """Classify the pages of an image or PDF item, or its document unit."""
        assert target.path is not None
        img_cfg, model_type, section, record = self._image_settings(target, spec)
        total = page_count(target.path, target.kind)
        if self.classifier.unit == "chunk":
            indices = self._selected(total)
        else:
            selection = self.classifier.selection
            first = selection.first_pages if selection is not None else total
            indices = list(range(min(first, total)))
        context = self._row_context(target, store, record)
        batch: list[Unit] = []
        images: list[Part] = []
        prepared = 0
        pages = page_units(
            target.path,
            target.kind,
            indices,
            img_cfg=img_cfg,
            model_type=model_type,
            section=section,
            max_px=self.max_px,
        )
        unprepared = 0
        while True:
            try:
                page = await anext(pages)
            except StopAsyncIteration:
                break
            except RuntimeError as exc:  # the page source's failure-rate check
                tally.failed += len(indices) - prepared - unprepared
                tally.errors.append(str(exc) or type(exc).__name__)
                return tally
            if page.unit is None or page.image is None:
                unprepared += 1
                tally.failed += 1
                tally.errors.append(f"page {page.index + 1}: {page.error}")
                continue
            prepared += 1
            if self.classifier.unit == "document":
                images.append(Image(page.image))
                continue
            batch.append(page.unit)
            if len(batch) == self._group(images=True):
                _check_input(target, seen)
                await self._ask_units(
                    target, batch, context, store, None, tally, images=True
                )
                batch = []
        _check_input(target, seen)
        if self.classifier.unit == "document":
            # The unit is the whole selection: a missing page fails it unasked.
            whole = images and not tally.failed
            batch = [Unit(DOCUMENT_LOCATOR, 0, tuple(images))] if whole else []
        if batch:
            await self._ask_units(
                target, batch, context, store, None, tally, images=True
            )
        if not prepared and not tally.failed:
            raise _NotClassified(SKIPPED, "it has no pages to classify")
        return tally

    def _text_units(self, chunks: Sequence[TextChunk], model: str) -> list[Unit]:
        if not chunks:
            raise _NotClassified(SKIPPED, "it has no text chunks to classify")
        if self.classifier.unit == "chunk":
            return [text_chunk_unit(chunks[i]) for i in self._selected(len(chunks))]
        selection = self.classifier.selection
        assert selection is not None
        return [text_document_unit(chunks, selection.max_tokens, model)]

    def _selected(self, total: int) -> list[int]:
        pages = self.plan.pages
        if pages is None:
            return list(range(total))
        selected = pages.resolve(total)
        if not selected:
            raise _NotClassified(
                SKIPPED,
                f"--pages '{pages.describe()}' selects none of its {total} chunks",
            )
        return selected

    def _image_settings(
        self, target: ClassifyTarget, spec: ClassifyItemsSpec
    ) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        """Return the page settings extract renders the target's pages with.

        A document with stored pages renders with the stored model and every
        stored image setting, detail included; the plan refused options that
        conflict with them.

        Raises:
            ValueError: An image setting is invalid.
        """
        recorded = target.image_settings
        images = spec.images.over(stored_images(recorded)) if recorded else spec.images
        provider, model = _model_of(spec.model)
        if recorded is not None:
            stored_provider = recorded.get("provider")
            stored_model = recorded.get("model_name")
            if isinstance(stored_provider, str) and isinstance(stored_model, str):
                provider, model = stored_provider, stored_model
        side = self.plan.settings.images.original_max_side_px
        if recorded is not None:  # the stored pages' limit, not today's setting
            stored_side = recorded.get("original_max_side_px")
            side = stored_side if isinstance(stored_side, int) else None
        img_cfg, model_type, section = resolve_image_settings(
            provider, model, images, side
        )
        return img_cfg, model_type, section, image_settings_record(img_cfg, model_type)

    async def _entries(self, target: ClassifyTarget) -> _Tally:
        doc_id = target.doc_id
        fields = self.classifier.fields
        assert doc_id is not None and fields is not None
        store = await self._store(target.root)
        lease: Lease | None = None
        if store is not None:
            try:
                lease = await store.acquire(doc_id)
            except KeyError:
                raise _NotClassified(SKIPPED, "it was dropped since planning") from None
            if lease is None:
                state = store.resume_state(doc_id)
                holder = state.lease_run_id if state is not None else None
                until = state.lease_until if state is not None else None
                raise _NotClassified(BLOCKED, lease_reason(holder, until))
        try:
            if store is not None:
                self._still_planned(store, target, doc_id)
            groups = read(
                database_path(target.root),
                lambda connection: entry_units(connection, doc_id, fields),
            )
            if not groups:
                raise _NotClassified(SKIPPED, "it has no entries")
            tally = _Tally()
            context = self._row_context(target, store, "entries")
            for units in groups:
                await self._ask_units(
                    target, units, context, store, lease, tally, images=False
                )
            return tally
        finally:
            if store is not None and lease is not None:
                with contextlib.suppress(sqlite3.Error):
                    await store.release(lease)

    def _still_planned(self, store: Store, target: ClassifyTarget, doc_id: int) -> None:
        """Check, under the lease, that the document still fits the plan.

        Raises:
            _NotClassified: Refused when its input changed since planning,
                skipped when it was dropped, is unfinished or was extracted
                with another schema since.
        """
        row = store.document(doc_id)
        if row is None or row.name != target.document:
            raise _NotClassified(SKIPPED, "it was dropped or renamed since planning")
        if row.input_sha256 != target.input_sha256:
            raise _NotClassified(REFUSED, "its input changed since planning")
        if row.status not in FINISHED:
            reason = f"its extraction is not finished ({row.status})"
            raise _NotClassified(SKIPPED, reason)
        meta, _sha = read(store.path, lambda c: document_metadata(c, doc_id))
        if meta.get("schema_name") != self.plan.schema:
            raise _NotClassified(
                SKIPPED, "it was extracted with another schema since planning"
            )

    def _group(self, *, images: bool) -> int:
        return min(self.size, self.budget) if images else self.size

    async def _ask_units(  # noqa: PLR0913 - one document's request context
        self,
        target: ClassifyTarget,
        units: Sequence[Unit],
        context: RowContext,
        store: Store | None,
        lease: Lease | None,
        tally: _Tally,
        *,
        images: bool,
    ) -> None:
        """Ask about *units* in requests of at most one group each."""
        size = self._group(images=images)
        parts = self._context_parts(target)
        for start in range(0, len(units), size):
            for request in plan_requests(
                self.classifier,
                target.document,
                units[start : start + size],
                context=parts,
                image_budget=self.budget,
            ):
                await self._request(target.root, request, context, store, lease, tally)

    async def _request(  # noqa: PLR0913 - one request's context
        self,
        root: Path,
        request: Request,
        context: RowContext,
        store: Store | None,
        lease: Lease | None,
        tally: _Tally,
    ) -> None:
        """Reuse the stored answers of *request*, else ask and record them."""
        count = len(request.units)
        try:
            if not self.plan.force:
                found = self._reusable(database_path(root), request, context)
                if found is not None:
                    if store is not None:
                        for unit, answers in zip(request.units, found, strict=True):
                            await republish_answers(
                                store,
                                answers,
                                run_id=context.run_id,
                                lease=lease,
                                locator=unit.locator,
                                ordinal=unit.ordinal,
                                seen_doc_id=context.seen_doc_id,
                            )
                    tally.reused += count
                    return
            if self.model is None or store is None:
                tally.asked += count
                return
            first, last = request.units[0].locator, request.units[-1].locator
            label = f"{request.document} {self.classifier.name} {first}..{last}"
            result = await self.model.runner.ask(request, label=label)
            rows = answer_rows(result, self.classifier, context, store.clock())
            await record_answers(
                store, rows, lease=lease, seen_doc_id=context.seen_doc_id
            )
            tally.asked += count
        except (httpx.HTTPError, MalformedAnswer, OSError) as exc:
            tally.failed += count
            units = f"{request.units[0].locator}..{request.units[-1].locator}"
            message = f"{units}: {str(exc) or type(exc).__name__}"
            tally.errors.append(message)
            self.events.warning(f"'{request.document}': the request about {message}")

    def _reusable(
        self, path: Path, request: Request, context: RowContext
    ) -> list[dict[str, dict[str, Any]]] | None:
        """Return the stored answers of every unit of *request*, else None.

        *path* is the database; one older than the classifications
        component (read by a dry run) holds none.
        """
        if not path.is_file():
            return None
        unit_kind = self.classifier.unit
        name = self.classifier.name

        def lookup(
            connection: sqlite3.Connection,
        ) -> list[dict[str, dict[str, Any]]] | None:
            found = []
            for unit in request.units:
                answers = reusable_answers(
                    connection,
                    context,
                    unit=unit_kind,
                    locator=unit.locator,
                    classifier=name,
                    content_sha256=unit.content_sha256,
                    request_sha256=request.sha256,
                )
                if set(answers) != self.questions:
                    return None
                found.append(answers)
            return found

        try:
            return read(path, lookup)
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return None
            raise


async def run_classify(
    plan: ClassifyPlan, events: RunEvents | None = None
) -> ClassifyResult:
    """Classify the plan's ready documents one after another.

    A dry run opens no store for writing and builds no client: it counts
    the units that would be asked and reused. A failed request fails its
    document, and the run goes on with the next request.

    Raises:
        ConfigurationError: The decision client cannot be built; nothing
            has been written. :class:`StoreRefusedError` when a database is
            refused at opening.
    """
    sink: RunEvents = events if events is not None else NullEvents()
    env = CallEnv.create(plan.settings, sink)
    started = time.monotonic()
    ready = plan.having(READY)
    model: DecisionModel | None = None
    stores: _Stores | None = None
    if not plan.dry_run and ready:
        choice = plan.spec.classify
        try:
            model = DecisionModel(
                model=choice.model,
                provider=choice.provider,
                endpoint=choice.endpoint,
                env=env,
                role=ROLE,
            )
        except OSError as exc:
            raise ConfigurationError(
                f"Cannot build the decision client: {exc}", items_total=len(ready)
            ) from exc
        stores = _Stores(plan, sink)
    worker = _Classify(plan, model, stores, sink)
    results: list[TargetResult] = []
    exit_code: int = ExitCode.FAILURE
    try:
        position = 0
        for target in plan.targets:
            if target.state == READY:
                position += 1
            results.append(await worker.target(target, position, len(ready)))
        exit_code = ClassifyResult(tuple(results)).exit_code
    except (asyncio.CancelledError, KeyboardInterrupt):
        exit_code = ExitCode.INTERRUPTED
        raise
    finally:
        if stores is not None:
            await stores.close(exit_code)
        if model is not None:
            await model.aclose()
    return ClassifyResult(
        targets=tuple(results),
        usage={role: env.usage.get(role) for role in env.usage.roles},
        databases=stores.databases if stores is not None else (),
        seconds=time.monotonic() - started,
    )
