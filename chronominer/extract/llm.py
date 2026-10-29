"""The extractor: an item's chunks sent to the run's model.

Every chunk of the item is a ``chunk`` unit of the item's store document,
planned as ``pending`` before the first call, so a ``--pages`` run leaves the
document partial. The chunks still to do (pending and failed) are extracted
concurrently through
:class:`~chronominer.extract.extraction.ChunkExtractor`, each unit committed
to the store with its entries as it finishes; under ``--batch`` they are
submitted to the provider's batch API instead. The extractor also records
the document's run metadata; it writes no file: the run core renders the
item from the store.

A text document's chunks come from its line ranges (see
:mod:`~chronominer.extract.chunks`). Under ``--chunking auto`` a document
with stored chunks resumes on their stored ranges; under the other modes a
document whose stored chunks were cut with other ranges is refused unless
``--force``. A PDF or image document is
refused when its stored pages used other image settings; a model change alone
resumes, and an image option the run leaves unset keeps its stored value.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from chronominer.common.decisions import MalformedAnswer
from chronominer.common.store import (
    BLOCKING_SUBMISSIONS,
    PlannedUnit,
    UnitRow,
    refresh_documents,
)
from chronominer.extract import ChunkCounter, Item, Job, NothingSelectedError
from chronominer.extract.chunks import (
    CHUNKING_METHODS,
    CHUNKING_TEXT_VERSION,
    PAGES_METHOD,
    RangeJob,
    TextChunk,
    chunk_slice,
    custom_id,
    extract_payload,
    text_chunks,
    text_ranges,
)
from chronominer.extract.decide import (
    SKIPPED_CONTENT,
    SKIPPED_OUTCOME,
    ChunkDecider,
)
from chronominer.extract.extraction import ROLE, ChunkExtractor
from chronominer.extract.images import (
    FolderPages,
    PageImage,
    file_provenance,
    resolve_image_settings,
    resume_images,
    stream_pages,
)
from chronominer.extract.pdf import PDF_PAGE_NAME, PdfPages, PDFProcessor
from chronominer.extract.units import DocumentHandle, EntrySchema, run_unit_stream
from chronominer.llm.caller import ModelCaller
from chronominer.llm.options import model_capabilities, response_format_for
from chronominer.output.store import (
    ENTRY_TABLE,
    chunk_decision_rows,
    clear_chunk_decisions,
    image_changes,
    image_settings_record,
    metadata_writer,
)
from chronominer.ranges.adjust import AdjustOptions
from chronominer.ranges.lines import Range, read_lines
from chronominer.spec import (
    ImageSpec,
    ResolvedSchema,
    RunSpec,
    SpecError,
    resolve_schema,
)

__all__ = [
    "ADJUST_ROLE",
    "ROLE",
    "BatchSubmissionFailedError",
    "ChunkingChangedError",
    "Extractor",
    "ImageSettingsChangedError",
    "image_record",
]

logger = logging.getLogger(__name__)

ADJUST_ROLE = "adjust"
"""The usage role of line-range boundary calls."""


class ImageSettingsChangedError(RuntimeError):
    """The document's stored pages were produced with other image settings."""


class ChunkingChangedError(RuntimeError):
    """The document's stored chunks were cut with other line ranges."""


class BatchSubmissionFailedError(RuntimeError):
    """A batch submission was refused, or left unresolved."""


def image_record(
    spec: RunSpec, original_max_side_px: int | None = None
) -> dict[str, Any]:
    """Return the image-settings record of *spec*'s model and image options.

    *original_max_side_px* is the ``images`` setting of that name.

    Raises:
        ValueError: An image setting is invalid.
    """
    model = spec.model
    settings, model_type, _section = resolve_image_settings(
        model.provider, model.model, spec.images, original_max_side_px
    )
    return image_settings_record(settings, model_type)


def _recorded(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return dict(value) if isinstance(value, Mapping) else None


def _same_layout(units: Sequence[UnitRow], planned: Sequence[PlannedUnit]) -> bool:
    """Whether stored units fit the planned chunks; unplaced ones fit any."""
    return len(units) == len(planned) and all(
        unit.idx == plan.idx
        and (unit.source_index is None or str(unit.source_index) == plan.source_index)
        for unit, plan in zip(units, planned, strict=True)
    )


def _stored_ranges(units: Sequence[UnitRow]) -> list[Range]:
    """Return the line ranges of stored text chunks; none when one is unplaced."""
    if any(unit.source_index is None for unit in units):
        return []
    bounds = [str(unit.source_index).split("-") for unit in units]
    return [(int(start), int(end)) for start, end in bounds]


class Extractor:
    """Extract the chunks of text, image and PDF items with the run's model.

    In batch mode the chunks go to the batch API and no extraction client is
    opened. The boundary calls of ``--chunking adjust`` get their own caller
    when a text item first needs them.
    """

    def __init__(self, job: Job) -> None:
        self.job = job
        self.schema: ResolvedSchema | None = None
        self._client: ChunkExtractor | None = None
        self._adjust: ModelCaller | None = None
        self._decider: ChunkDecider | None = None

    def _schema(self) -> ResolvedSchema:
        if self.schema is None:
            path = self.job.schema_path
            argument = str(path) if path is not None else self.job.spec.prompt.schema
            self.schema = resolve_schema(argument)
        return self.schema

    async def open(self) -> None:
        """Resolve the schema; build the extraction client unless in batch mode.

        Raises:
            SpecError: The schema cannot be resolved.
            OSError: The key variable is unset.
        """
        schema = self._schema()
        spec = self.job.spec
        if spec.decision.mode != "off":
            try:
                self._decider = ChunkDecider(
                    spec.decision,
                    self.job.env,
                    schema,
                    concurrency=spec.run.concurrency,
                )
            except ValueError as exc:
                raise SpecError(str(exc)) from exc
        if self.job.batch:
            return
        self._client = ChunkExtractor(
            spec.model,
            self.job.env,
            schema=schema.schema,
            schema_name=schema.name,
            images=spec.images,
            role=ROLE,
        )

    async def close(self) -> None:
        """Close the clients."""
        client, self._client = self._client, None
        adjust, self._adjust = self._adjust, None
        decider, self._decider = self._decider, None
        if client is not None:
            await client.close()
        if adjust is not None:
            await adjust.aclose()
        if decider is not None:
            await decider.aclose()

    def _adjust_caller(self) -> ModelCaller:
        if self._adjust is None:
            self._adjust = self.job.env.caller(ADJUST_ROLE, self.job.spec.model)
        return self._adjust

    async def extract(self, item: Item) -> list[Path]:
        """Extract one item's chunks into its store document.

        Returns no paths: the run core renders the item.

        Raises:
            RuntimeError: The run core opened no store document for the item.
            ImageSettingsChangedError: The stored pages used other image
                settings.
            ChunkingChangedError: The stored chunks used other line ranges.
            BatchSubmissionFailedError: The batch submission was not accepted.
            UnitsFailedError: A chunk failed or could not be stored.
        """
        if item.document is None:
            raise RuntimeError(f"No store document is open for '{item.name}'.")
        handle = item.document
        if item.source == "text":
            await self._extract_text(item, handle)
        else:
            await self._extract_pages(item, handle)
        return []

    def _selected(self, item: Item, total: int, noun: str) -> list[int] | None:
        """Return the chunk indices *item.pages* selects; None without chunks.

        Raises:
            NothingSelectedError: The range selects no chunk.
        """
        if total == 0:
            self.job.events.warning(f"No {noun} found in '{item.name}'.")
            return None
        if item.pages is None:
            return list(range(total))
        selected = item.pages.resolve(total)
        if not selected:
            raise NothingSelectedError(
                f"Page range '{item.pages.describe()}' yielded no {noun} for "
                f"'{item.name}' ({total} {noun}). Skipping."
            )
        return selected

    async def _extract_text(self, item: Item, handle: DocumentHandle) -> None:
        spec = self.job.spec
        lines = read_lines(item.path)
        ranges: list[Range] = []
        if spec.chunking.mode == "auto" and not self.job.force:
            stored = handle.store.units(handle.doc_id, kind=handle.kind)
            ranges = _stored_ranges(stored)
        if not ranges:
            ranges = await text_ranges(
                item.path,
                lines,
                self._range_job(item),
                store=handle.store,
                lease=handle.lease,
                document=handle.name,
            )
        chunks = text_chunks(item.path, lines, ranges)
        selected = self._selected(item, len(chunks), "chunks")
        if selected is None:
            return
        await self._plan_text(handle, item, ranges)
        await self._write_metadata(
            handle,
            item,
            chunking_method=CHUNKING_METHODS.get(spec.chunking.mode, "unknown"),
            text_version=CHUNKING_TEXT_VERSION,
            total=len(chunks),
            image_provenance=None,
        )

        async def source(indices: list[int]) -> AsyncIterator[TextChunk]:
            for index in indices:
                yield chunks[index]

        await self._process(item, handle, selected, source)

    def _range_job(self, item: Item) -> RangeJob:
        spec = self.job.spec
        chunking = spec.chunking
        if chunking.mode != "adjust":
            return RangeJob(chunking.mode, chunking.chunk_tokens, spec.model.model)
        caller = self._adjust_caller()
        caps = model_capabilities(spec.model, self.job.settings)
        options = AdjustOptions(
            boundary_type=self._schema().name,
            context=item.adjust_context,
            response_format=response_format_for(spec.model, caps),
            context_window=chunking.context_window,
            chunk_tokens=chunking.chunk_tokens,
            prompt_caching=self.job.settings.prompt_caching,
            force=self.job.force,
        )
        return RangeJob(
            chunking.mode,
            chunking.chunk_tokens,
            spec.model.model,
            caller=caller,
            adjust=options,
            events=self.job.events,
        )

    async def _plan_text(
        self, handle: DocumentHandle, item: Item, ranges: Sequence[Range]
    ) -> None:
        """Plan the text chunks; refuse or replace chunks cut with other ranges.

        Raises:
            ChunkingChangedError: Stored chunks used other ranges and
                ``--force`` is not set.
        """
        planned = [
            PlannedUnit(index, f"{start}-{end}", f"{start}-{end}")
            for index, (start, end) in enumerate(ranges)
        ]
        stored = handle.store.units(handle.doc_id, kind=handle.kind)
        if stored and not _same_layout(stored, planned):
            if not self.job.force:
                raise ChunkingChangedError(
                    f"The chunks of '{handle.name}' were cut with other line "
                    f"ranges ({len(stored)} stored, {len(planned)} now). Skipping "
                    "it; use --force to extract it again, or restore the "
                    "earlier chunking settings."
                )
            await self._drop_chunks(handle)
        await handle.plan(planned)
        if self.job.force:
            await handle.reset()
            await handle.store.write(handle.lease, clear_chunk_decisions(handle.doc_id))

    @staticmethod
    async def _drop_chunks(handle: DocumentHandle) -> None:
        """Delete the document's chunk units and entries; responses stay.

        The batch requests of settled submissions go with the units; a
        request of a blocking submission keeps its unit, and the delete fails.
        """
        doc_id, kind = handle.doc_id, handle.kind

        def drop(connection: sqlite3.Connection) -> None:
            connection.execute(
                f"DELETE FROM {ENTRY_TABLE} WHERE doc_id = ? AND kind = ?",
                (doc_id, kind),
            )
            clear_chunk_decisions(doc_id)(connection)
            connection.execute(
                "DELETE FROM submission_requests WHERE doc_id = ? AND kind = ? "
                "AND submission_id NOT IN (SELECT submission_id FROM submissions "
                "WHERE status IN (SELECT value FROM json_each(?)))",
                (doc_id, kind, json.dumps(BLOCKING_SUBMISSIONS)),
            )
            connection.execute(
                "DELETE FROM units WHERE doc_id = ? AND kind = ?", (doc_id, kind)
            )
            refresh_documents(connection, doc_id)

        await handle.store.write(handle.lease, drop)

    async def _extract_pages(self, item: Item, handle: DocumentHandle) -> None:
        is_pdf = item.source == "pdf"
        units = self._units(item.path, is_pdf)
        noun = "pages" if is_pdf else "images"
        selected = self._selected(item, len(units), noun)
        if selected is None:
            return
        model = self.job.spec.model
        img_cfg, model_type, section = resolve_image_settings(
            model.provider,
            model.model,
            self._images(handle),
            self.job.settings.images.original_max_side_px,
        )
        record = image_settings_record(img_cfg, model_type)
        await self._record_settings(handle, record)
        if self.job.force:
            stored = handle.store.units(handle.doc_id, kind=handle.kind)
            if {unit.idx for unit in stored} - {unit.idx for unit in units}:
                await self._drop_chunks(handle)
        await handle.plan(units)
        if self.job.force:
            await handle.reset()
            await handle.store.write(handle.lease, clear_chunk_decisions(handle.doc_id))
        await self._write_metadata(
            handle,
            item,
            chunking_method=PAGES_METHOD,
            text_version=None,
            total=None,
            image_provenance=file_provenance(item.path, img_cfg, model_type),
        )

        async def source(indices: list[int]) -> AsyncIterator[PageImage]:
            pages: PdfPages | FolderPages
            if is_pdf:
                pages = PdfPages(item.path, img_cfg, model_type, section=section)
            else:
                pages = FolderPages(
                    item.path.parent,
                    img_cfg,
                    model_type,
                    image_paths=[item.path],
                    section=section,
                )
            async for page in stream_pages(pages, indices):
                yield replace(
                    page, custom_id=custom_id(item.path, item.source, page.index)
                )

        await self._process(item, handle, selected, source, upstream=record)

    @staticmethod
    def _units(path: Path, is_pdf: bool) -> list[PlannedUnit]:
        """Return every page of the item as an expected unit."""
        if not is_pdf:
            return [PlannedUnit(0, path.name, path.name)]
        total = PDFProcessor(path).page_count()
        return [
            PlannedUnit(idx, PDF_PAGE_NAME.format(number=idx + 1), idx)
            for idx in range(total)
        ]

    async def _process(
        self,
        item: Item,
        handle: DocumentHandle,
        selected: list[int],
        source: Any,
        *,
        upstream: Mapping[str, Any] | None = None,
    ) -> None:
        """Extract or submit the selected chunks that are still to do.

        *upstream* is the image-settings record of page chunks, None for text.
        """
        todo = set(handle.todo(retry_errors=self.job.retry_errors))
        needed = [index for index in selected if index in todo]
        already = len(selected) - len(needed)
        if already:
            logger.info("Skipped %d chunks already stored for %s", already, item.name)
        needed = await self._decide(handle, needed, source, upstream)
        if not needed:
            return
        if self._client is not None:
            self._client.update_context(item.context)
            self._client.update_context_image(item.context_image)
        if self.job.batch:
            payloads = [payload async for payload in source(needed)]
            await self._submit(item, handle, payloads)
            return
        if self._client is None:
            raise RuntimeError("The extractor was not opened.")
        counter = ChunkCounter(
            self.job.events, item.name, total=len(needed), already_complete=already
        )
        schema = self._schema()
        await run_unit_stream(
            handle,
            source(needed),
            self._client,
            extract_payload,
            concurrency=self.job.spec.run.concurrency,
            delay=self.job.settings.concurrency.delay_between_tasks,
            on_unit=counter,
            entries=EntrySchema(schema.schema, schema.name),
        )

    async def _decide(
        self,
        handle: DocumentHandle,
        needed: list[int],
        source: Any,
        upstream: Mapping[str, Any] | None,
    ) -> list[int]:
        """Record the relevance of *needed*; return the chunks still to send.

        Without a decision model, *needed* comes back unchanged. Under
        ``apply`` with a threshold for the request, chunks below it are
        written as skipped and left out of the returned list.
        """
        decider = self._decider
        if decider is None or not needed:
            return needed

        async def chunks() -> AsyncIterator[Any]:
            async for chunk in source(needed):
                if chunk.render_error is None:
                    yield chunk

        try:
            decisions = await decider.classify(handle.name, chunks())
        except (httpx.HTTPError, MalformedAnswer, OSError) as exc:
            self.job.events.warning(
                f"The decision pre-pass failed for '{handle.name}' ({exc}); "
                "its chunks are extracted without it."
            )
            return needed
        if not decisions:
            return needed
        fingerprint = decider.fingerprint(upstream)
        await handle.store.write(
            handle.lease,
            chunk_decision_rows(
                handle.doc_id,
                [(d.idx, d.relevance) for d in decisions],
                question_sha256=decider.question_sha,
                model=decider.spec.model,
                fingerprint=fingerprint,
                run_id=handle.run_id,
            ),
        )
        mode = self.job.spec.decision.mode
        threshold = decider.threshold(fingerprint)
        if mode != "apply" or threshold is None:
            if mode == "apply":
                self.job.events.warning(
                    f"No validated decision threshold for request {fingerprint}; "
                    f"'{handle.name}' is recorded only (see decisions.thresholds)."
                )
            return needed
        skipped = {d.idx: d for d in decisions if d.relevance < threshold}
        for idx, decision in skipped.items():
            await handle.write(
                idx,
                status="done",
                content=SKIPPED_CONTENT,
                outcome=SKIPPED_OUTCOME,
                provenance={
                    "decision": {
                        "relevance": round(decision.relevance, 4),
                        "threshold": threshold,
                        "question_sha256": decider.question_sha,
                        "fingerprint": fingerprint,
                    }
                },
                model=decider.spec.model,
                provider=decider.dialect,
            )
        if skipped:
            logger.info("Skipped %d irrelevant chunks in %s", len(skipped), handle.name)
        return [idx for idx in needed if idx not in skipped]

    async def _write_metadata(
        self,
        handle: DocumentHandle,
        item: Item,
        *,
        chunking_method: str,
        text_version: int | None,
        total: int | None,
        image_provenance: Mapping[str, Any] | None,
    ) -> None:
        """Record the document's run metadata and its schema text."""
        schema = self._schema()
        text = schema.path.read_text(encoding="utf-8")
        sha = await handle.store.store_artifact("schema", text)
        values = {
            "schema_name": schema.name,
            "model_name": self.job.spec.model.model,
            "chunking_method": chunking_method,
            "chunking_text_version": text_version,
            "document_total_chunks": total,
            "chunk_slice": chunk_slice(item.pages),
            "image_provenance": image_provenance,
        }
        await handle.store.write(
            handle.lease,
            metadata_writer(handle.store, handle.doc_id, values, schema_sha256=sha),
        )

    def _images(self, handle: DocumentHandle) -> ImageSpec:
        """Return the run's image options for the document's pages.

        While the document holds stored pages, an option the run leaves
        unset keeps its stored value (:func:`resume_images`), unless
        ``--force``.
        """
        images = self.job.spec.images
        if self.job.force:
            return images
        row = handle.store.document(handle.doc_id)
        if row is None or (row.units_done + row.units_failed) == 0:
            return images
        return resume_images(images, _recorded(row.image_settings))

    async def _record_settings(
        self, handle: DocumentHandle, record: dict[str, Any]
    ) -> None:
        """Compare the document's image settings with *record*, then store it.

        A changed model alone passes; a changed user setting refuses
        the item while it holds stored pages, unless ``--force``.

        Raises:
            ImageSettingsChangedError: A user image setting changed.
        """
        row = handle.store.document(handle.doc_id)
        recorded = _recorded(row.image_settings) if row is not None else None
        stored = row is not None and (row.units_done + row.units_failed) > 0
        if recorded is not None and stored and not self.job.force:
            changed = image_changes(recorded, record)
            if changed:
                raise ImageSettingsChangedError(
                    f"Image settings changed for '{handle.name}': "
                    f"{', '.join(changed)}. Skipping it; use --force to replace "
                    "the existing run."
                )
        if recorded != record:
            await handle.store.set_image_settings(handle.lease, record)

    async def _submit(
        self, item: Item, handle: DocumentHandle, payloads: list[Any]
    ) -> None:
        """Submit the chunks to the provider's batch API.

        Raises:
            BatchSubmissionFailedError: The submission was refused or left
                unresolved.
        """
        submit = self.job.submit
        if submit is None:
            raise RuntimeError("The run core set no batch submitter.")
        outcome = await submit(
            payloads,
            document=handle,
            spec=self.job.spec,
            settings=self.job.settings,
            schema_path=self._schema().path,
            context=item.context,
            context_image=item.context_image,
        )
        if outcome.status == "submitted":
            return
        if outcome.status == "refused":
            raise BatchSubmissionFailedError(
                outcome.detail or "the provider refused the batch"
            )
        detail = f" ({outcome.detail})" if outcome.detail else ""
        raise BatchSubmissionFailedError(
            f"The batch submission of '{item.name}' is unresolved{detail}: the "
            "provider may have created the batch. Its chunks stay blocked; run "
            "'chronominer batch check' to reconcile it."
        )
