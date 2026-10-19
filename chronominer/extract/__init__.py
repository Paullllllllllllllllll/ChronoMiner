"""Extraction of an item's chunks behind one interface.

The extractor (:class:`~chronominer.extract.llm.Extractor`) implements
:class:`ExtractionMethod`: the run opens it once, hands it the items to
extract and closes it. A text document's chunks are its line ranges, an image
is one chunk and a PDF has one chunk per page; each chunk is one unit of the
item's store document.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from chronominer.common.selection import PageRange
from chronominer.common.store import Store
from chronominer.discovery import ItemPaths, item_paths
from chronominer.events import ChunkDone, RunEvents
from chronominer.extract.units import DocumentHandle
from chronominer.llm.caller import CallEnv
from chronominer.output.render import output_formats
from chronominer.settings import Settings
from chronominer.spec import RunSpec

__all__ = [
    "BatchSubmitter",
    "ChunkCounter",
    "ExtractionMethod",
    "Item",
    "Job",
    "NothingSelectedError",
    "SubmitOutcome",
]


class NothingSelectedError(Exception):
    """``--pages`` selects no chunk of the item; the message names the range.

    The run core skips the item: nothing is called or rendered.
    """


class SubmitOutcome(Protocol):
    """The outcome of a batch submission.

    *status* is ``submitted`` (the units wait for ``batch check``),
    ``refused`` (the provider certainly made no batch; the units stay
    pending) or ``unresolved`` (the create's outcome is unknown; the units
    stay blocked). *detail* says why a submission was not accepted.
    """

    @property
    def status(self) -> str: ...

    @property
    def detail(self) -> str | None: ...


class BatchSubmitter(Protocol):
    """Submit one document's chunk payloads to the provider's batch API."""

    async def __call__(
        self,
        payloads: Sequence[Any],
        *,
        document: DocumentHandle,
        spec: RunSpec,
        settings: Settings,
        schema_path: Path | None,
        context: str | None = None,
        context_image: Path | None = None,
    ) -> SubmitOutcome: ...


@dataclass(frozen=True)
class Job:
    """What the items of one run share.

    *output_dir* None writes beside each input; *input_root* is the run's
    input, against which the mirror layout is taken. *schema_path* is the
    resolved schema file (None resolves ``--schema``). *batch* submits the
    chunks to the provider's batch API through *submit*.
    """

    spec: RunSpec
    settings: Settings
    events: RunEvents
    env: CallEnv
    output_dir: Path | None
    input_root: Path
    schema_path: Path | None = None
    batch: bool = False
    store: Store | None = None
    submit: BatchSubmitter | None = None

    @property
    def force(self) -> bool:
        """Whether earlier results are discarded instead of resumed."""
        return self.spec.run.force

    @property
    def retry_errors(self) -> bool:
        """Whether resume extracts failed chunks again; always."""
        return True

    @property
    def formats(self) -> tuple[str, ...]:
        """The formats the run renders; the first names ``_output.json``."""
        return output_formats(self.spec.output.formats)

    def locate(self, item: Path) -> ItemPaths:
        """Return the paths of *item* without creating anything."""
        return item_paths(
            item,
            output_dir=self.output_dir,
            layout=self.spec.output.layout,
            input_root=self.input_root,
        )


@dataclass(frozen=True)
class Item:
    """One item as the extractor works it.

    *source* is the input type (text, image or pdf); *pages* selects
    chunks. *context* and *context_image* are the item's resolved extraction
    context, *adjust_context* the context of its boundary decisions under
    ``--chunking adjust``. *document* is the item's store document, set by
    the run core.
    """

    path: Path
    source: str
    pages: PageRange | None = None
    context: str | None = None
    context_image: Path | None = None
    adjust_context: str | None = None
    document: DocumentHandle | None = None

    @property
    def name(self) -> str:
        """The item's file name."""
        return self.path.name


class ExtractionMethod(Protocol):
    """The extraction of a run.

    ``open`` builds what every item shares (a model client) before the
    first item; ``extract`` writes one item's chunks to its store document,
    raising when the item fails; ``close`` releases the resources.
    """

    async def open(self) -> None: ...

    async def extract(self, item: Item) -> list[Path]: ...

    async def close(self) -> None: ...


@dataclass
class ChunkCounter:
    """Report each finished chunk of an item as a :class:`ChunkDone` event."""

    events: RunEvents
    item: str
    total: int
    already_complete: int = 0
    done: int = field(default=0, init=False)

    def __call__(self, index: int, failed: bool) -> None:
        self.done += 1
        self.events.chunk_done(
            ChunkDone(
                item=self.item,
                chunk=index,
                status="failed" if failed else "ok",
                done=self.done,
                total=self.total,
                already_complete=self.already_complete,
            )
        )
