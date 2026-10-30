"""The decision pre-pass: the relevance of every chunk still to extract.

:class:`ChunkDecider` asks a decision model one predicate per chunk: whether
the chunk's text or page image holds at least one entry of the kind the
schema extracts. The question comes from the schema's ``name`` and
``description``, or from its wrapper key ``decision_question``. Chunks are
read from their stream ``--concurrency`` at a time; page images are
converted with :func:`~chronominer.common.images.decision_jpeg` at
``decisions.image_max_px``. Each chunk is one request of the run's
:class:`~chronominer.llm.decision_model.DecisionModel` under the role
``decide``: through the extension seam (client kind ``decision``), the
provider's rate limiter and the shared retry ladder, which follows each failed
attempt's decision; their usage, failed attempts and the billed responses of
malformed answers included, counts toward the run's totals. The attempts are
not stored in the store's attempt rows.

A threshold is looked up by the request's fingerprint
(:meth:`ChunkDecider.fingerprint`), which covers the question and the
upstream image settings; without one, ``apply`` skips nothing and the pass
only records.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from collections.abc import AsyncIterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from chronominer.common.classifiers import Classifier, parse_classifier
from chronominer.common.decision_runner import Request, Unit
from chronominer.common.decisions import (
    Image,
    Json,
    Part,
    Predicate,
    Text,
    request_fingerprint,
)
from chronominer.common.images import decision_jpeg
from chronominer.llm.caller import CallEnv
from chronominer.llm.decision_model import KIND, DecisionModel
from chronominer.spec import DecisionSpec, ResolvedSchema

__all__ = [
    "KIND",
    "QUESTION_KEY",
    "ROLE",
    "SKIPPED_CONTENT",
    "SKIPPED_OUTCOME",
    "ChunkDecider",
    "ChunkDecision",
    "relevance_question",
]

ROLE: Final = "decide"
QUESTION_KEY: Final = "decision_question"
"""The schema wrapper key whose text replaces the built relevance question."""
SKIPPED_OUTCOME: Final = "skipped_irrelevant"
"""The outcome of a chunk ``apply`` skipped as holding no requested entry."""
SKIPPED_CONTENT: Final = (
    '{"contains_no_content_of_requested_type": true, "entries": []}'
)
"""The stored answer of a skipped chunk: no content, rendered as such."""
_NAME: Final = "relevant"


@dataclass(frozen=True)
class ChunkDecision:
    """The relevance probability of one chunk."""

    idx: int
    relevance: float


def relevance_question(schema: ResolvedSchema) -> Predicate:
    """Return the relevance predicate of *schema*'s chunks.

    Raises:
        ValueError: The wrapper's ``decision_question`` is not a nonempty text.
    """
    custom = schema.schema.get(QUESTION_KEY)
    if custom is not None:
        if not isinstance(custom, str) or not custom.strip():
            raise ValueError(f"the schema's {QUESTION_KEY} must be a nonempty text")
        return Predicate(_NAME, custom.strip())
    description = schema.schema.get("description")
    what = f" ({description.strip()})" if isinstance(description, str) else ""
    return Predicate(
        _NAME,
        "The input (text or page image) contains at least one entry of the kind "
        f"the extraction schema {schema.name}{what} asks for.",
    )


def _relevance_classifier(question: Predicate) -> Classifier:
    """Return the one-question classifier the pre-pass asks about each chunk."""
    return parse_classifier(
        {
            "name": "relevance",
            "version": 1,
            "unit": "chunk",
            "questions": [
                {
                    "name": question.name,
                    "type": "predicate",
                    "instructions": question.instructions,
                }
            ],
        },
        "relevance pre-pass",
    )


class ChunkDecider(DecisionModel):
    """Ask the run's decision model whether chunks hold requested entries.

    Raises:
        OSError: None of the key variables of the decision provider is set.
        ValueError: The schema's ``decision_question`` is invalid.
    """

    def __init__(
        self,
        spec: DecisionSpec,
        env: CallEnv,
        schema: ResolvedSchema,
        *,
        concurrency: int = 1,
    ) -> None:
        self.spec = spec
        self.schema_name = schema.name
        self.question = relevance_question(schema)
        self.classifier = _relevance_classifier(self.question)
        block = env.settings.decisions
        self.image_max_px = block.image_max_px
        self.concurrency = max(1, concurrency)
        self._thresholds = block.thresholds
        super().__init__(
            model=spec.model,
            provider=spec.provider,
            endpoint=spec.endpoint,
            env=env,
            role=ROLE,
        )

    @property
    def question_sha(self) -> str:
        """The first 16 hex digits of the SHA-256 of the question text."""
        text = self.question.instructions.encode("utf-8")
        return hashlib.sha256(text).hexdigest()[:16]

    def fingerprint(self, upstream: Mapping[str, Any] | None = None) -> str:
        """Return the fingerprint of this pass's requests (see the settings).

        *upstream* is the image-settings record the pages were rendered with;
        text chunks pass None.
        """
        return request_fingerprint(
            tool="chronominer",
            dialect=self.dialect,
            model=self.spec.model,
            endpoint=self.spec.endpoint,
            schema=self.schema_name,
            question=self.question_sha,
            image_max_px=self.image_max_px if upstream is not None else None,
            context="document+schema",
            upstream=dict(upstream) if upstream is not None else "text",
        )

    def threshold(self, fingerprint: str) -> float | None:
        """Return the relevance below which ``apply`` skips, or None."""
        return self._thresholds.get(fingerprint)

    async def classify(
        self, document: str, chunks: AsyncIterable[Any]
    ) -> list[ChunkDecision]:
        """Return the relevance of each chunk of *chunks*.

        A chunk is a text chunk (``text``) or a page image (``base64``); one
        whose answer is a refusal is left out.

        Raises:
            MalformedAnswer: An answer does not fit its question.
            httpx.HTTPError: A request failed after the retries.
        """
        decisions: list[ChunkDecision] = []
        window: list[Any] = []
        async for chunk in chunks:
            window.append(chunk)
            if len(window) == self.concurrency:
                decisions += await self._window(document, window)
                window = []
        if window:
            decisions += await self._window(document, window)
        return decisions

    async def _window(
        self, document: str, chunks: Sequence[Any]
    ) -> list[ChunkDecision]:
        found = await asyncio.gather(*(self._one(document, chunk) for chunk in chunks))
        return [decision for decision in found if decision is not None]

    async def _one(self, document: str, chunk: Any) -> ChunkDecision | None:
        request = Request(
            self.classifier,
            document,
            (Unit(f"chunk {chunk.index + 1}", chunk.index, (self._content(chunk),)),),
            (Json({"document": document, "schema": self.schema_name}),),
        )
        label = f"{document} relevance {chunk.index}"
        result = await self.runner.ask(request, label=label)
        answer = result.units[0].answers[0]
        if answer.refusal is not None or answer.probability is None:
            return None
        return ChunkDecision(chunk.index, answer.probability)

    def _content(self, chunk: Any) -> Part:
        if getattr(chunk, "mime_type", "") == "text/plain" or not chunk.base64:
            return Text(chunk.text)
        data = base64.b64decode(chunk.base64)
        return Image(decision_jpeg(data, self.image_max_px))
