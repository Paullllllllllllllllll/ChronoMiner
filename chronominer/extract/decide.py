"""The decision pre-pass: the relevance of every chunk still to extract.

:class:`ChunkDecider` asks a decision model one predicate per chunk: whether
the chunk's text or page image holds at least one entry of the kind the
schema extracts. The question comes from the schema's ``name`` and
``description``, or from its wrapper key ``decision_question``. Chunks are
read from their stream ``--concurrency`` at a time; page images are
downscaled to ``decisions.image_max_px`` on their longer side. Calls run under
the role ``decide`` through the extension seam (client kind ``decision``), the
provider's rate limiter and the shared retry ladder, which follows each failed
attempt's decision; their usage, failed attempts included, counts toward the
run's totals. The attempts are not stored in the store's attempt rows.

A threshold is looked up by the request's fingerprint
(:meth:`ChunkDecider.fingerprint`), which covers the question and the
upstream image settings; without one, ``apply`` skips nothing and the pass
only records.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import os
from collections.abc import AsyncIterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx
from PIL import Image as PILImage

from chronominer import ext
from chronominer.common.capture import async_capture_hooks
from chronominer.common.decision_client import DecisionClient
from chronominer.common.decisions import (
    DecisionResult,
    Image,
    Json,
    MalformedAnswer,
    Part,
    Predicate,
    PredicateAnswer,
    Question,
    Text,
    dialect_for,
    request_fingerprint,
)
from chronominer.common.retry import (
    Decision,
    ErrorKind,
    RetryHooks,
    call_with_retry,
    classify_error,
)
from chronominer.common.usage import Usage
from chronominer.llm.caller import CallEnv
from chronominer.llm.retrying import AttemptDecisions
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
KIND: Final = "decision"
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


def _downscale(data: bytes, max_px: int) -> bytes:
    """Return *data* as a JPEG whose longer side is at most *max_px*."""
    with PILImage.open(io.BytesIO(data)) as opened:
        image = opened.convert("RGB")
    image.thumbnail((max_px, max_px))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


class ChunkDecider:
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
        settings = env.settings
        block = settings.decisions
        self.spec = spec
        self.env = env
        self.schema_name = schema.name
        self.question = relevance_question(schema)
        self.dialect = dialect_for(spec.model, spec.provider)
        self.image_max_px = block.image_max_px
        self.concurrency = max(1, concurrency)
        self._thresholds = block.thresholds
        self._policy = settings.retry.policy()
        self._timeout_attempts = settings.retry.timeout_attempts
        variables = ext.key_variables(self.dialect, kind=KIND, settings=settings)
        if variables and not any(os.environ.get(name) for name in variables):
            raise OSError(f"the decision model needs the key variable {variables[0]}")
        self._http = httpx.AsyncClient(
            timeout=block.timeout_s, event_hooks=async_capture_hooks()
        )
        self._clients: dict[str, DecisionClient] = {}

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
        parts: list[Part] = [
            Json({"document": document, "schema": self.schema_name}),
            self._content(chunk),
        ]
        label = f"{document} relevance {chunk.index}"
        result = await self._call(parts, (self.question,), label)
        answer = result.answers[_NAME]
        if not isinstance(answer, PredicateAnswer):
            return None
        return ChunkDecision(chunk.index, answer.probability)

    def _content(self, chunk: Any) -> Part:
        if getattr(chunk, "mime_type", "") == "text/plain" or not chunk.base64:
            return Text(chunk.text)
        data = base64.b64decode(chunk.base64)
        return Image(_downscale(data, self.image_max_px))

    def client_for(self, identity: ext.Identity) -> DecisionClient:
        """Return the client for an attempt identity, building it if new.

        The client takes the extension's client options of the identity.

        Raises:
            OSError: The identity's key variable is unset.
        """
        client = self._clients.get(identity.api_key_env)
        if client is None:
            options = ext.client_options(identity)
            key = options.api_key or os.environ.get(identity.api_key_env, "").strip()
            if not key:
                raise OSError(
                    f"the decision model needs the key variable {identity.api_key_env}"
                )
            client = DecisionClient(
                self._http,
                dialect=self.dialect,
                api_key=key,
                endpoint=options.base_url or self.spec.endpoint,
            )
            self._clients[identity.api_key_env] = client
        return client

    async def _call(
        self, parts: Sequence[Part], questions: Sequence[Question], label: str
    ) -> DecisionResult:
        decisions = AttemptDecisions()

        async def one_attempt() -> DecisionResult:
            attempt = ext.attempt(
                ROLE,
                self.dialect,
                None,
                settings=self.env.settings,
                build_client=self.client_for,
                endpoint=None,
                model=self.spec.model,
                kind=KIND,
            )
            async with contextlib.AsyncExitStack() as stack:
                call = await decisions.enter(
                    stack, attempt, on_failure=lambda: self._record(attempt.usage)
                )
                try:
                    result = await call.client.decide(self.spec.model, parts, questions)
                except Exception as exc:
                    decisions.failed(call, exc)
                    self._record(call.usage)
                    raise
                call.commit(
                    Usage(
                        input_tokens=result.usage.input_tokens,
                        output_tokens=result.usage.output_tokens,
                        total_tokens=result.usage.input_tokens
                        + result.usage.output_tokens,
                        cached_tokens=result.usage.cached_tokens,
                    )
                )
                self._record(call.usage)
                return result

        timeouts = 0

        def decide(exc: BaseException) -> Decision:
            nonlocal timeouts
            decision = decisions.decide(exc)
            retrying = decision in (Decision.RETRY, Decision.SWITCH_KEY)
            if not retrying or classify_error(exc) is not ErrorKind.TIMEOUT:
                return decision
            timeouts += 1
            return decision if timeouts < self._timeout_attempts else Decision.FAIL

        hooks = RetryHooks(on_error=decide, gate=self.env.limiters.get(self.dialect))
        return await call_with_retry(
            one_attempt,
            policy=self._policy,
            hooks=hooks,
            label=label,
            validation_errors=(MalformedAnswer,),
        )

    def _record(self, usage: Usage | None) -> None:
        if usage is not None:
            self.env.record(ROLE, usage)

    async def aclose(self) -> None:
        """Close the HTTP client. Never raises."""
        with contextlib.suppress(Exception):
            await self._http.aclose()
