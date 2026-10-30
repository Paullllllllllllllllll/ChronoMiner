"""Plan classifier requests and send them to a decision model.

A :class:`Unit` is one thing a classifier judges: its locator (the tool's
name for it, such as ``page 3`` or ``lines 120-180``), its ordinal in the
document, the decision parts sent about it and the SHA-256 of exactly those
parts. :func:`plan_requests` groups units in order into requests of at most
the classifier's ``per_request`` units, at most
:data:`~.decisions.MAX_QUESTIONS` questions and at most *image_budget*
images, those of the shared context included; :attr:`Request.sha256` covers
the classifier, the document name, the context parts and every unit's
locator and content in order.

:class:`DecisionRunner` sends one request through the tool's attempt seam
(:class:`Seam`): each try opens an attempt (the tool's ``ext.attempt`` with
the client kind ``decision``), calls its client, commits the usage and hands
the attempt's usage, failed attempts included, to the tool's recorder. Retries
run through :func:`~.retry.call_with_retry` behind the tool's rate-limit
gate, under the tool's retry settings: transient errors follow each failed
attempt's decision with a separate timeout budget, and a
:class:`~.decisions.MalformedAnswer` is retried on the validation budget,
the usage of its billed response recorded.
A refusal is an answer, returned with its reason. The module imports no tool
package.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Final, Protocol

from .classifiers import Classifier, ClassifierQuestion, Kind, slot_name
from .decisions import (
    MAX_QUESTIONS,
    Answer,
    ChoiceAnswer,
    DecisionResult,
    DecisionUsage,
    Dialect,
    Image,
    Json,
    MalformedAnswer,
    Part,
    PredicateAnswer,
    Question,
    RefusalAnswer,
    Text,
)
from .retry import (
    Decision,
    ErrorKind,
    RetryGate,
    RetryHooks,
    call_with_retry,
    classify_error,
    default_decision,
)
from .settings_schema import RetrySettings
from .usage import Usage

__all__ = [
    "DEFAULT_IMAGE_BUDGET",
    "DecisionAttempt",
    "DecisionCaller",
    "DecisionRunner",
    "Request",
    "RequestResult",
    "Seam",
    "Unit",
    "UnitAnswer",
    "UnitAnswers",
    "parts_sha256",
    "plan_requests",
]

DEFAULT_IMAGE_BUDGET: Final = 20
"""The most images one request carries by default."""


def _part_entry(part: Part) -> list[str]:
    if isinstance(part, Text):
        return ["text", part.text]
    if isinstance(part, Json):
        return ["json", json.dumps(part.data, ensure_ascii=False)]
    return ["image", hashlib.sha256(part.data).hexdigest()]


def parts_sha256(parts: Sequence[Part]) -> str:
    """Return the SHA-256 of decision *parts* in order (images by their bytes).

    JSON parts are hashed as sent, keys in their order, so a reordered object
    changes the hash.
    """
    text = json.dumps([_part_entry(part) for part in parts], ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Unit:
    """One unit to ask about; ``content_sha256`` hashes exactly its parts."""

    locator: str
    ordinal: int
    parts: tuple[Part, ...]
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.locator:
            raise ValueError("a unit needs a locator")
        if not self.parts:
            raise ValueError(f"unit {self.locator!r} has no parts")
        object.__setattr__(self, "parts", tuple(self.parts))
        object.__setattr__(self, "content_sha256", parts_sha256(self.parts))

    @property
    def images(self) -> int:
        """Return the number of image parts."""
        return sum(isinstance(part, Image) for part in self.parts)


@dataclass(frozen=True)
class Request:
    """The units of one document in one decision request, in the order sent.

    ``sha256`` covers the classifier, the document name, the context parts
    and each unit's locator and content in order, so equal content of
    another document or another unit never shares it.
    """

    classifier: Classifier
    document: str
    units: tuple[Unit, ...]
    context: tuple[Part, ...] = ()
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError("a request needs at least one unit")
        body = {
            "classifier": self.classifier.sha256,
            "document": self.document,
            "context": parts_sha256(self.context),
            "units": [[unit.locator, unit.content_sha256] for unit in self.units],
        }
        text = json.dumps(body, sort_keys=True)
        object.__setattr__(self, "sha256", hashlib.sha256(text.encode()).hexdigest())

    def slot(self, position: int) -> int | None:
        """Return the slot number of the unit at *position* (None: ungrouped)."""
        return position + 1 if self.classifier.grouped else None

    def parts(self) -> list[Part]:
        """Return the parts sent: the context, then each unit (labelled if grouped)."""
        parts: list[Part] = list(self.context)
        label = self.classifier.unit.capitalize()
        for position, unit in enumerate(self.units):
            slot = self.slot(position)
            if slot is not None:
                parts.append(Text(f"{label} {slot}:"))
            parts.extend(unit.parts)
        return parts

    def questions(self) -> list[Question]:
        """Return the questions sent, unit by unit."""
        return [
            question
            for position in range(len(self.units))
            for question in self.classifier.questions_for(self.slot(position))
        ]


def plan_requests(
    classifier: Classifier,
    document: str,
    units: Sequence[Unit],
    *,
    context: Sequence[Part] = (),
    image_budget: int = DEFAULT_IMAGE_BUDGET,
) -> list[Request]:
    """Group the *units* of *document* in order into requests about them.

    A request holds at most ``classifier.per_request`` units, at most
    :data:`~.decisions.MAX_QUESTIONS` questions and at most *image_budget*
    images, the images of *context* (which opens every request) included.

    Raises:
        ValueError: *image_budget* is below 1, the context alone exceeds
            the budget, or one unit does not fit beside the context.
    """
    if image_budget < 1:
        raise ValueError("image_budget must be at least 1")
    context = tuple(context)
    shared = sum(isinstance(part, Image) for part in context)
    if shared > image_budget:
        raise ValueError(
            f"the context carries {shared} images, more than the budget of "
            f"{image_budget}"
        )
    per_unit = len(classifier.questions)
    size = min(classifier.per_request, MAX_QUESTIONS // per_unit)
    requests: list[Request] = []
    group: list[Unit] = []
    images = shared
    for unit in units:
        if shared + unit.images > image_budget:
            raise ValueError(
                f"unit {unit.locator!r} carries {unit.images} images; with the "
                f"context's {shared} that exceeds the budget of {image_budget}"
            )
        if group and (len(group) == size or images + unit.images > image_budget):
            requests.append(Request(classifier, document, tuple(group), context))
            group, images = [], shared
        group.append(unit)
        images += unit.images
    if group:
        requests.append(Request(classifier, document, tuple(group), context))
    return requests


@dataclass(frozen=True)
class UnitAnswer:
    """The answer to one classifier question about one unit.

    ``value`` is the chosen value of a choice and the expected level (0-based
    index) of a score; a predicate has only ``probability``. ``probability``
    is that of the chosen value, a score's confidence where reported.
    ``refusal`` is set (the reason, or ``""``) when the model declined, and
    then nothing else is.
    """

    question: str
    kind: Kind
    value: str | float | None = None
    probability: float | None = None
    probabilities: Mapping[str, float] | None = None
    refusal: str | None = None


@dataclass(frozen=True)
class UnitAnswers:
    """Every question's answer about one unit, in classifier order."""

    unit: Unit
    answers: tuple[UnitAnswer, ...]


@dataclass(frozen=True)
class RequestResult:
    """The answers of one request with the model, dialect and usage."""

    request: Request
    units: tuple[UnitAnswers, ...]
    model: str
    dialect: Dialect
    usage: DecisionUsage
    request_id: str | None = None


def _unit_answer(question: ClassifierQuestion, answer: Answer) -> UnitAnswer:
    name, kind = question.name, question.kind
    if isinstance(answer, RefusalAnswer):
        return UnitAnswer(name, kind, refusal=answer.reason or "")
    if isinstance(answer, PredicateAnswer):
        return UnitAnswer(name, kind, probability=answer.probability)
    if isinstance(answer, ChoiceAnswer):
        probability = answer.probabilities.get(answer.choice, answer.confidence)
        return UnitAnswer(
            name, kind, answer.choice, probability, dict(answer.probabilities)
        )
    return UnitAnswer(
        name, kind, answer.score, answer.confidence, dict(answer.probabilities)
    )


class DecisionCaller(Protocol):
    """A decision client (:class:`~.decision_client.DecisionClient`)."""

    async def decide(
        self, model: str, parts: Sequence[Part], questions: Sequence[Question]
    ) -> DecisionResult:
        """Return the model's answers to *questions* about *parts*."""
        ...


class DecisionAttempt(Protocol):
    """One attempt context of the tool's extension seam (``ext.Attempt``)."""

    @property
    def client(self) -> DecisionCaller:
        """The client built for the attempt's identity."""
        ...

    @property
    def usage(self) -> Usage:
        """Committed and recovered usage together."""
        ...

    def commit(self, usage: Usage | None) -> None:
        """Record the usage of a successful call."""
        ...

    def on_error(self, exc: BaseException) -> Decision:
        """Record what a failed call recovered and decide what follows."""
        ...

    async def __aenter__(self) -> object:
        """Build the client; raise when the attempt cannot start."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Release the attempt."""
        ...


@dataclass(frozen=True)
class Seam:
    """What a tool injects into the runner.

    ``attempt`` opens one attempt (the tool's ``ext.attempt`` bound to role,
    provider, settings, client builder, model and client kind ``decision``);
    ``record`` adds usage to the run's totals; ``retry`` holds the attempt
    budgets; ``gate`` is the provider's rate limiter; ``sleep`` replaces
    ``asyncio.sleep`` between retries.
    """

    attempt: Callable[[], DecisionAttempt]
    record: Callable[[Usage], None]
    retry: RetrySettings = field(default_factory=RetrySettings)
    gate: RetryGate | None = None
    sleep: Callable[[float], Awaitable[None]] | None = None


class _AttemptDecisions:
    """The decisions of failed attempts, until the retry ladder asks for them."""

    def __init__(self) -> None:
        self._pending: dict[int, tuple[BaseException, Decision]] = {}

    def failed(self, attempt: DecisionAttempt, exc: Exception) -> None:
        self._pending[id(exc)] = (exc, attempt.on_error(exc))

    def decide(self, exc: BaseException) -> Decision:
        entry = self._pending.pop(id(exc), None)
        return default_decision(exc) if entry is None else entry[1]


class DecisionRunner:
    """Send classifier requests to one decision model in one dialect."""

    def __init__(self, seam: Seam, *, model: str, dialect: Dialect) -> None:
        self.seam = seam
        self.model = model
        self.dialect: Dialect = dialect

    async def ask(self, request: Request, *, label: str = "") -> RequestResult:
        """Return the answers of every unit of *request*.

        *label* names the request in retry warnings.

        Raises:
            MalformedAnswer: The answers stayed invalid on every allowed try.
            Exception: The last error of a call that failed after its retries,
                such as ``httpx.HTTPError`` or the seam's ``OSError``.
        """
        units = request.units
        name = label or (
            f"{request.classifier.name} {units[0].locator}..{units[-1].locator}"
        )
        result = await self._call(request.parts(), request.questions(), name)
        answers = tuple(
            UnitAnswers(
                unit,
                tuple(
                    _unit_answer(
                        question,
                        result.answers[slot_name(question.name, request.slot(n))],
                    )
                    for question in request.classifier.questions
                ),
            )
            for n, unit in enumerate(units)
        )
        return RequestResult(
            request,
            answers,
            result.model or self.model,
            self.dialect,
            result.usage,
            result.request_id,
        )

    async def _call(
        self, parts: Sequence[Part], questions: Sequence[Question], label: str
    ) -> DecisionResult:
        seam = self.seam
        decisions = _AttemptDecisions()

        async def one_attempt() -> DecisionResult:
            attempt = seam.attempt()
            async with contextlib.AsyncExitStack() as stack:
                try:
                    await stack.enter_async_context(attempt)
                except Exception as exc:
                    decisions.failed(attempt, exc)
                    seam.record(attempt.usage)
                    raise
                try:
                    result = await attempt.client.decide(self.model, parts, questions)
                except Exception as exc:
                    if isinstance(exc, MalformedAnswer) and exc.usage is not None:
                        attempt.commit(_usage(exc.usage))
                    decisions.failed(attempt, exc)
                    seam.record(attempt.usage)
                    raise
                attempt.commit(_usage(result.usage))
                seam.record(attempt.usage)
                return result

        timeouts = 0

        def decide(exc: BaseException) -> Decision:
            nonlocal timeouts
            decision = decisions.decide(exc)
            retrying = decision in (Decision.RETRY, Decision.SWITCH_KEY)
            if not retrying or classify_error(exc) is not ErrorKind.TIMEOUT:
                return decision
            timeouts += 1
            return decision if timeouts < seam.retry.timeout_attempts else Decision.FAIL

        hooks = RetryHooks(on_error=decide, gate=seam.gate, sleep=seam.sleep)
        return await call_with_retry(
            one_attempt,
            policy=seam.retry.policy(),
            hooks=hooks,
            label=label,
            validation_errors=(MalformedAnswer,),
        )


def _usage(usage: DecisionUsage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
        cached_tokens=usage.cached_tokens,
    )
