"""The shared decision runner: request planning, the call loop and downscaling."""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any

import httpx
import pytest
from PIL import Image as PILImage

from chronominer.common.classifiers import Classifier, parse_classifier
from chronominer.common.decision_client import DecisionClient
from chronominer.common.decision_runner import (
    DecisionRunner,
    Request,
    Seam,
    Unit,
    plan_requests,
)
from chronominer.common.decisions import (
    MAX_QUESTIONS,
    Choice,
    ChoiceAnswer,
    DecisionResult,
    DecisionUsage,
    Image,
    Json,
    MalformedAnswer,
    Part,
    Predicate,
    PredicateAnswer,
    Question,
    RefusalAnswer,
    ScoreAnswer,
    Text,
)
from chronominer.common.images import decision_jpeg
from chronominer.common.retry import Decision, default_decision
from chronominer.common.settings_schema import RetrySettings
from chronominer.common.usage import Usage

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def _classifier(per_request: int = 1, questions: int = 1, **extra: Any) -> Classifier:
    return parse_classifier(
        {
            "name": "front",
            "version": 1,
            "unit": "page",
            "per_request": per_request,
            "questions": [
                {
                    "name": f"q{n}",
                    "type": "predicate",
                    "instructions": f"Statement {n}.",
                }
                for n in range(questions)
            ],
            **extra,
        }
    )


def _unit(n: int, images: int = 1) -> Unit:
    parts: tuple[Part, ...] = tuple(Image(JPEG + bytes([n, k])) for k in range(images))
    return Unit(f"page {n}", n, parts or (Text(f"page {n}"),))


def test_requests_respect_per_request_and_keep_order() -> None:
    requests = plan_requests(
        _classifier(per_request=2), "book", [_unit(n) for n in range(5)]
    )

    assert [[u.ordinal for u in r.units] for r in requests] == [[0, 1], [2, 3], [4]]
    assert {r.document for r in requests} == {"book"}


def test_requests_respect_the_question_limit() -> None:
    classifier = _classifier(per_request=10, questions=150)

    requests = plan_requests(classifier, "book", [_unit(n) for n in range(5)])

    assert [len(r.units) for r in requests] == [2, 2, 1]
    assert all(len(r.questions()) <= MAX_QUESTIONS for r in requests)


def test_requests_respect_the_image_budget() -> None:
    units = [_unit(n, images=3) for n in range(5)]

    requests = plan_requests(_classifier(per_request=10), "b", units, image_budget=7)

    assert [len(r.units) for r in requests] == [2, 2, 1]
    with pytest.raises(ValueError, match="carries 8 images"):
        plan_requests(_classifier(), "b", [_unit(0, images=8)], image_budget=7)


def test_context_images_count_against_every_request() -> None:
    units = [_unit(n, images=3) for n in range(4)]
    context = (Text("Title page:"), Image(JPEG))

    requests = plan_requests(
        _classifier(per_request=10), "b", units, context=context, image_budget=7
    )

    assert [len(r.units) for r in requests] == [2, 2]
    assert all(sum(isinstance(p, Image) for p in r.parts()) <= 7 for r in requests)
    with pytest.raises(ValueError, match="with the context's 1"):
        plan_requests(
            _classifier(), "b", [_unit(0, 7)], context=context, image_budget=7
        )
    with pytest.raises(ValueError, match="context carries 2 images"):
        plan_requests(
            _classifier(),
            "b",
            [Unit("p", 0, (Text("x"),))],
            context=(Image(JPEG), Image(JPEG)),
            image_budget=1,
        )


def test_the_request_hash_follows_units_order_context_and_classifier() -> None:
    classifier = _classifier(per_request=2)
    a, b = _unit(1), _unit(2)
    first = Request(classifier, "book", (a, b))

    assert Request(classifier, "book", (_unit(1), _unit(2))).sha256 == first.sha256
    assert Request(classifier, "book", (b, a)).sha256 != first.sha256
    assert Request(classifier, "book", (a, b), (Json({"d": "x"}),)).sha256 != (
        first.sha256
    )
    assert Request(_classifier(per_request=3), "book", (a, b)).sha256 != first.sha256
    assert a.content_sha256 != b.content_sha256


def test_equal_content_in_another_document_or_unit_hashes_apart() -> None:
    classifier = _classifier()
    blank = (Text("blank"),)
    here = Request(classifier, "book", (Unit("page 1", 1, blank),))

    elsewhere = Request(classifier, "other", (Unit("page 1", 1, blank),))
    later = Request(classifier, "book", (Unit("page 9", 9, blank),))

    assert len({here.sha256, elsewhere.sha256, later.sha256}) == 3
    assert here.units[0].content_sha256 == later.units[0].content_sha256


def test_json_parts_hash_in_the_order_sent() -> None:
    one = Unit("p", 0, (Json({"a": 1, "b": 2}),))
    other = Unit("p", 0, (Json({"b": 2, "a": 1}),))

    assert one.content_sha256 != other.content_sha256


def test_grouped_requests_label_units_and_suffix_questions() -> None:
    request = Request(
        _classifier(per_request=2), "book", (_unit(1), _unit(2)), (Text("c"),)
    )

    parts = request.parts()

    assert parts[0] == Text("c")
    assert parts[1] == Text("Page 1:")
    assert parts[3] == Text("Page 2:")
    assert [q.name for q in request.questions()] == ["q0__01", "q0__02"]
    single = Request(_classifier(), "book", (_unit(1),))
    assert [q.name for q in single.questions()] == ["q0"]
    assert single.parts() == list(_unit(1).parts)


def _answers(questions: Sequence[Question]) -> DecisionResult:
    answers: dict[str, Any] = {}
    for question in questions:
        if isinstance(question, Predicate):
            answers[question.name] = PredicateAnswer(question.name, 0.75)
        elif isinstance(question, Choice):
            first, *_ = question.choices
            answers[question.name] = ChoiceAnswer(question.name, first, {first: 0.9})
        else:
            label = question.levels[1].label
            answers[question.name] = ScoreAnswer(question.name, 1.2, {label: 0.6}, 0.6)
    return DecisionResult(answers, "jev-test", DecisionUsage(10, 2, 4), "dec_1")


type Step = Exception | Callable[[Sequence[Question]], DecisionResult]


class FakeClient:
    """A decision client that plays one step per call."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps = steps
        self.calls: list[tuple[str, list[Part], list[str]]] = []

    async def decide(
        self, model: str, parts: Sequence[Part], questions: Sequence[Question]
    ) -> DecisionResult:
        self.calls.append((model, list(parts), [q.name for q in questions]))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(questions)


class FakeAttempt:
    """An attempt of the extension seam; a failed call recovers *recover* tokens."""

    def __init__(
        self, client: Any, fail_entry: Exception | None, recover: int = 5
    ) -> None:
        self._client = client
        self._fail_entry = fail_entry
        self._recover = recover
        self.committed = Usage()
        self.recovered = Usage()

    @property
    def client(self) -> Any:
        return self._client

    @property
    def usage(self) -> Usage:
        return self.committed + self.recovered

    def commit(self, usage: Usage | None) -> None:
        if usage is not None:
            self.committed = self.committed + usage

    def on_error(self, exc: BaseException) -> Decision:
        recovered = Usage(input_tokens=self._recover, total_tokens=self._recover)
        self.recovered = self.recovered + recovered
        return default_decision(exc)

    async def __aenter__(self) -> FakeAttempt:
        if self._fail_entry is not None:
            raise self._fail_entry
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class Harness:
    """A runner over a fake client, with the recorded usage and sleeps."""

    def __init__(
        self,
        steps: list[Step],
        *,
        retry: RetrySettings | None = None,
        fail_entry: Exception | None = None,
    ) -> None:
        self.client = FakeClient(steps)
        self.recorded: list[Usage] = []
        self.slept: list[float] = []

        async def sleep(delay: float) -> None:
            self.slept.append(delay)

        seam = Seam(
            attempt=lambda: FakeAttempt(self.client, fail_entry),
            record=self.recorded.append,
            retry=retry or RetrySettings(max_attempts=4, validation_attempts=2),
            sleep=sleep,
        )
        self.runner = DecisionRunner(seam, model="typesafe/jev", dialect="openrouter")


def _mixed() -> Classifier:
    return parse_classifier(
        {
            "name": "mixed",
            "version": 2,
            "unit": "page",
            "per_request": 2,
            "questions": [
                {"name": "kind", "type": "page_types"},
                {"name": "title", "type": "predicate", "instructions": "A title."},
                {
                    "name": "legible",
                    "type": "score",
                    "instructions": "How legible?",
                    "levels": [{"label": "no"}, {"label": "yes"}],
                },
            ],
        }
    )


async def test_answers_map_back_to_units_and_questions() -> None:
    harness = Harness([_answers])
    request = Request(_mixed(), "book", (_unit(1), _unit(2)))

    result = await harness.runner.ask(request)

    assert harness.client.calls[0][2][:3] == ["kind__01", "title__01", "legible__01"]
    assert [u.unit.ordinal for u in result.units] == [1, 2]
    kind, title, legible = result.units[1].answers
    assert (kind.question, kind.kind, kind.value, kind.probability) == (
        "kind",
        "choice",
        "binding",
        0.9,
    )
    assert (title.kind, title.value, title.probability) == ("predicate", None, 0.75)
    assert (legible.kind, legible.value, legible.probabilities) == (
        "score",
        1.2,
        {"yes": 0.6},
    )
    assert (result.model, result.dialect, result.request_id) == (
        "jev-test",
        "openrouter",
        "dec_1",
    )
    assert result.usage == DecisionUsage(10, 2, 4)
    assert harness.recorded == [
        Usage(input_tokens=10, output_tokens=2, total_tokens=12, cached_tokens=4)
    ]


async def test_a_transient_error_is_retried_and_its_usage_recorded() -> None:
    harness = Harness([httpx.ConnectError("reset"), _answers])

    result = await harness.runner.ask(Request(_classifier(), "book", (_unit(1),)))

    assert result.units[0].answers[0].probability == 0.75
    assert len(harness.client.calls) == 2
    assert len(harness.slept) == 1
    assert harness.recorded[0] == Usage(input_tokens=5, total_tokens=5)
    assert harness.recorded[1].input_tokens == 10


async def test_a_malformed_answer_is_retried_on_the_validation_budget() -> None:
    harness = Harness([MalformedAnswer("q0: no answer"), _answers])
    result = await harness.runner.ask(Request(_classifier(), "book", (_unit(1),)))
    assert result.units[0].answers[0].probability == 0.75

    exhausted = Harness(
        [MalformedAnswer("first"), MalformedAnswer("second"), _answers],
        retry=RetrySettings(max_attempts=4, validation_attempts=2),
    )
    with pytest.raises(MalformedAnswer, match="second"):
        await exhausted.runner.ask(Request(_classifier(), "book", (_unit(1),)))
    assert len(exhausted.recorded) == 2


async def test_a_refusal_is_returned_as_a_refusal() -> None:
    def refuse(questions: Sequence[Question]) -> DecisionResult:
        return DecisionResult(
            {q.name: RefusalAnswer(q.name, "policy") for q in questions},
            "",
            DecisionUsage(),
        )

    harness = Harness([refuse])
    result = await harness.runner.ask(Request(_classifier(), "book", (_unit(1),)))

    answer = result.units[0].answers[0]
    assert (answer.refusal, answer.probability, answer.value) == ("policy", None, None)
    assert result.model == "typesafe/jev"


async def test_timeouts_stop_at_their_own_budget() -> None:
    timeout = httpx.ReadTimeout("slow")
    harness = Harness(
        [timeout, httpx.ReadTimeout("slower"), _answers],
        retry=RetrySettings(max_attempts=8, timeout_attempts=2),
    )

    with pytest.raises(httpx.ReadTimeout, match="slower"):
        await harness.runner.ask(Request(_classifier(), "book", (_unit(1),)))
    assert len(harness.client.calls) == 2


async def test_an_attempt_that_cannot_start_fails_with_its_error() -> None:
    harness = Harness([_answers], fail_entry=OSError("no key"))

    with pytest.raises(OSError, match="no key"):
        await harness.runner.ask(Request(_classifier(), "book", (_unit(1),)))
    assert harness.client.calls == []
    assert harness.recorded[0] == Usage(input_tokens=5, total_tokens=5)


async def test_a_billed_malformed_response_counts_its_usage() -> None:
    bodies = [
        {
            "model": "jev",
            "answers": {},
            "usage": {"input_tokens": 30, "output_tokens": 1},
        },
        {
            "model": "jev",
            "answers": {"q0": {"type": "noul", "noul": 0.6}},
            "usage": {"input_tokens": 20, "output_tokens": 1},
        },
    ]
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=bodies.pop(0)))
    recorded: list[Usage] = []

    async def no_sleep(_: float) -> None:
        return None

    async with httpx.AsyncClient(transport=transport) as http:
        client = DecisionClient(http, dialect="openrouter", api_key="sk-test")
        seam = Seam(
            attempt=lambda: FakeAttempt(client, None, recover=0),
            record=recorded.append,
            retry=RetrySettings(validation_attempts=2),
            sleep=no_sleep,
        )
        runner = DecisionRunner(seam, model="typesafe/jev", dialect="openrouter")
        result = await runner.ask(Request(_classifier(), "book", (_unit(1),)))

    assert result.units[0].answers[0].probability == 0.6
    assert recorded == [
        Usage(input_tokens=30, output_tokens=1, total_tokens=31),
        Usage(input_tokens=20, output_tokens=1, total_tokens=21),
    ]


def _png(width: int, height: int, mode: str = "RGBA") -> bytes:
    buffer = io.BytesIO()
    PILImage.new(mode, (width, height)).save(buffer, "PNG")
    return buffer.getvalue()


def _encoded(image: PILImage.Image, fmt: str, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "data",
    [
        _png(40, 40),
        _encoded(PILImage.new("RGB", (40, 40)), "PNG", transparency=(0, 0, 0)),
        _encoded(PILImage.new("L", (40, 40)), "PNG", transparency=0),
    ],
    ids=["alpha", "rgb-colour-key", "grey-colour-key"],
)
def test_decision_images_flatten_transparency_onto_white(data: bytes) -> None:
    with PILImage.open(io.BytesIO(decision_jpeg(data, 300))) as image:
        assert min(image.getpixel((20, 20))) >= 250


def test_decision_images_follow_the_exif_orientation() -> None:
    exif = PILImage.Exif()
    exif[0x0112] = 6
    data = _encoded(PILImage.new("RGB", (40, 20)), "JPEG", exif=exif)

    with PILImage.open(io.BytesIO(decision_jpeg(data, 300))) as image:
        assert image.size == (20, 40)


def test_decision_images_scale_sixteen_bit_depth() -> None:
    data = _encoded(PILImage.new("I;16", (10, 10), 32768), "PNG")

    with PILImage.open(io.BytesIO(decision_jpeg(data, 300))) as image:
        assert 120 <= min(image.getpixel((5, 5))) <= 136


def test_decision_images_are_rgb_jpegs_within_the_size() -> None:
    data = decision_jpeg(_png(1200, 600), 300)

    with PILImage.open(io.BytesIO(data)) as image:
        assert (image.format, image.mode, image.size) == ("JPEG", "RGB", (300, 150))
    with PILImage.open(io.BytesIO(decision_jpeg(_png(100, 50, "L"), 300))) as small:
        assert small.size == (100, 50)
    assert Image(data).media_type == "image/jpeg"
