"""Characterization of the request options per provider, model and option.

Every case records what one model spec sends: the constructor kwargs of the
synchronous chat model, the image blocks of a synchronous page request and,
for the providers with a batch API, the batch request body of a page (and,
with the default options, of a text chunk). The cases cover the providers,
representative models and the options that shape a request (reasoning effort,
verbosity, temperature, top_p, service tier, output limit and image detail).
``golden/request_matrix.json`` holds the requests of v1.0.0 before the
request options moved to ``common/request_options.py``; the move left every
case identical. The settled capability facts changed the cases listed in
:data:`FACT_CHANGES`, each with the fact it follows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from typing import Any

import pytest

from chronominer.batch.bodies import BodyPlan, build_plan
from chronominer.common.batch import BatchRequest
from chronominer.common.capabilities import Capabilities
from chronominer.common.testing import WriteGuard
from chronominer.extract.extraction import ChunkExtractor
from chronominer.extract.images import resolve_image_settings
from chronominer.llm.options import constructor_kwargs, request_plan
from chronominer.spec import ImageSpec, ModelSpec, resolve_schema
from tests.llm.helpers import CUSTOM_ENDPOINT, custom_spec, env, model_spec, settings
from tests.matrix_golden import check_matrix

MODELS: tuple[tuple[str, str], ...] = (
    ("openai", "gpt-6-luna"),
    ("openai", "gpt-5.6-luna"),
    ("openai", "gpt-5.3"),
    ("anthropic", "claude-haiku-5-5"),
    ("anthropic", "claude-sonnet-4-0"),
    ("google", "gemini-3.1-pro"),
    ("google", "gemini-2.5-flash"),
    ("openrouter", "anthropic/claude-sonnet-4.5"),
    ("custom", "org/fake-model"),
)
BATCH_PROVIDERS = frozenset({"openai", "anthropic", "google"})

VARIANTS: Mapping[str, Mapping[str, Any]] = {
    "defaults": {},
    "effort-none": {"reasoning_effort": "none"},
    "effort-high": {"reasoning_effort": "high"},
    "effort-max": {"reasoning_effort": "max"},
    "effort-none-limit-2000": {"reasoning_effort": "none", "max_output_tokens": 2000},
    "effort-max-limit-2000": {"reasoning_effort": "max", "max_output_tokens": 2000},
    "verbosity-low": {"verbosity": "low"},
    "verbosity-high": {"verbosity": "high"},
    "temperature-0.3": {"temperature": 0.3},
    "top-p-0.5": {"top_p": 0.5},
    "samplers-effort-none": {
        "reasoning_effort": "none",
        "temperature": 0.3,
        "top_p": 0.5,
    },
    "tier-flex": {"service_tier": "flex"},
    "tier-priority": {"service_tier": "priority"},
    "tier-default": {"service_tier": "default"},
    "tier-auto": {"service_tier": "auto"},
    "limit-unset": {"max_output_tokens": None},
    "limit-2000": {"max_output_tokens": 2000},
    "limit-500000": {"max_output_tokens": 500000},
}
DETAILS: tuple[str | None, ...] = (None, "low", "high", "auto", "original")
SCHEMA = resolve_schema("wage_records")
TEXT_REQUEST = BatchRequest("req-2", text="Paid to John Carter, mason, 2s.")
FACT_CHANGES: Mapping[str, str] = {
    "openai:gpt-5.3:": (
        "OpenAI documents no gpt-5.3 id (no model page, no pricing row): the "
        "id takes the conservative profile, so no image input, no structured "
        "output and no family output limit."
    ),
    "anthropic:claude-sonnet-4-0:": (
        "The retired Claude Sonnet 4 is not in Anthropic's structured-output "
        "model list: prompted JSON instead of the enforced schema."
    ),
}
"""The golden cases, by case-name prefix, that the capability facts changed."""
KEY_VARIABLES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    CUSTOM_ENDPOINT.api_key_env,
)


def _spec(provider: str, model: str, **changes: Any) -> ModelSpec:
    if provider == "custom":
        return custom_spec(**changes)
    return model_spec(model, provider, **changes)


def _outcome(build: Callable[..., Any], *args: Any) -> Any:
    """Return what *build* returns for *args*, or the error it raises."""
    try:
        return build(*args)
    except Exception as exc:  # noqa: BLE001 - the error is the recorded outcome
        return {"<error>": f"{type(exc).__name__}: {exc}"}


def _image_blocks(spec: ModelSpec, detail: str | None) -> Any:
    extractor = ChunkExtractor(
        spec,
        env(),
        schema=SCHEMA.schema,
        schema_name=SCHEMA.name,
        images=ImageSpec(detail=detail),
    )
    _system, human = extractor.image_messages("UEFHRQ==", "image/jpeg")
    return [
        block
        for block in human.content
        if isinstance(block, dict) and block.get("type") != "text"
    ]


def _batch_bodies(spec: ModelSpec, detail: str | None, text: bool) -> Any:
    images = ImageSpec(detail=detail)
    resolved = resolve_image_settings(spec.provider, spec.model, images)[0]
    plan = build_plan(
        spec,
        settings(),
        schema=SCHEMA.schema,
        detail=resolved["resolved_detail"],
        images=images,
    )
    page = BatchRequest(
        "req-1",
        image_base64="UEFHRQ==",
        mime_type="image/jpeg",
        image_detail=plan.detail,
    )
    bodies = {"page": plan.body(page)}
    if text:
        bodies["text"] = plan.body(TEXT_REQUEST)
    return bodies


def _case(spec: ModelSpec, detail: str | None, text: bool = False) -> dict[str, Any]:
    case: dict[str, Any] = {
        "sync": _outcome(constructor_kwargs, spec, settings()),
        "sync_images": _outcome(_image_blocks, spec, detail),
    }
    if spec.provider in BATCH_PROVIDERS:
        case["batch"] = _outcome(_batch_bodies, spec, detail, text)
    return case


def _cases() -> Iterator[tuple[str, dict[str, Any]]]:
    for provider, model in MODELS:
        for name, changes in VARIANTS.items():
            spec = _spec(provider, model, **changes)
            yield f"{provider}:{model}:{name}", _case(spec, None, name == "defaults")
        for detail in DETAILS[1:]:
            spec = _spec(provider, model)
            yield f"{provider}:{model}:detail-{detail}", _case(spec, detail)


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set every key variable, so the synchronous clients can be built."""
    for name in KEY_VARIABLES:
        monkeypatch.setenv(name, f"test-{name}")


@pytest.mark.unit
@pytest.mark.usefixtures("keys")
def test_requests_match_the_golden(write_guard: WriteGuard) -> None:
    check_matrix(__file__, "request_matrix", dict(_cases()), write_guard.consume)


@pytest.mark.unit
def test_every_fact_change_names_golden_cases() -> None:
    names = [name for name, _ in _cases()]
    for prefix in FACT_CHANGES:
        assert any(name.startswith(prefix) for name in names), prefix


def _verbosity_caps(*, reasoning: bool, verbosity: bool) -> Capabilities:
    return Capabilities(
        provider_name="openai",
        model_name="verbosity-model",
        family="verbosity-model",
        supports_vision=True,
        is_reasoning_model=reasoning,
        supports_reasoning_effort=reasoning,
        supports_text_verbosity=verbosity,
        supports_temperature=not reasoning,
        supports_top_p=not reasoning,
        max_output_tokens=8000,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reasoning", "verbosity"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_verbosity_follows_the_table_in_sync_and_batch(
    reasoning: bool, verbosity: bool
) -> None:
    """Both paths send the verbosity where ``supports_text_verbosity`` is set.

    This settles the earlier split, in which a synchronous request keyed the
    verbosity on ``is_reasoning_model`` and a batch body on
    ``supports_reasoning_effort``. OpenAI documents the verbosity for its
    GPT-5 and GPT-6 reasoning models, which the table marks.
    """
    caps = _verbosity_caps(reasoning=reasoning, verbosity=verbosity)
    spec = model_spec("verbosity-model", verbosity="low")

    sync = constructor_kwargs(spec, settings(), caps)
    plan = BodyPlan(
        plan=replace(request_plan(spec, settings(), caps), limit=8000),
        prompts={"text": "Extract.", "image": "Extract."},
        response_format="json",
    )
    body = plan.body(TEXT_REQUEST)

    assert (sync.get("verbosity") == "low") is verbosity
    assert (body.get("text", {}).get("verbosity") == "low") is verbosity
