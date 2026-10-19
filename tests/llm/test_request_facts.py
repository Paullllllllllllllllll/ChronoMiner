"""Requests follow the settled capability facts in both paths.

Every case builds the synchronous constructor kwargs and, where the provider
has a batch API, the batch body of the same model spec, and checks that both
carry what the vendor documents (the sources are cited in
``common/capability_table.py``): ``flex`` only for the models OpenAI lists,
a reasoning effort or thinking level the model documents, no reasoning on the
chat-latest models, image input and their effort range on the codex models,
the prompt-cache parameter the model takes, and the documented samplers.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI

from chronominer.batch.bodies import BodyPlan
from chronominer.common.batch import BatchRequest
from chronominer.llm.capabilities import ensure_image_support
from chronominer.llm.client import build_chat_model
from chronominer.llm.options import (
    constructor_kwargs,
    model_capabilities,
    request_plan,
)
from chronominer.settings import PromptCaching
from chronominer.spec import ModelSpec
from tests.llm.helpers import model_spec, settings

REQUEST = BatchRequest("req-1", image_base64="UEFHRQ==", mime_type="image/jpeg")


def _paths(spec: ModelSpec, **values: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the synchronous kwargs and the batch body of *spec*."""
    current = settings(**values)
    caps = model_capabilities(spec, current)
    sync = constructor_kwargs(spec, current, caps)
    plan = BodyPlan(
        plan=replace(request_plan(spec, current, caps), limit=8000),
        prompts={"text": "Extract.", "image": "Extract."},
        response_format="json",
        detail="high",
    )
    return sync, plan.body(REQUEST)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "tier", "sync_tier", "batch_tier"),
    [
        ("gpt-6-luna", "flex", "flex", "auto"),
        ("gpt-5.4-mini", "flex", "flex", "auto"),
        ("o3", "flex", "flex", "auto"),
        ("gpt-5.2-pro", "flex", None, None),
        ("gpt-5.3-codex", "flex", None, None),
        ("gpt-5.3-chat-latest", "flex", None, None),
        ("gpt-4o", "flex", None, None),
        # This tool sends a tier to reasoning models only, in both paths.
        ("gpt-4o", "priority", None, None),
    ],
)
def test_flex_goes_only_to_the_listed_models(
    model: str, tier: str, sync_tier: str | None, batch_tier: str | None
) -> None:
    sync, body = _paths(model_spec(model, service_tier=tier))

    assert sync.get("service_tier") == sync_tier
    assert body.get("service_tier") == batch_tier


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "effort", "sent"),
    [
        ("gpt-6-astra", "none", "low"),
        ("gpt-6-astra", "minimal", "low"),
        ("gpt-6.1-sol", "none", "low"),
        ("gpt-6-luna", "minimal", "low"),
        ("gpt-6-luna", "none", "none"),
        ("gpt-6-luna", "max", "max"),
        ("gpt-5.6-terra", "minimal", "low"),
        ("gpt-5.5", "max", "xhigh"),
        ("gpt-5.4-mini", "minimal", "low"),
        ("gpt-5.1", "xhigh", "high"),
        ("gpt-5", "none", "minimal"),
        ("gpt-5", "max", "high"),
        ("gpt-5.5-pro", "low", "medium"),
        ("gpt-5.4-pro", "max", "xhigh"),
        ("gpt-5.3-codex", "none", "low"),
    ],
)
def test_openai_effort_is_one_the_model_documents(
    model: str, effort: str, sent: str
) -> None:
    sync, body = _paths(model_spec(model, reasoning_effort=effort))

    assert sync["reasoning"] == {"effort": sent}
    assert body["reasoning"] == {"effort": sent}


@pytest.mark.unit
def test_an_effort_clamp_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    spec = model_spec("gpt-6-astra-2026-09-01", reasoning_effort="none")
    with caplog.at_level("WARNING"):
        _paths(spec)
        _paths(spec)

    hits = [r for r in caplog.records if "gpt-6-astra-2026-09-01" in r.getMessage()]
    assert len(hits) == 1
    assert "'none'" in hits[0].getMessage() and "'low'" in hits[0].getMessage()


@pytest.mark.unit
@pytest.mark.parametrize("model", ["o1", "o1-mini"])
def test_o1_models_get_no_reasoning_effort(model: str) -> None:
    sync, body = _paths(model_spec(model, reasoning_effort="high"))

    assert "reasoning" not in sync
    assert "reasoning" not in body


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "effort", "level"),
    [
        ("gemini-3.1-pro", "minimal", "low"),
        ("gemini-3.8-flash", "minimal", "low"),
        ("gemini-3.7-flash", "minimal", "low"),
        ("gemini-3-flash", "minimal", "minimal"),
        ("gemma-4-31b-it", "low", "high"),
        ("gemma-4-31b-it", "minimal", "minimal"),
    ],
)
def test_gemini_level_is_one_the_model_documents(
    model: str, effort: str, level: str
) -> None:
    sync, body = _paths(model_spec(model, "google", reasoning_effort=effort))

    assert sync["thinking_level"] == level
    assert body["config"]["thinking_config"] == {"thinking_level": level}


@pytest.mark.unit
@pytest.mark.parametrize("model", ["gemini-3-pro-image", "gemini-3.1-flash-image"])
def test_gemini_image_models_get_no_thinking_level(model: str) -> None:
    sync, body = _paths(model_spec(model, "google", reasoning_effort="high"))

    assert "thinking_level" not in sync
    assert "thinking_config" not in body["config"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "model", ["gpt-5-chat-latest", "gpt-5.2-chat-latest", "gpt-5.3-chat-latest"]
)
def test_chat_latest_models_are_not_reasoning_models(model: str) -> None:
    spec = model_spec(
        model, reasoning_effort="high", verbosity="low", temperature=0.3, top_p=0.5
    )
    sync, body = _paths(spec)

    assert sync["max_tokens"] == 16384
    for key in ("reasoning", "verbosity", "temperature", "top_p"):
        assert key not in sync
        assert key not in body
    assert "verbosity" not in body.get("text", {})


@pytest.mark.unit
@pytest.mark.parametrize("model", ["gpt-5.2-codex", "gpt-5.3-codex"])
def test_codex_models_take_images_and_their_effort_range(model: str) -> None:
    spec = model_spec(model, reasoning_effort="max", verbosity="low")
    sync, body = _paths(spec)

    ensure_image_support(model, model_capabilities(spec, settings()))
    assert sync["reasoning"] == {"effort": "xhigh"}
    assert "verbosity" not in sync
    assert "verbosity" not in body.get("text", {})
    assert body["input"][1]["content"][-1]["detail"] == "high"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "sent"),
    [
        ("gpt-6-luna", {"prompt_cache_options": {"ttl": "30m"}}),
        ("gpt-5.6-terra", {"prompt_cache_options": {"ttl": "30m"}}),
        ("gpt-5.5", {"model_kwargs": {"prompt_cache_retention": "24h"}}),
        ("gpt-4.1", {"model_kwargs": {"prompt_cache_retention": "24h"}}),
        ("gpt-5.4-mini", {}),
        ("gpt-4o", {}),
    ],
)
def test_prompt_cache_parameter_follows_the_model(
    model: str, sent: dict[str, Any]
) -> None:
    caching = PromptCaching(enabled=True, openai_retention="24h")
    sync = constructor_kwargs(model_spec(model), settings(prompt_caching=caching))

    keys = ("prompt_cache_options", "model_kwargs")
    assert {key: sync[key] for key in keys if key in sync} == sent


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "verbosity_sent"),
    [("gpt-6-luna", True), ("gpt-5.1", True), ("gpt-5.5-pro", False), ("o3", False)],
)
def test_verbosity_goes_to_the_documented_models(
    model: str, verbosity_sent: bool
) -> None:
    sync, body = _paths(model_spec(model, verbosity="low"))

    assert (sync.get("verbosity") == "low") is verbosity_sent
    assert (body.get("text", {}).get("verbosity") == "low") is verbosity_sent


@pytest.mark.unit
def test_openrouter_samplers_and_penalties_follow_the_model_list() -> None:
    r1 = constructor_kwargs(
        model_spec("deepseek/deepseek-r1", "openrouter", temperature=0.3),
        settings(),
    )
    gpt4o = constructor_kwargs(model_spec("openai/gpt-4o", "openrouter"), settings())
    opus = constructor_kwargs(
        model_spec("anthropic/claude-opus-5.5", "openrouter", temperature=0.3),
        settings(),
    )

    assert r1["temperature"] == 0.3
    assert (gpt4o["frequency_penalty"], gpt4o["presence_penalty"]) == (0.0, 0.0)
    assert "temperature" not in opus and "top_p" not in opus


@pytest.mark.unit
@pytest.mark.parametrize("model", ["gemini-3.5-flash", "gemini-3.6-flash"])
def test_gemini_flash_takes_no_samplers(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kwargs are left out: the client rejects temperature=None."""
    spec = model_spec(model, "google", temperature=0.3, top_p=0.9)
    sync, body = _paths(spec)

    assert "temperature" not in sync and "top_p" not in sync
    assert "temperature" not in body["config"] and "top_p" not in body["config"]
    monkeypatch.setenv("GOOGLE_API_KEY", "secret")
    assert type(build_chat_model(spec, settings())) is ChatGoogleGenerativeAI
