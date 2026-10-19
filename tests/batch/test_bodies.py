"""Request bodies of batch extraction per provider, chunk kind and format."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from PIL import Image

from chronominer.batch.bodies import BodyPlan, build_plan
from chronominer.common.batch import BatchRequest
from chronominer.common.batch_google import file_request
from chronominer.common.structured import ANTHROPIC_UNION_LIMIT, prepare_request
from chronominer.extract.extraction import (
    ANTHROPIC_SCHEMA_DESCRIPTION,
    anthropic_schema,
)
from chronominer.llm.options import (
    constructor_kwargs,
    model_capabilities,
    output_limit,
    response_format_for,
)
from chronominer.llm.prompts import (
    CONTEXT_IMAGE_LABEL,
    EXTRACTION_SCHEMA_NAME,
    IMAGE_INSTRUCTION,
    extraction_prompt,
    extraction_schema,
)
from chronominer.settings import PromptCaching, Settings
from chronominer.spec import resolve_schema
from tests.batch.conftest import model_spec

SCHEMA = resolve_schema("wage_records")
REQUEST = BatchRequest("req-1", image_base64="UEFHRQ==", mime_type="image/jpeg")
TEXT_REQUEST = BatchRequest("req-2", text="Paid to John Carter, mason, 2s.")


def plan(provider: str, settings: Settings | None = None, **kwargs: Any) -> BodyPlan:
    flags = kwargs.pop("flags", {})
    return build_plan(
        model_spec(provider, kwargs.pop("model", None), **flags),
        settings or Settings(),
        schema=SCHEMA.schema,
        **kwargs,
    )


def bare_schema() -> dict[str, Any]:
    return dict(SCHEMA.schema["schema"])


def _limit(provider: str) -> int | None:
    spec = model_spec(provider)
    return output_limit(spec, model_capabilities(spec, Settings()))


def test_openai_body_enforces_the_schema_by_default() -> None:
    body = plan("openai", detail="original").body(REQUEST)
    assert set(body) == {
        "model",
        "input",
        "max_output_tokens",
        "service_tier",
        "text",
        "reasoning",
    }
    assert body["model"] == "gpt-5.6-luna"
    assert body["max_output_tokens"] == _limit("openai")
    assert body["service_tier"] == "default"
    assert body["reasoning"] == {"effort": "xhigh"}
    assert body["text"] == {
        "format": {
            "type": "json_schema",
            "name": EXTRACTION_SCHEMA_NAME,
            "schema": bare_schema(),
            "strict": True,
        },
        "verbosity": "medium",
    }
    system, user = body["input"]
    assert system["role"] == "system"
    assert "The JSON schema:" in system["content"][0]["text"]
    assert user["content"] == [
        {"type": "input_text", "text": IMAGE_INSTRUCTION},
        {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,UEFHRQ==",
            "detail": "original",
        },
    ]


def test_a_text_chunk_is_the_user_message_with_the_text_prompt() -> None:
    built = plan("openai")
    body = built.body(TEXT_REQUEST)
    system, user = body["input"]
    assert user["content"] == [
        {"type": "input_text", "text": "Paid to John Carter, mason, 2s."}
    ]
    assert system["content"][0]["text"] == built.prompts["text"]
    assert built.prompts["text"] != built.prompts["image"]
    assert IMAGE_INSTRUCTION not in json.dumps(body)
    assert body["text"]["format"]["name"] == EXTRACTION_SCHEMA_NAME


@pytest.mark.parametrize("provider", ["anthropic", "google"])
def test_a_text_chunk_carries_its_text_and_no_image(provider: str) -> None:
    text = json.dumps(plan(provider).body(TEXT_REQUEST))

    assert "Paid to John Carter" in text
    assert IMAGE_INSTRUCTION not in text


@pytest.mark.parametrize("kind", ["text", "image"])
def test_the_system_prompts_are_those_of_the_sync_path(kind: str) -> None:
    built = plan("openai", context="Register of 1791.")
    expected = extraction_prompt(kind, SCHEMA.schema, context="Register of 1791.")
    assert built.prompts[kind] == expected


def test_openai_json_format_keeps_the_schema_in_the_prompt_only() -> None:
    body = plan("openai", flags={"response_format": "json"}).body(REQUEST)
    assert body["text"] == {"verbosity": "medium"}
    assert "The JSON schema:" in body["input"][0]["content"][0]["text"]


def test_anthropic_body_enforces_the_schema_by_default() -> None:
    body = plan("anthropic").body(REQUEST)
    assert set(body) == {
        "model",
        "max_tokens",
        "system",
        "messages",
        "thinking",
        "output_config",
    }
    assert body["max_tokens"] == _limit("anthropic")
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": IMAGE_INSTRUCTION},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "UEFHRQ==",
            },
        },
    ]


def test_anthropic_json_format_sends_no_output_format() -> None:
    body = plan("anthropic", flags={"response_format": "json"}).body(REQUEST)
    assert "format" not in body.get("output_config", {})
    (system,) = body["system"]
    assert "The JSON schema:" in system["text"]


def sync_anthropic_schema(spec: dict[str, Any]) -> Any:
    """Return the ``output_config.format`` schema a synchronous call sends."""
    model = ChatAnthropic(
        api_key="unused",  # type: ignore[arg-type]
        model_name="claude-sonnet-4-6",
        timeout=None,
        stop=None,
    )
    request_schema = anthropic_schema(extraction_schema(spec))
    prepared = prepare_request(model, "anthropic", "schema", request_schema)
    return prepared.runnable.first.steps__["raw"].kwargs["output_config"]["format"][
        "schema"
    ]


def nullable_schema(count: int) -> dict[str, Any]:
    """Return an entries schema with *count* nullable fields and unsupported
    keywords."""
    item: dict[str, Any] = {
        f"field_{index}": {
            "type": ["integer", "null"],
            "minimum": 1,
            "maximum": 7,
            "description": "Use null if unknown.",
        }
        for index in range(count)
    }
    item["kind"] = {"type": ["string", "null"], "enum": ["a", "b", None]}
    properties: dict[str, Any] = {
        "entries": {
            "type": "array",
            "maxItems": 50,
            "items": {
                "type": "object",
                "properties": item,
                "required": sorted(item),
                "additionalProperties": False,
            },
        }
    }
    return {
        "name": "Nullables",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": ["entries"],
            "additionalProperties": False,
        },
    }


@pytest.mark.parametrize("spec", [SCHEMA.schema, nullable_schema(3)])
def test_anthropic_batch_schema_is_the_synchronous_schema(
    spec: dict[str, Any],
) -> None:
    built = build_plan(model_spec("anthropic"), Settings(), schema=spec)
    schema = built.body(REQUEST)["output_config"]["format"]["schema"]
    assert schema == sync_anthropic_schema(spec)
    assert schema["title"] == EXTRACTION_SCHEMA_NAME
    assert schema["description"] == ANTHROPIC_SCHEMA_DESCRIPTION
    sent = json.dumps(schema)
    assert all(f'"{key}"' not in sent for key in ("minimum", "maximum", "maxItems"))


def test_anthropic_nullable_fields_keep_null_with_their_keywords() -> None:
    built = build_plan(model_spec("anthropic"), Settings(), schema=nullable_schema(1))
    schema = built.body(REQUEST)["output_config"]["format"]["schema"]
    item = schema["properties"]["entries"]["items"]["properties"]
    rating, kind = item["field_0"], item["kind"]
    assert rating["anyOf"][1] == {"type": "null"}
    assert rating["anyOf"][0]["type"] == "integer"
    assert rating["description"] == "Use null if unknown."
    assert "minimum: 1" in rating["anyOf"][0]["description"]
    assert kind["anyOf"] == [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}]


@pytest.mark.parametrize(
    ("count", "fmt"),
    [(ANTHROPIC_UNION_LIMIT - 1, "schema"), (ANTHROPIC_UNION_LIMIT, "json")],
)
def test_an_anthropic_schema_over_the_union_limit_uses_prompted_json(
    count: int, fmt: str
) -> None:
    """*count* rated fields plus the nullable ``kind`` field."""
    built = build_plan(
        model_spec("anthropic"), Settings(), schema=nullable_schema(count)
    )
    body = built.body(REQUEST)
    assert built.response_format == fmt
    assert ("format" in body["output_config"]) is (fmt == "schema")
    (system,) = body["system"]
    assert "The JSON schema:" in system["text"]


def test_anthropic_caching_marks_the_system_prompt() -> None:
    off = Settings(prompt_caching=PromptCaching(enabled=False))
    assert isinstance(plan("anthropic", off).body(REQUEST)["system"], str)
    (default,) = plan("anthropic").body(REQUEST)["system"]
    assert default["cache_control"] == {"type": "ephemeral"}
    settings = Settings(prompt_caching=PromptCaching(enabled=True, anthropic_ttl="1h"))
    body = plan("anthropic", settings).body(REQUEST)
    (block,) = body["system"]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize(
    ("model", "enabled", "expected"),
    [
        ("gpt-5.6-luna", True, {"prompt_cache_options": {"ttl": "30m"}}),
        ("gpt-5.4", True, {"prompt_cache_retention": "24h"}),
        ("gpt-5.6-luna", False, {}),
    ],
)
def test_openai_batch_bodies_carry_the_cache_parameter_of_sync_requests(
    model: str, enabled: bool, expected: dict[str, Any]
) -> None:
    caching = PromptCaching(enabled=enabled, openai_retention="24h")
    body = plan("openai", Settings(prompt_caching=caching), model=model).body(REQUEST)
    keys = ("prompt_cache_options", "prompt_cache_retention")
    assert {key: body[key] for key in keys if key in body} == expected


def test_google_body_takes_the_family_output_limit() -> None:
    body = plan("google").body(REQUEST)
    assert body["model"] == "gemini-3-flash-preview"
    assert body["config"]["max_output_tokens"] == _limit("google")
    assert body["config"]["temperature"] == 1.0
    assert body["contents"][0]["parts"][0] == {"text": IMAGE_INSTRUCTION}


@pytest.mark.parametrize("response_format", [None, "schema"])
def test_google_schema_format_sends_the_json_schema(
    response_format: str | None,
) -> None:
    flags = {"response_format": response_format} if response_format else {}
    body = plan("google", flags=flags).body(REQUEST)
    assert body["config"]["response_mime_type"] == "application/json"
    assert body["config"]["response_json_schema"] == bare_schema()


def test_google_json_format_sends_no_json_schema() -> None:
    body = plan("google", flags={"response_format": "json"}).body(REQUEST)
    assert "response_json_schema" not in body["config"]
    assert "response_mime_type" not in body["config"]


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "gpt-5.6-luna", "schema"),
        ("anthropic", "claude-sonnet-4-6", "schema"),
        ("google", "gemini-2.5-flash", "schema"),
        ("openai", "o3-deep-research", "json"),
        ("anthropic", "claude-3-5-haiku", "json"),
    ],
)
def test_batch_and_sync_pick_the_same_default_response_format(
    provider: str, model: str, expected: str
) -> None:
    spec = model_spec(provider, model)
    caps = model_capabilities(spec, Settings())
    batch = build_plan(spec, Settings(), schema=SCHEMA.schema)
    assert spec.response_format is None
    assert batch.response_format == response_format_for(spec, caps) == expected


_SAMPLERS = ("temperature", "top_p")
_EFFORTS = ("none", "low", "high", "max")


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    ("model", "max_output_tokens"),
    [
        ("claude-opus-5-5", None),
        ("claude-sonnet-4-6", None),
        ("claude-opus-4-5", None),
        ("claude-sonnet-4-5", None),
        ("claude-opus-4-1", 2000),
        ("claude-sonnet-4-5", 1200),
        ("claude-3-5-haiku", None),
    ],
)
def test_anthropic_batch_thinking_matches_the_sync_client(
    effort: str, model: str, max_output_tokens: int | None
) -> None:
    flags: dict[str, Any] = {"reasoning_effort": effort}
    if max_output_tokens is not None:
        flags["max_output_tokens"] = max_output_tokens
    spec = model_spec("anthropic", model, **flags)
    kwargs = constructor_kwargs(spec, Settings())

    body = build_plan(spec, Settings(), schema=SCHEMA.schema).body(REQUEST)

    assert body.get("thinking") == kwargs.get("thinking")
    assert body.get("output_config", {}).get("effort") == kwargs.get("effort")
    assert body["max_tokens"] == kwargs["max_tokens"]
    if "thinking" in body:
        assert not set(_SAMPLERS) & set(body)
    else:
        assert body.get("temperature") == kwargs.get("temperature")


def test_anthropic_batch_thinking_follows_the_effort() -> None:
    budgets = {
        effort: plan(
            "anthropic", flags={"reasoning_effort": effort}, model="claude-sonnet-4-5"
        )
        for effort in ("none", "low", "high")
    }
    assert "thinking" not in budgets["none"].body(REQUEST)
    assert budgets["low"].body(REQUEST)["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }
    assert budgets["high"].body(REQUEST)["thinking"]["budget_tokens"] == 16384


@pytest.mark.parametrize("response_format", ["schema", "json"])
def test_anthropic_batch_body_carries_the_effort(response_format: str) -> None:
    bodies = {
        effort: plan(
            "anthropic",
            flags={"reasoning_effort": effort, "response_format": response_format},
            model="claude-opus-5-5",
        ).body(REQUEST)
        for effort in ("low", "high")
    }
    assert bodies["low"] != bodies["high"]
    assert bodies["low"]["output_config"]["effort"] == "low"
    assert bodies["high"]["output_config"]["effort"] == "high"
    assert bodies["low"]["thinking"] == {"type": "adaptive"}
    if response_format == "schema":
        assert bodies["low"]["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_batch_body_without_effort_ignores_the_level() -> None:
    bodies = [
        plan(
            "anthropic", flags={"reasoning_effort": effort}, model="claude-3-7-sonnet"
        ).body(REQUEST)
        for effort in ("low", "high")
    ]
    assert bodies[0] == bodies[1]
    assert "effort" not in bodies[0].get("output_config", {})


@pytest.mark.parametrize("effort", _EFFORTS)
@pytest.mark.parametrize(
    "model", ["gemini-3-flash-preview", "gemini-3.1-pro", "gemini-2.5-flash"]
)
def test_google_batch_thinking_matches_the_sync_client(effort: str, model: str) -> None:
    spec = model_spec("google", model, reasoning_effort=effort)
    kwargs = constructor_kwargs(spec, Settings())

    body = build_plan(spec, Settings(), schema=SCHEMA.schema).body(REQUEST)

    level = kwargs.get("thinking_level")
    budget = kwargs.get("thinking_budget")
    expected = None
    if level is not None:
        expected = {"thinking_level": level}
    elif budget is not None:
        expected = {"thinking_budget": budget}
    assert body["config"].get("thinking_config") == expected
    if model == "gemini-3-flash-preview" and effort == "low":
        assert expected == {"thinking_level": "low"}
    if model == "gemini-2.5-flash" and effort == "high":
        assert budget is not None


@pytest.mark.parametrize("effort", ["low", "high"])
def test_openai_batch_body_carries_the_effort_and_no_thinking(effort: str) -> None:
    body = plan("openai", flags={"reasoning_effort": effort}).body(REQUEST)
    assert body["reasoning"] == {"effort": effort}
    assert "thinking" not in body
    assert "thinking_config" not in json.dumps(body)


def test_openai_samplers_match_the_sync_client() -> None:
    spec = model_spec("openai", "gpt-4o", temperature=0.2)
    kwargs = constructor_kwargs(spec, Settings())

    body = build_plan(spec, Settings(), schema=SCHEMA.schema).body(REQUEST)

    samplers = ("temperature", "top_p", "frequency_penalty", "presence_penalty")
    assert {name: body.get(name) for name in samplers} == {
        name: kwargs.get(name) for name in samplers
    }
    assert body["temperature"] == 0.2


@pytest.mark.parametrize(
    "model", ["gemini-3-flash-preview", "gemini-3.1-pro", "gemini-2.5-flash"]
)
def test_google_samplers_match_the_sync_client(model: str) -> None:
    spec = model_spec("google", model, temperature=0.2, top_p=0.9)
    kwargs = constructor_kwargs(spec, Settings())

    body = build_plan(spec, Settings(), schema=SCHEMA.schema).body(REQUEST)

    assert {name: body["config"].get(name) for name in _SAMPLERS} == {
        name: kwargs.get(name) for name in _SAMPLERS
    }


@pytest.mark.parametrize(
    ("plan_detail", "request_detail", "expected"),
    [
        ("high", None, "MEDIA_RESOLUTION_HIGH"),
        ("high", "low", "MEDIA_RESOLUTION_LOW"),
        ("auto", None, None),
        (None, None, None),
    ],
)
def test_a_google_page_carries_the_media_resolution_of_its_detail(
    plan_detail: str | None, request_detail: str | None, expected: str | None
) -> None:
    request = replace(REQUEST, image_detail=request_detail)
    body = plan("google", detail=plan_detail).body(request)
    image = body["contents"][0]["parts"][1]
    rest = file_request(body)["contents"][0]["parts"][1]
    level = image.get("media_resolution", {}).get("level")
    assert level == rest.get("media_resolution", {}).get("level") == expected


def test_a_set_output_limit_is_clamped_to_the_family_on_google() -> None:
    body = plan("google", flags={"max_output_tokens": 500000}).body(REQUEST)
    assert body["config"]["max_output_tokens"] == _limit("google")


@pytest.mark.parametrize("request_", [REQUEST, TEXT_REQUEST], ids=["image", "text"])
def test_the_context_image_precedes_the_chunk_on_openai(
    request_: BatchRequest, tmp_path: Path
) -> None:
    image = tmp_path / "context.png"
    Image.new("L", (40, 40), "white").save(image)
    body = plan("openai", context_image=image, context="Register of 1791.").body(
        request_
    )
    text = json.dumps(body)
    chunk = IMAGE_INSTRUCTION if request_.is_visual else "Paid to John Carter"
    assert text.index(CONTEXT_IMAGE_LABEL) < text.index(chunk)
    assert "Register of 1791." in text
    label, context = body["input"][1]["content"][:2]
    assert label == {"type": "input_text", "text": CONTEXT_IMAGE_LABEL}
    assert context["image_url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("provider", ["anthropic", "google"])
def test_other_providers_get_no_context_image_as_in_sync_calls(
    provider: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image = tmp_path / "context.png"
    Image.new("L", (40, 40), "white").save(image)
    built = plan(provider, context_image=image, context="Register of 1791.")
    text = json.dumps(built.body(REQUEST))
    assert built.context_image is None
    assert CONTEXT_IMAGE_LABEL not in text
    assert "Register of 1791." in text
    assert "not sent to provider" in caplog.text


def test_original_detail_becomes_high_where_the_model_lacks_it() -> None:
    body = plan("openai", model="gpt-4o", detail="original").body(REQUEST)
    assert body["input"][1]["content"][-1]["detail"] == "high"


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_a_png_page_is_sent_as_png(provider: str, tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    Image.new("L", (32, 48), 128).save(path)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    body = plan(provider).body(
        BatchRequest("req-1", image_base64=data, mime_type="image/png")
    )
    text = json.dumps(body)
    assert "image/png" in text and data in text


def test_a_request_detail_replaces_the_plan_detail() -> None:
    request = BatchRequest(
        "req-1", image_base64="UEFHRQ==", mime_type="image/png", image_detail="low"
    )
    body = plan("openai", detail="original").body(request)
    image = body["input"][1]["content"][-1]
    assert image["detail"] == "low"
    assert image["image_url"].startswith("data:image/png;base64,")


def test_providers_without_a_batch_api_are_refused() -> None:
    routed = replace(model_spec("openai"), provider="openrouter")
    with pytest.raises(ValueError, match="no batch API"):
        build_plan(routed, Settings(), schema=SCHEMA.schema)


def test_a_request_without_text_or_image_is_refused() -> None:
    with pytest.raises(ValueError, match="no text or image"):
        plan("openai").body(BatchRequest("req-1"))
