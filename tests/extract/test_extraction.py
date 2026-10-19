"""ChunkExtractor: messages per provider, response formats, results, retries."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from PIL import Image

import chronominer.llm.caller as caller_module
from chronominer.common.testing import FakeRequest
from chronominer.common.usage import Usage
from chronominer.extract.extraction import (
    ANTHROPIC_SCHEMA_DESCRIPTION,
    ChunkExtractor,
    usage_record,
)
from chronominer.extract.response_parsing import INVALID_JSON, NO_OBJECT
from chronominer.llm.caller import CallEnv
from chronominer.llm.prompts import EXTRACTION_SCHEMA_NAME, IMAGE_INSTRUCTION
from chronominer.settings import PromptCaching, RetrySettings
from chronominer.spec import ImageSpec, ModelSpec, resolve_schema
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.helpers import RecordingExt, StatusError, message, model_spec

PAGE = "QUJD"
SCHEMA = resolve_schema("WageRecords")
ANSWER: dict[str, Any] = {
    "contains_no_content_of_requested_type": False,
    "entries": [
        {
            "person": "John Carter",
            "occupation": "mason",
            "wage": 12.5,
            "currency": "shillings",
            "period": "week",
            "date": "1701",
            "employer": None,
            "source_page": None,
            "notes": None,
        }
    ],
}
SPECS: dict[str, ModelSpec] = {
    "openai": model_spec(),
    "anthropic": model_spec("claude-sonnet-4-5", "anthropic"),
    "google": model_spec("gemini-3-flash", "google"),
    "openrouter": model_spec("anthropic/claude-sonnet-4.5", "openrouter"),
    "custom": helpers.custom_spec(),
}


def usage_dict(
    input_tokens: int, output_tokens: int, total_tokens: int, *, cached: int = 0
) -> dict[str, int]:
    """Return the stored usage shape of these counts."""
    return Usage(input_tokens, output_tokens, total_tokens, cached).as_dict()


def _extractor(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[Any],
    spec: ModelSpec | None = None,
    *,
    env: CallEnv | None = None,
    **options: Any,
) -> tuple[ChunkExtractor, helpers.Script]:
    script = helpers.install(monkeypatch, answers)
    options.setdefault("schema", SCHEMA.schema)
    options.setdefault("schema_name", SCHEMA.name)
    extractor = ChunkExtractor(spec or model_spec(), env or helpers.env(), **options)
    return extractor, script


def _page(extractor: ChunkExtractor, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("label", "page_0001")
    return asyncio.run(
        extractor.extract_image_from_base64(PAGE, "image/jpeg", **kwargs)
    )


def _text(extractor: ChunkExtractor, text: str = "Record 001.") -> dict[str, Any]:
    return asyncio.run(extractor.extract_text(text, label="1-17"))


def _human(request: FakeRequest) -> Any:
    return request.messages[-1].content


def _system(request: FakeRequest) -> Any:
    return request.messages[0].content


def _context_image(tmp_path: Path) -> Path:
    path = tmp_path / "title.png"
    Image.new("RGB", (40, 60), (200, 10, 10)).save(path)
    return path


def test_the_openai_request_carries_detail_and_the_enforced_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER])

    result = _page(extractor)

    (request,) = script.requests
    instruction, block = _human(request)
    assert instruction == {"type": "text", "text": IMAGE_INSTRUCTION}
    assert block == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{PAGE}", "detail": "original"},
    }
    assert request.kwargs["text"]["format"]["name"] == EXTRACTION_SCHEMA_NAME
    assert "The JSON schema:" in str(_system(request))
    assert json.loads(result["output_text"]) == ANSWER
    assert "error" not in result


def test_a_text_chunk_is_the_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER])

    result = _text(extractor, "Record 001.\nItem: wheat")

    (request,) = script.requests
    assert _human(request) == "Record 001.\nItem: wheat"
    assert "input text" in str(_system(request)).lower()
    assert json.loads(result["output_text"]) == ANSWER


@pytest.mark.parametrize(
    ("model", "requested", "sent"),
    [
        ("gpt-5.6-luna", None, "original"),
        ("gpt-5.6-luna", "low", "low"),
        ("o1-pro", "original", "high"),
        ("gpt-4o", "auto", "auto"),
        ("gpt-4o", "nonsense", "high"),
    ],
)
def test_openai_keeps_only_the_detail_tiers_the_model_takes(
    monkeypatch: pytest.MonkeyPatch, model: str, requested: str | None, sent: str
) -> None:
    extractor, script = _extractor(
        monkeypatch, [ANSWER], model_spec(model), images=ImageSpec(detail=requested)
    )

    _page(extractor)

    assert _human(script.requests[0])[-1]["image_url"]["detail"] == sent


def test_a_per_call_detail_replaces_the_page_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER])

    _page(extractor, image_detail="low")

    assert _human(script.requests[0])[-1]["image_url"]["detail"] == "low"


@pytest.mark.parametrize(
    ("provider", "page_block"),
    [
        (
            "anthropic",
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": PAGE},
            },
        ),
        (
            "google",
            {
                "type": "image",
                "base64": PAGE,
                "mime_type": "image/jpeg",
                "media_resolution": "MEDIA_RESOLUTION_HIGH",
            },
        ),
        (
            "openrouter",
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{PAGE}"},
            },
        ),
        (
            "custom",
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{PAGE}"},
            },
        ),
    ],
)
def test_each_provider_gets_its_image_block(
    monkeypatch: pytest.MonkeyPatch, provider: str, page_block: dict[str, Any]
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER], SPECS[provider])

    _page(extractor)

    assert _human(script.requests[0])[-1] == page_block


def test_a_gemini_detail_without_a_resolution_sends_no_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER], SPECS["google"])

    _page(extractor, image_detail="nonsense")

    assert _human(script.requests[0])[-1] == {
        "type": "image",
        "base64": PAGE,
        "mime_type": "image/jpeg",
    }


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
def test_the_context_image_precedes_the_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str
) -> None:
    spec = (
        SPECS["openai"]
        if provider == "openai"
        else model_spec("openai/gpt-5.4", "openrouter")
    )
    extractor, script = _extractor(monkeypatch, [ANSWER], spec)
    extractor.update_context_image(_context_image(tmp_path))

    _page(extractor)

    blocks = _human(script.requests[0])
    assert [block["type"] for block in blocks] == [
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    assert blocks[0]["text"] == "Context image:"
    assert blocks[2]["text"] == IMAGE_INSTRUCTION
    assert blocks[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_the_context_image_comes_before_a_text_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER])
    extractor.update_context_image(_context_image(tmp_path))

    _text(extractor, "Record 002.")

    blocks = _human(script.requests[0])
    assert [block["type"] for block in blocks] == ["text", "image_url", "text"]
    assert blocks[2] == {"type": "text", "text": "Record 002."}


@pytest.mark.parametrize("provider", ["anthropic", "google", "custom"])
def test_other_providers_skip_the_context_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER], SPECS[provider])

    with caplog.at_level(logging.WARNING):
        extractor.update_context_image(_context_image(tmp_path))
    _page(extractor)

    assert len(_human(script.requests[0])) == 2
    assert "Context images are not sent" in caplog.text


def test_the_context_image_detail_follows_the_page_tiers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    extractor, script = _extractor(
        monkeypatch, [ANSWER], model_spec("gpt-4o"), images=ImageSpec(detail="low")
    )
    extractor.update_context_image(_context_image(tmp_path))

    _page(extractor)

    assert _human(script.requests[0])[1]["image_url"]["detail"] == "low"


def test_a_missing_or_cleared_context_image_sends_two_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER])
    extractor.update_context_image(_context_image(tmp_path))
    extractor.update_context_image(None)
    extractor.update_context_image(tmp_path / "missing.png")

    _page(extractor)

    assert len(_human(script.requests[0])) == 2


def test_openrouter_sends_only_the_tiers_it_takes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = model_spec("openai/gpt-5.4", "openrouter")
    extractor, script = _extractor(
        monkeypatch, [ANSWER, ANSWER], spec, images=ImageSpec(detail="original")
    )

    _page(extractor)
    _page(extractor, image_detail="low")

    first, second = script.requests
    assert _human(first)[-1]["image_url"]["detail"] == "high"
    assert _human(second)[-1]["image_url"]["detail"] == "low"


def test_the_context_text_fills_and_empties_its_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER])

    extractor.update_context("Register of the city market of Lyon.")
    _page(extractor)
    _text(extractor)
    extractor.update_context(None)
    _page(extractor)

    with_page, with_text, without = (str(_system(r)) for r in script.requests)
    assert "Register of the city market of Lyon." in with_page
    assert "Register of the city market of Lyon." in with_text
    assert "{{CONTEXT}}" not in with_page + with_text + without
    assert "Register of the city market" not in without


@pytest.mark.parametrize(("fmt", "enforced"), [("schema", True), ("json", False)])
def test_the_response_format_picks_the_enforcement(
    monkeypatch: pytest.MonkeyPatch, fmt: str, enforced: bool
) -> None:
    extractor, script = _extractor(
        monkeypatch, [ANSWER], model_spec(response_format=fmt)
    )

    result = _page(extractor)

    (request,) = script.requests
    assert "The JSON schema:" in str(_system(request))
    assert ("text" in request.kwargs) is enforced
    assert json.loads(result["output_text"]) == ANSWER
    assert "error" not in result


def test_the_anthropic_schema_is_titled_and_keeps_its_nullable_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER], SPECS["anthropic"])
    original = json.dumps(extractor.request_schema, sort_keys=True)

    result = _page(extractor)

    structured = script.requests[0].structured
    assert structured is not None
    schema = structured.schema
    assert schema["title"] == EXTRACTION_SCHEMA_NAME
    assert schema["description"] == ANTHROPIC_SCHEMA_DESCRIPTION
    entry = schema["properties"]["entries"]["items"]["properties"]
    assert {"type": "null"} in entry["employer"]["anyOf"]
    assert json.dumps(extractor.request_schema, sort_keys=True) == original
    assert json.loads(result["output_text"])["entries"][0]["employer"] is None


def test_an_anthropic_schema_over_the_union_limit_is_prompted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    wide = json.loads(json.dumps(SCHEMA.schema))
    item = wide["schema"]["properties"]["entries"]["items"]
    for index in range(17):
        item["properties"][f"extra_{index}"] = {"type": ["string", "null"]}
        item["required"].append(f"extra_{index}")
    caplog.set_level(logging.INFO, logger="chronominer.common.structured")
    extractor, script = _extractor(
        monkeypatch, [ANSWER, ANSWER], SPECS["anthropic"], schema=wide
    )

    _page(extractor)
    _page(extractor)

    assert extractor.response_format == "json"
    assert all(request.structured is None for request in script.requests)
    assert "The JSON schema:" in str(_system(script.requests[0]))
    assert extractor.request().required_keys == tuple(wide["schema"]["required"])
    notes = [r for r in caplog.records if "union parameters" in r.getMessage()]
    assert len(notes) == 1 and notes[0].levelno == logging.INFO


def test_a_bare_schema_is_sent_under_the_extraction_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare = dict(SCHEMA.schema["schema"])
    extractor, script = _extractor(monkeypatch, [ANSWER], schema=bare)

    _page(extractor)

    text_format = script.requests[0].kwargs["text"]["format"]
    assert (text_format["name"], text_format["strict"]) == (
        EXTRACTION_SCHEMA_NAME,
        True,
    )
    assert text_format["schema"] == bare


@pytest.mark.parametrize("provider", ["google", "openrouter", "custom"])
def test_the_selected_schema_is_enforced_on_every_provider(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER], SPECS[provider])

    result = _page(extractor)

    request = script.requests[0]
    body = SCHEMA.schema["schema"]
    if provider == "google":
        assert request.kwargs["response_schema"] == body
    elif provider == "custom":
        sent = request.kwargs["response_format"]["json_schema"]
        assert sent["name"] == EXTRACTION_SCHEMA_NAME
        assert sent["schema"] == body
    else:
        assert request.structured is not None
        assert request.structured.schema["title"] == EXTRACTION_SCHEMA_NAME
    assert json.loads(result["output_text"]) == ANSWER


def _caching_env(**values: Any) -> CallEnv:
    return helpers.env(prompt_caching=PromptCaching(enabled=True, **values))


def _caching_off() -> CallEnv:
    return helpers.env(prompt_caching=PromptCaching(enabled=False))


@pytest.mark.parametrize(
    ("ttl", "control"),
    [("5m", {"type": "ephemeral"}), ("1h", {"type": "ephemeral", "ttl": "1h"})],
)
def test_anthropic_caches_the_system_prompt_when_caching_is_on(
    monkeypatch: pytest.MonkeyPatch, ttl: str, control: dict[str, str]
) -> None:
    env = _caching_env(anthropic_ttl=ttl)
    extractor, script = _extractor(monkeypatch, [ANSWER], SPECS["anthropic"], env=env)

    _page(extractor)

    (block,) = _system(script.requests[0])
    assert block["type"] == "text"
    assert block["cache_control"] == control
    assert "The JSON schema:" in block["text"]


@pytest.mark.parametrize(
    ("spec", "env", "cached"),
    [
        (SPECS["openrouter"], _caching_env(), True),
        (model_spec("openai/gpt-5.4", "openrouter"), _caching_env(), False),
        (SPECS["openrouter"], _caching_env(openrouter_cache_control=False), False),
        (SPECS["openrouter"], _caching_off(), False),
        (SPECS["anthropic"], _caching_off(), False),
        (SPECS["anthropic"], helpers.env(), True),
        (SPECS["openrouter"], helpers.env(), True),
    ],
)
def test_cache_control_marks_only_cacheable_system_prompts(
    monkeypatch: pytest.MonkeyPatch, spec: ModelSpec, env: CallEnv, cached: bool
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER], spec, env=env)

    _page(extractor)

    system = _system(script.requests[0])
    assert isinstance(system, list) is cached
    if cached:
        assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_the_response_dict_and_the_role_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    env = helpers.env()
    answer = message(json.dumps(ANSWER), response_metadata={"model_name": "m"})
    extractor, _ = _extractor(monkeypatch, [answer], env=env)

    result = _page(extractor)

    assert result == {
        "output_text": json.dumps(ANSWER),
        "usage": usage_dict(100, 20, 120),
        "metadata": {"model_name": "m"},
    }
    assert env.usage.get("extract") == Usage(100, 20, 120)


@pytest.mark.parametrize(
    ("provider", "fields", "expected"),
    [
        pytest.param(
            "anthropic",
            {
                "response_metadata": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 80,
                        "cache_creation_input_tokens": 5,
                    }
                }
            },
            usage_dict(185, 20, 205, cached=80),
            id="anthropic-cache",
        ),
        pytest.param(
            "anthropic",
            {
                "response_metadata": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 90,
                    }
                }
            },
            usage_dict(190, 20, 210),
            id="anthropic-zero-read",
        ),
        pytest.param(
            "openai",
            {
                "response_metadata": {
                    "token_usage": {
                        "prompt_tokens": 300,
                        "completion_tokens": 50,
                        "total_tokens": 350,
                        "prompt_tokens_details": {"cached_tokens": 200},
                    }
                },
                "usage_metadata": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
            usage_dict(1, 1, 2),
            id="usage-metadata-first",
        ),
        pytest.param(
            "openai",
            {
                "usage_metadata": {
                    "input_tokens": 300,
                    "output_tokens": 50,
                    "total_tokens": 0,
                    "input_token_details": {"cache_read": 200},
                }
            },
            usage_dict(300, 50, 350, cached=200),
            id="usage-metadata",
        ),
        pytest.param(
            "google",
            {
                "response_metadata": {
                    "usage_metadata": {
                        "prompt_token_count": 7,
                        "candidates_token_count": 3,
                        "total_token_count": 10,
                    }
                }
            },
            usage_dict(7, 3, 10),
            id="google",
        ),
        pytest.param(
            "openai",
            {},
            usage_dict(0, 0, 0),
            id="none",
        ),
    ],
)
def test_usage_record(
    provider: str, fields: dict[str, Any], expected: dict[str, int]
) -> None:
    assert usage_record(AIMessage(content="x", **fields), provider) == expected


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(ANSWER),
        f"```json\n{json.dumps(ANSWER)}\n```",
        f"Here is the page:\n\n{json.dumps(ANSWER)}",
    ],
    ids=["json", "fenced", "preamble"],
)
def test_a_prompted_answer_is_kept_as_received(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    extractor, _ = _extractor(monkeypatch, [text], model_spec(response_format="json"))

    result = _page(extractor)

    assert result["output_text"] == text
    assert "error" not in result


@pytest.mark.parametrize(
    "text",
    ['{"entries": [{"person": "cut', "Plain page text.", "42"],
    ids=["malformed", "text", "number"],
)
def test_a_prompted_answer_without_an_object_fails_the_chunk(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    extractor, _ = _extractor(monkeypatch, [text], model_spec(response_format="json"))

    result = _page(extractor)

    assert result["output_text"] == ""
    assert result["error"].startswith(NO_OBJECT)
    assert result["error_type"] == INVALID_JSON


def test_a_model_without_vision_fails_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER], model_spec("o3-mini"))

    result = _page(extractor)

    assert script.requests == []
    assert result["output_text"] == ""
    assert "does not support vision" in result["error"]


def test_a_model_without_vision_still_extracts_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, script = _extractor(monkeypatch, [ANSWER], model_spec("o3-mini"))

    result = _text(extractor)

    assert len(script.requests) == 1
    assert "error" not in result


def test_an_api_error_after_the_ladder_is_an_error_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, _ = _extractor(monkeypatch, [StatusError(401)])

    result = _page(extractor)

    assert result == {
        "output_text": "",
        "usage": usage_dict(0, 0, 0),
        "error": "status 401",
    }


@pytest.mark.parametrize(
    "answer",
    [
        message(
            "{}",
            response_metadata={
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
        ),
        message("{}", additional_kwargs={"refusal": "I cannot help."}),
        message('{"entries": [', response_metadata={"stop_reason": "max_tokens"}),
        message('{"entries": [', response_metadata={"finish_reason": "MAX_TOKENS"}),
        message(
            '{"entries": [',
            response_metadata={
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        ),
    ],
    ids=["content-filter", "refusal", "anthropic-max", "google-max", "openai-max"],
)
def test_a_stopped_schema_answer_fails_the_chunk_after_one_call(
    monkeypatch: pytest.MonkeyPatch, answer: AIMessage
) -> None:
    retry = RetrySettings(validation_attempts=3)
    extractor, script = _extractor(
        monkeypatch, [answer, ANSWER], env=helpers.env(retry)
    )

    result = _page(extractor)

    assert len(script.requests) == 1
    assert result["output_text"] == ""
    assert result["error"]
    assert result["usage"]["total_tokens"] == 120


def test_a_truncated_json_answer_fails_the_chunk_after_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truncated = message(
        '{"contains_no_content_of_requested_type": false, "entries": '
        '[{"person": "Ann"}, {"person": "Bob"}, {"person": "Ca',
        response_metadata={
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    )
    retry = RetrySettings(validation_attempts=3)
    extractor, script = _extractor(
        monkeypatch,
        [truncated, ANSWER],
        model_spec(response_format="json"),
        env=helpers.env(retry),
    )

    result = _page(extractor)

    assert len(script.requests) == 1
    assert result["output_text"] == ""
    assert result["error"]


def test_invalid_json_is_asked_again_within_the_validation_attempts(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2))
    missing = {"entries": []}
    extractor, script = _extractor(
        monkeypatch, [missing, ANSWER], model_spec(response_format="json"), env=env
    )

    result = _page(extractor)

    assert len(script.requests) == 2
    assert json.loads(result["output_text"]) == ANSWER
    assert env.usage.get("extract") == Usage(200, 40, 240)


def test_an_unparsed_schema_answer_fails_after_the_validation_attempts(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2))
    extractor, script = _extractor(
        monkeypatch, ["not json"], SPECS["anthropic"], env=env
    )

    result = _page(extractor)

    assert len(script.requests) == 2
    assert result["output_text"] == ""
    assert result["error_type"] == INVALID_JSON


def test_an_empty_answer_is_returned_without_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, script = _extractor(monkeypatch, [""])

    result = _page(extractor)

    assert len(script.requests) == 1
    assert result["output_text"] == ""
    assert "error" not in result


def _low_input(request: FakeRequest) -> AIMessage:
    usage = {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}
    return message(json.dumps(ANSWER), usage_metadata=usage)


def test_an_image_answer_billed_below_the_input_minimum_is_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2, min_input_tokens=50))
    extractor, script = _extractor(monkeypatch, [_low_input, ANSWER], env=env)

    result = _page(extractor)

    assert len(script.requests) == 2
    assert json.loads(result["output_text"]) == ANSWER
    assert env.usage.get("extract") == Usage(140, 40, 180)


def test_the_input_minimum_is_bounded_by_the_validation_attempts(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2, min_input_tokens=50))
    extractor, script = _extractor(monkeypatch, [_low_input], env=env)

    result = _page(extractor)

    assert len(script.requests) == 2
    assert "below minimum threshold (50)" in result["error"]


def test_the_input_minimum_does_not_apply_to_text_chunks(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2, min_input_tokens=50))
    extractor, script = _extractor(monkeypatch, [_low_input], env=env)

    result = _text(extractor)

    assert len(script.requests) == 1
    assert "error" not in result


def test_a_malformed_sdk_response_is_retried_as_invalid_output(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2))
    shape_error = TypeError("'NoneType' object is not iterable")
    extractor, script = _extractor(monkeypatch, [shape_error, ANSWER], env=env)

    result = _page(extractor)

    assert len(script.requests) == 2
    assert json.loads(result["output_text"]) == ANSWER


def _timeout() -> httpx.ReadTimeout:
    return httpx.ReadTimeout("read timed out")


def test_timeouts_stop_at_the_timeout_attempts(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(
        RetrySettings(max_attempts=8, timeout_attempts=2), clock=sleeps.clock
    )
    extractor, script = _extractor(monkeypatch, [_timeout()], env=env)

    result = _page(extractor)

    assert len(script.requests) == 2
    assert "read timed out" in result["error"]


def test_connection_errors_keep_the_full_attempt_limit(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(
        RetrySettings(max_attempts=4, timeout_attempts=1), clock=sleeps.clock
    )
    wrapped = RuntimeError("APIConnectionError")
    wrapped.__cause__ = httpx.ConnectError("connection refused")
    extractor, script = _extractor(monkeypatch, [wrapped], env=env)

    _page(extractor)

    assert len(script.requests) == 4


def test_a_server_retry_after_raises_the_wait(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    error = StatusError(429)
    error.response = httpx.Response(429, headers={"retry-after": "30"})  # type: ignore[attr-defined]
    env = helpers.env(clock=sleeps.clock)
    extractor, _ = _extractor(monkeypatch, [error, ANSWER], env=env)

    _page(extractor)

    assert sleeps[0] >= 30.0


def _hang(monkeypatch: pytest.MonkeyPatch, error: BaseException | None = None) -> None:
    async def invoke(*_args: Any) -> Any:
        if error is not None:
            raise error
        await asyncio.Event().wait()

    monkeypatch.setattr(caller_module, "default_invoke", invoke)


def test_the_page_watchdog_ends_a_chunk_that_runs_too_long(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    env = helpers.env(RetrySettings(page_timeout=0.05))
    extractor, _ = _extractor(monkeypatch, [ANSWER], env=env)
    _hang(monkeypatch)

    with caplog.at_level(logging.ERROR):
        result = _page(extractor, label="page_0007")

    assert result["error"] == (
        "Page 'page_0007' exceeded its 0s wall-clock ceiling across all retry attempts"
    )
    assert "Page watchdog fired for page_0007" in caplog.text
    assert all(attempt.released for attempt in recorder.attempts)


def test_a_timeout_error_of_the_call_is_not_the_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = helpers.env(RetrySettings(page_timeout=60.0, timeout_attempts=1))
    extractor, _ = _extractor(monkeypatch, [ANSWER], env=env)
    _hang(monkeypatch, TimeoutError("socket timed out"))

    result = _page(extractor)

    assert result["error"] == "socket timed out"


def test_a_cancellation_during_a_timeout_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    async def invoke(*_args: Any) -> Any:
        calls.append(1)
        try:
            raise _timeout()
        except httpx.ReadTimeout:
            raise asyncio.CancelledError from None

    extractor, _ = _extractor(monkeypatch, [ANSWER])
    monkeypatch.setattr(caller_module, "default_invoke", invoke)

    with pytest.raises(asyncio.CancelledError):
        _page(extractor)
    assert calls == [1]


def test_page_timeout_zero_turns_the_watchdog_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, _ = _extractor(
        monkeypatch, [ANSWER], env=helpers.env(RetrySettings(page_timeout=0))
    )

    assert extractor.caller.deadline is None
    assert json.loads(_page(extractor)["output_text"]) == ANSWER


def test_the_auto_page_timeout_covers_the_timeout_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor, _ = _extractor(
        monkeypatch, [ANSWER], env=helpers.env(RetrySettings(timeout_attempts=3))
    )

    assert extractor.caller.deadline == 900.0 * 3 + 300.0


def test_retry_logs_name_the_chunk(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps, caplog: pytest.LogCaptureFixture
) -> None:
    env = helpers.env(clock=sleeps.clock)
    extractor, _ = _extractor(monkeypatch, [StatusError(503), ANSWER], env=env)

    with caplog.at_level(logging.WARNING):
        _page(extractor, label="page_0042")
        asyncio.run(extractor.extract_text("Record 9.", label="18-35"))

    assert "Retrying page_0042" in caplog.text


def test_every_llm_call_path_runs_in_an_attempt(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps, tmp_path: Path
) -> None:
    """Page and text calls with a context image, retries included."""
    recorder = RecordingExt().install(monkeypatch)
    env = helpers.env(clock=sleeps.clock)
    extractor, script = _extractor(
        monkeypatch,
        [StatusError(503), ANSWER, StatusError(503), ANSWER],
        env=env,
    )
    extractor.update_context_image(_context_image(tmp_path))

    page = _page(extractor)
    chunk = _text(extractor)

    assert json.loads(page["output_text"]) == ANSWER
    assert json.loads(chunk["output_text"]) == ANSWER
    assert recorder.roles == ["extract"] * 4
    assert [type(error) for error in recorder.errors] == [StatusError, StatusError]
    assert len(recorder.commits) == 2
    assert all(attempt.released for attempt in recorder.attempts)
    assert len(script.requests) == 4
    assert env.usage.get("extract") == Usage(200, 40, 240)
