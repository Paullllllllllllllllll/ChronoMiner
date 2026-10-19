"""The factory passes the constructor kwargs of the recorded golden runs.

Each expected mapping is the ``models`` section of a characterization golden
(``tests/characterization/golden/llm/<case>/requests.json``), copied here as
a literal with the output limit of 128,000 tokens; most specs below set that
limit, and unset limits are checked too; the OpenAI cases take the current
default model and effort, while o3-deep-research and the Anthropic and Google
cases set the recorded effort high. :data:`CORRECTED` lists the limits
the family table sets instead. Objects such as the hooked httpx clients
appear by their type name, as in the goldens.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from chronominer.llm.client import build_chat_model
from chronominer.spec import ModelSpec
from tests.llm.helpers import custom_spec, model_spec, settings

HARNESS_LIMIT = 128000

_HTTPX_TIMEOUT = {
    "httpx.Timeout": {"connect": 10.0, "pool": 30.0, "read": 900.0, "write": 30.0}
}
_OPENAI_CLIENTS = {
    "http_async_client": "<_DefaultAsyncHttpxClient>",
    "http_client": "<_DefaultHttpxClient>",
}
_GOOGLE_CLIENT_ARGS = {
    "event_hooks": {"response": ["<function>"]},
    "transport": "<EitherTransport>",
}
_ALL_SAMPLERS_OFF = {
    "frequency_penalty": None,
    "presence_penalty": None,
    "temperature": None,
    "top_p": None,
}

PDF_OPENAI_SCHEMA_MARKDOWN: dict[str, Any] = {
    "constructor": "ChatOpenAI",
    "kwargs": {
        "api_key": "<env:OPENAI_API_KEY>",
        "disabled_params": _ALL_SAMPLERS_OFF,
        "frequency_penalty": None,
        **_OPENAI_CLIENTS,
        "max_completion_tokens": 128000,
        "max_retries": 0,
        "model": "gpt-6-luna",
        "presence_penalty": None,
        "reasoning": {"effort": "xhigh"},
        "service_tier": "default",
        "temperature": None,
        "timeout": _HTTPX_TIMEOUT,
        "top_p": None,
        "use_responses_api": True,
        "verbosity": "medium",
    },
    "provider": "openai",
}

PDF_OPENAI_SCHEMA_OVERRIDES: dict[str, Any] = {
    "constructor": "ChatOpenAI",
    "kwargs": {
        "api_key": "<env:OPENAI_API_KEY>",
        "disabled_params": _ALL_SAMPLERS_OFF,
        "frequency_penalty": None,
        **_OPENAI_CLIENTS,
        "max_completion_tokens": 4096,
        "max_retries": 0,
        "model": "gpt-6-luna",
        "presence_penalty": None,
        "reasoning": {"effort": "low"},
        "service_tier": "priority",
        "temperature": None,
        "timeout": _HTTPX_TIMEOUT,
        "top_p": None,
        "use_responses_api": True,
        "verbosity": "concise",
    },
    "provider": "openai",
}

PDF_OPENAI_JSON_NO_STRUCTURED_OUTPUTS: dict[str, Any] = {
    "constructor": "ChatOpenAI",
    "kwargs": {
        key: value
        for key, value in {
            **PDF_OPENAI_SCHEMA_MARKDOWN["kwargs"],
            "max_completion_tokens": 128000,
            "model": "o3-deep-research",
            "reasoning": {"effort": "high"},
        }.items()
        # No documented verbosity for o3-deep-research; the default tier
        # goes to every OpenAI reasoning model.
        if key != "verbosity"
    },
    "provider": "openai",
}

PDF_ANTHROPIC_SCHEMA: dict[str, Any] = {
    "constructor": "ChatAnthropic",
    "kwargs": {
        "api_key": "<env:ANTHROPIC_API_KEY>",
        "max_retries": 0,
        "max_tokens": 65536,
        "model": "claude-sonnet-4-5",
        "thinking": {"budget_tokens": 16384, "type": "enabled"},
        "timeout": 900.0,
    },
    "provider": "anthropic",
}

IMAGES_GOOGLE_SCHEMA: dict[str, Any] = {
    "constructor": "ChatGoogleGenerativeAI",
    "kwargs": {
        "client_args": _GOOGLE_CLIENT_ARGS,
        "google_api_key": "<env:GOOGLE_API_KEY>",
        "max_retries": 0,
        "max_tokens": 65536,
        "model": "gemini-3-flash",
        "temperature": 1.0,
        "thinking_level": "high",
        "timeout": 900.0,
        "top_k": None,
        "top_p": 1.0,
    },
    "provider": "google",
}

PDF_OPENROUTER_SCHEMA: dict[str, Any] = {
    "constructor": "ChatOpenAI",
    "kwargs": {
        "api_key": "<env:OPENROUTER_API_KEY>",
        "base_url": "https://openrouter.ai/api/v1",
        "default_headers": {
            "HTTP-Referer": "https://github.com/ChronoMiner",
            "X-Title": "ChronoMiner",
        },
        "disabled_params": {"frequency_penalty": None, "presence_penalty": None},
        "extra_body": {"reasoning": {"max_tokens": 32000}},
        **_OPENAI_CLIENTS,
        "max_retries": 0,
        "max_tokens": 128000,
        "model": "anthropic/claude-sonnet-4.5",
        "temperature": 1.0,
        "timeout": _HTTPX_TIMEOUT,
        "top_p": 1.0,
    },
    "provider": "openrouter",
}

PDF_CUSTOM_SCHEMA: dict[str, Any] = {
    "constructor": "ChatOpenAI",
    "kwargs": {
        "api_key": "<env:CM_FAKE_CUSTOM_API_KEY>",
        "base_url": "https://fake-endpoint.invalid/v1/",
        **_OPENAI_CLIENTS,
        "max_retries": 0,
        "max_tokens": 128000,
        "model": "org/fake-model",
        "temperature": 1.0,
        "timeout": _HTTPX_TIMEOUT,
        "top_p": 1.0,
    },
    "provider": "custom",
}

# Output limits the family table sets, per golden case: claude-sonnet-4.5
# takes its family limit of 64,000 output tokens, and o3-deep-research without
# a set limit takes its family limit of 100,000.
CORRECTED: dict[str, dict[str, Any]] = {
    "pdf_anthropic_schema": {"max_tokens": 64000},
    "pdf_openai_json_no_structured_outputs": {"max_completion_tokens": 100000},
}

CASES: list[tuple[str, dict[str, Any], ModelSpec]] = [
    ("pdf_openai_schema_markdown", PDF_OPENAI_SCHEMA_MARKDOWN, model_spec()),
    (
        "pdf_openai_schema_markdown",
        PDF_OPENAI_SCHEMA_MARKDOWN,
        model_spec(max_output_tokens=HARNESS_LIMIT),
    ),
    (
        "pdf_openai_schema_overrides",
        PDF_OPENAI_SCHEMA_OVERRIDES,
        model_spec(
            reasoning_effort="low",
            verbosity="concise",
            max_output_tokens=4096,
            service_tier="priority",
        ),
    ),
    (
        "pdf_openai_json_no_structured_outputs",
        PDF_OPENAI_JSON_NO_STRUCTURED_OUTPUTS,
        model_spec("o3-deep-research", reasoning_effort="high"),
    ),
    (
        "pdf_anthropic_schema",
        PDF_ANTHROPIC_SCHEMA,
        model_spec(
            "claude-sonnet-4-5",
            "anthropic",
            reasoning_effort="high",
            max_output_tokens=HARNESS_LIMIT,
        ),
    ),
    (
        "pdf_anthropic_schema",
        PDF_ANTHROPIC_SCHEMA,
        model_spec("claude-sonnet-4-5", "anthropic", reasoning_effort="high"),
    ),
    (
        "images_google_schema",
        IMAGES_GOOGLE_SCHEMA,
        model_spec(
            "gemini-3-flash",
            "google",
            reasoning_effort="high",
            max_output_tokens=HARNESS_LIMIT,
        ),
    ),
    (
        "images_google_schema",
        IMAGES_GOOGLE_SCHEMA,
        model_spec("gemini-3-flash", "google", reasoning_effort="high"),
    ),
    (
        "pdf_openrouter_schema",
        PDF_OPENROUTER_SCHEMA,
        model_spec(
            "anthropic/claude-sonnet-4.5",
            "openrouter",
            max_output_tokens=HARNESS_LIMIT,
        ),
    ),
    (
        "pdf_custom_schema",
        PDF_CUSTOM_SCHEMA,
        custom_spec(max_output_tokens=HARNESS_LIMIT),
    ),
]

_KEY_VARIABLES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "CM_FAKE_CUSTOM_API_KEY",
)


def _jsonable(value: Any) -> Any:
    """Return *value* in the form the golden files record."""
    if isinstance(value, httpx.Timeout):
        return {
            "httpx.Timeout": {
                "connect": value.connect,
                "pool": value.pool,
                "read": value.read,
                "write": value.write,
            }
        }
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str) and value.startswith("secret-"):
        return f"<env:{value.removeprefix('secret-')}>"
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return f"<{type(value).__name__}>"


@pytest.fixture
def constructions(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every chat model construction instead of building it."""
    for name in _KEY_VARIABLES:
        monkeypatch.setenv(name, f"secret-{name}")
    recorded: list[dict[str, Any]] = []

    def recorder(constructor: str) -> Any:
        def construct(**kwargs: Any) -> str:
            recorded.append({"constructor": constructor, "kwargs": kwargs})
            return constructor

        return construct

    monkeypatch.setattr("langchain_openai.ChatOpenAI", recorder("ChatOpenAI"))
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", recorder("ChatAnthropic"))
    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        recorder("ChatGoogleGenerativeAI"),
    )
    return recorded


@pytest.mark.parametrize(
    ("case", "golden", "spec"), CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)]
)
def test_constructor_kwargs_match_the_golden_models_section(
    constructions: list[dict[str, Any]],
    case: str,
    golden: dict[str, Any],
    spec: ModelSpec,
) -> None:
    build_chat_model(spec, settings())

    (built,) = constructions
    expected = {**golden["kwargs"], **CORRECTED.get(case, {})}
    assert built["constructor"] == golden["constructor"]
    assert spec.provider == golden["provider"]
    assert _jsonable(built["kwargs"]) == expected


@pytest.mark.parametrize("case", sorted(CORRECTED))
def test_only_the_listed_limits_differ_from_the_goldens(
    constructions: list[dict[str, Any]], case: str
) -> None:
    _name, golden, spec = next(entry for entry in CASES if entry[0] == case)
    build_chat_model(spec, settings())

    kwargs = _jsonable(constructions[0]["kwargs"])
    differing = {
        key: value
        for key, value in kwargs.items()
        if golden["kwargs"].get(key) != value
    }
    assert differing == CORRECTED[case]
