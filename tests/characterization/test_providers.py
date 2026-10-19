"""Characterization of the provider classes, model kwargs and image messages.

Each case runs ``extract`` through ``run_tool``, twice in fresh roots, and
compares the second run with ``golden/providers/<case>/``. Case names follow
``<source>_<provider>[_variant]``: ``text`` is a twelve-record ledger,
``image`` one marked PNG page and ``pdf`` a two-page scanned PDF. Every
provider case passes the same overrides (reasoning effort, verbosity, max
output tokens and service tier), so ``requests.json`` records which of them
reach each provider class and in which form. Image cases record the image
block shapes: ``detail`` for OpenAI, the Anthropic base64 source and the
Gemini and OpenRouter forms.

The custom-endpoint cases route ``ChatOpenAI`` to ``base_url`` with the key
from the configured variable. The endpoint's ``supports_vision`` and
``supports_schema`` settings decide what it gets: with both off, a text
input is extracted with prompted JSON, and an image input is refused with
exit code 2 before any call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.characterization.adapter import (
    CUSTOM_BASE_URL,
    RunResult,
    custom_endpoint,
    deep_merge,
    run_tool,
)
from tests.characterization.conftest import CaseRoot, GoldenCheck
from tests.characterization.fakes import CUSTOM_KEY_ENV, FakeLLM, page_unit
from tests.characterization.golden import compare, snapshot
from tests.characterization.inputs import (
    make_image,
    make_scanned_pdf,
    make_text_file,
)

pytestmark = pytest.mark.integration

RECORDS = 12
PDF_PAGES = 2
OVERRIDES: Mapping[str, Any] = {
    "reasoning_effort": "low",
    "verbosity": "low",
    "max_output_tokens": 8192,
    "service_tier": "flex",
}
"""Model overrides every provider case passes."""

MODELS: Mapping[str, str] = {
    "openai": "gpt-5-mini",
    "anthropic": "claude-sonnet-4-5",
    "google": "gemini-2.5-flash",
    "openrouter": "anthropic/claude-sonnet-4.5",
}

CONSTRUCTORS: Mapping[str, str] = {
    "openai": "ChatOpenAI",
    "anthropic": "ChatAnthropic",
    "google": "ChatGoogleGenerativeAI",
    "openrouter": "ChatOpenAI",
    "custom": "ChatOpenAI",
}

IMAGE_BLOCK_TYPES: Mapping[str, str] = {
    "openai": "image_url",
    "anthropic": "image",
    "google": "image",
    "openrouter": "image_url",
    "custom": "image_url",
}
"""Type of the page block in a visual request, per provider."""

CUSTOM_CAPABILITIES_OFF = {
    "extraction_model": {
        "custom_endpoint": {
            "capabilities": {
                "supports_vision": False,
                "supports_structured_output": False,
            }
        }
    }
}

Check = Callable[[RunResult], None]


@dataclass(frozen=True)
class ProviderCase:
    """One provider case; ``flags`` and ``config`` go to ``run_tool``."""

    name: str
    provider: str
    source: str
    flags: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    check: Check | None = None


def _text(root: Path) -> Path:
    return make_text_file(root / "docs" / "ledger.txt", RECORDS)


def _image(root: Path) -> Path:
    return make_image(root / "scans" / "page.png", 0)


def _pdf(root: Path) -> Path:
    return make_scanned_pdf(root / "docs" / "scan.pdf", PDF_PAGES)


SOURCES: dict[str, Callable[[Path], Path]] = {
    "text": _text,
    "image": _image,
    "pdf": _pdf,
}


def _human_blocks(call: Mapping[str, Any]) -> list[dict[str, Any]]:
    message = call["messages"][-1]
    assert message["role"] == "human"
    return list(message["blocks"])


def _page_blocks(result: RunResult) -> list[dict[str, Any]]:
    return [_human_blocks(call)[-1] for call in result.calls]


def _kwargs(result: RunResult) -> dict[str, Any]:
    (model,) = result.models
    return dict(model["kwargs"])


def _check_openai(result: RunResult) -> None:
    kwargs = _kwargs(result)
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["verbosity"] == "low"
    assert kwargs["max_completion_tokens"] == OVERRIDES["max_output_tokens"]
    assert kwargs["service_tier"] == "flex"
    assert kwargs["use_responses_api"] is True


def _check_openai_image(result: RunResult) -> None:
    _check_openai(result)
    assert {block.get("detail") for block in _page_blocks(result)} == {"low"}


def _check_anthropic(result: RunResult) -> None:
    kwargs = _kwargs(result)
    assert kwargs["thinking"]["type"] == "enabled"
    assert "temperature" not in kwargs
    assert "service_tier" not in kwargs
    for call in result.calls:
        assert call["structured"] is not None


def _check_anthropic_image(result: RunResult) -> None:
    _check_anthropic(result)
    for block in _page_blocks(result):
        assert block["source_type"] == "base64"


def _check_google(result: RunResult) -> None:
    kwargs = _kwargs(result)
    assert "thinking_budget" in kwargs
    assert "service_tier" not in kwargs


def _check_google_pdf(result: RunResult) -> None:
    _check_google(result)
    assert [call["unit"] for call in result.calls] == [
        page_unit(index) for index in range(PDF_PAGES)
    ]


def _check_openrouter(result: RunResult) -> None:
    kwargs = _kwargs(result)
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert "reasoning" in kwargs["extra_body"]
    assert "service_tier" not in kwargs


def _check_custom(result: RunResult) -> None:
    kwargs = _kwargs(result)
    assert kwargs["base_url"] == CUSTOM_BASE_URL
    assert kwargs["api_key"] == f"<env:{CUSTOM_KEY_ENV}>"
    for call in result.calls:
        assert call["structured"] is not None
        assert call["structured"]["method"] == "json_schema"


def _check_custom_prompted(result: RunResult) -> None:
    kwargs = _kwargs(result)
    assert kwargs["base_url"] == CUSTOM_BASE_URL
    assert [call["structured"] for call in result.calls] == [None] * len(result.calls)


CASES: tuple[ProviderCase, ...] = (
    ProviderCase("text_openai", "openai", "text", check=_check_openai),
    ProviderCase(
        "image_openai",
        "openai",
        "image",
        flags={"image_detail": "low"},
        check=_check_openai_image,
    ),
    ProviderCase("text_anthropic", "anthropic", "text", check=_check_anthropic),
    ProviderCase("image_anthropic", "anthropic", "image", check=_check_anthropic_image),
    ProviderCase("text_google", "google", "text", check=_check_google),
    ProviderCase("pdf_google", "google", "pdf", check=_check_google_pdf),
    ProviderCase("text_openrouter", "openrouter", "text", check=_check_openrouter),
    ProviderCase("pdf_openrouter", "openrouter", "pdf", check=_check_openrouter),
    ProviderCase(
        "text_custom",
        "custom",
        "text",
        config={"model_config": custom_endpoint()},
        check=_check_custom,
    ),
    ProviderCase(
        "image_custom",
        "custom",
        "image",
        config={"model_config": custom_endpoint()},
        check=_check_custom,
    ),
    ProviderCase(
        "text_custom_capabilities_off",
        "custom",
        "text",
        config={"model_config": deep_merge(custom_endpoint(), CUSTOM_CAPABILITIES_OFF)},
        check=_check_custom_prompted,
    ),
)

REFUSED: tuple[ProviderCase, ...] = (
    ProviderCase(
        "image_custom_capabilities_off",
        "custom",
        "image",
        config={"model_config": deep_merge(custom_endpoint(), CUSTOM_CAPABILITIES_OFF)},
    ),
)
"""Cases the plan refuses with exit code 2 before any call."""


def _run(case: ProviderCase, llm: FakeLLM, root: Path) -> RunResult:
    flags: dict[str, Any] = {**OVERRIDES, **dict(case.flags)}
    if case.provider != "custom":
        flags.update(provider=case.provider, model=MODELS[case.provider])
    return run_tool(
        "extract",
        llm=llm,
        root=root,
        config=case.config,
        input=SOURCES[case.source](root),
        output=root / "out",
        **flags,
    )


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_provider_case(
    case: ProviderCase,
    fake_llm: FakeLLM,
    case_root: CaseRoot,
    golden: GoldenCheck,
) -> None:
    first = snapshot(_run(case, fake_llm, case_root("run1")))
    result = _run(case, fake_llm, case_root("run2"))
    problems = compare(first, snapshot(result))
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)

    assert result.exit_code == 0, result.stdout + result.stderr
    assert [model["constructor"] for model in result.models] == [
        CONSTRUCTORS[case.provider]
    ]
    assert {model["provider"] for model in result.models} == {case.provider}
    assert result.calls
    if case.source == "text":
        assert all(
            block["type"] == "text"
            for call in result.calls
            for block in _human_blocks(call)
        )
    else:
        assert {block["type"] for block in _page_blocks(result)} == {
            IMAGE_BLOCK_TYPES[case.provider]
        }
    if case.check is not None:
        case.check(result)
    golden(f"providers/{case.name}", result)


@pytest.mark.parametrize("case", REFUSED, ids=[case.name for case in REFUSED])
def test_refused_provider_case(
    case: ProviderCase,
    fake_llm: FakeLLM,
    case_root: CaseRoot,
    golden: GoldenCheck,
) -> None:
    first = snapshot(_run(case, fake_llm, case_root("run1")))
    result = _run(case, fake_llm, case_root("run2"))
    problems = compare(first, snapshot(result))
    assert not problems, "two identical runs differ:\n" + "\n".join(problems)

    assert result.exit_code == 2, result.console
    assert result.calls == []
    assert result.models == []
    assert result.created == []
    assert "image input" in result.stderr
    golden(f"providers/{case.name}", result)


def test_case_names_are_unique() -> None:
    names = [case.name for case in (*CASES, *REFUSED)]
    assert len(names) == len(set(names))
